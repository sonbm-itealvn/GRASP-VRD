"""Optional RegionCLIP adapter for YOLO proposals.

This module deliberately does not vendor RegionCLIP/Detectron2. Set
REGIONCLIP_ROOT to a checkout of the official Microsoft RegionCLIP repository
and provide the official checkpoint plus concept embeddings.

The adapter accepts YOLO boxes, extracts RegionCLIP region features with the
official recognition backbone and RoIAlign path, then matches them against
concept embeddings. A low score is rejected by the caller instead of silently
being treated as a valid open-vocabulary label.
"""
from __future__ import annotations

import os
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class RegionCLIPUnavailable(RuntimeError):
    """Raised when the optional official RegionCLIP runtime is not available."""


@dataclass
class RegionCLIPConfig:
    enabled: bool
    root: Path
    config_file: Path
    weights: Path
    text_emb: Optional[Path]
    labels_file: Optional[Path]
    text_emb_dim: int = 640
    pooler_resolution: int = 18
    device: str = "cuda"


@dataclass
class RegionCLIPPrediction:
    label: Optional[str]
    score: float
    index: int
    accepted: bool


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: Optional[str] = None) -> Optional[Path]:
    value = os.getenv(name, default)
    if not value:
        return None
    return Path(value).expanduser().resolve()


class RegionCLIPClassifier:
    """Run official RegionCLIP recognition on externally supplied boxes."""

    def __init__(self, cfg: RegionCLIPConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = None
        self.concept_emb = None
        self.labels: List[str] = []
        self._load()

    @classmethod
    def from_env(cls) -> Optional["RegionCLIPClassifier"]:
        root = _path_env("REGIONCLIP_ROOT")
        weights = _path_env("REGIONCLIP_WEIGHTS")
        config_file = _path_env(
            "REGIONCLIP_CONFIG",
            str(root / "configs/LVISv1-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_custom_img.yaml")
            if root else None,
        )
        text_emb = _path_env("REGIONCLIP_TEXT_EMB")
        labels_file = _path_env("REGIONCLIP_LABELS")
        enabled = _env_bool("REGIONCLIP_ENABLED", bool(root and weights and text_emb))
        if not enabled:
            return None
        missing = [name for name, path in {
            "REGIONCLIP_ROOT": root,
            "REGIONCLIP_CONFIG": config_file,
            "REGIONCLIP_WEIGHTS": weights,
            "REGIONCLIP_TEXT_EMB": text_emb,
        }.items() if path is None or not path.exists()]
        if missing:
            raise RegionCLIPUnavailable(
                "RegionCLIP enabled but required paths are missing: " + ", ".join(missing)
            )
        return cls(RegionCLIPConfig(
            enabled=True,
            root=root,
            config_file=config_file,
            weights=weights,
            text_emb=text_emb,
            labels_file=labels_file,
            text_emb_dim=int(os.getenv("REGIONCLIP_TEXT_EMB_DIM", "640")),
            pooler_resolution=int(os.getenv("REGIONCLIP_POOLER_RESOLUTION", "18")),
            device=os.getenv("REGIONCLIP_DEVICE", "cuda"),
        ))

    def _load(self) -> None:
        root = str(self.cfg.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            # These imports register the official RegionCLIP Detectron2 components.
            import detectron2  # noqa: F401
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultTrainer
            import detectron2.modeling.backbone.clip_backbone  # noqa: F401
            import detectron2.modeling.meta_arch.clip_rcnn  # noqa: F401
            import detectron2.modeling.roi_heads.clip_roi_heads  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise RegionCLIPUnavailable(
                "Cannot import official RegionCLIP/Detectron2. Install its dependencies "
                f"and set REGIONCLIP_ROOT. Original error: {exc}"
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(str(self.cfg.config_file))
        cfg.defrost()
        cfg.MODEL.WEIGHTS = str(self.cfg.weights)
        cfg.MODEL.CLIP.TEXT_EMB_PATH = str(self.cfg.text_emb)
        cfg.MODEL.CLIP.TEXT_EMB_DIM = self.cfg.text_emb_dim
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = self.cfg.pooler_resolution
        cfg.MODEL.CLIP.CROP_REGION_TYPE = "GT"
        cfg.freeze()

        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model).resume_or_load(str(self.cfg.weights), resume=False)
        self.model.to(self.device)
        self.model.eval()

        emb = torch.load(self.cfg.text_emb, map_location=self.device, weights_only=False)
        if isinstance(emb, dict):
            for key in ("embeddings", "text_emb", "concept_emb", "cls_emb"):
                if key in emb:
                    emb = emb[key]
                    break
        if not torch.is_tensor(emb):
            raise RegionCLIPUnavailable(f"Unsupported concept embedding format: {self.cfg.text_emb}")
        self.concept_emb = emb.float().to(self.device)
        self.concept_emb = self.concept_emb / self.concept_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.labels = self._load_labels(self.cfg.labels_file, self.concept_emb.shape[0])

    @staticmethod
    def _load_labels(path: Optional[Path], count: int) -> List[str]:
        if path is None or not path.exists():
            raise RegionCLIPUnavailable(
                "REGIONCLIP_LABELS is required so embedding rows can be mapped to labels "
                "without guessing the category order."
            )
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            categories = data.get("categories", data) if isinstance(data, dict) else data
            if not isinstance(categories, list):
                raise RegionCLIPUnavailable(f"Unsupported labels JSON format: {path}")
            labels = [item["name"].strip() for item in sorted(categories, key=lambda x: x.get("id", 0))]
        else:
            labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(labels) != count:
            raise RegionCLIPUnavailable(
                f"REGIONCLIP_LABELS has {len(labels)} labels but embeddings have {count} rows"
            )
        return labels

    @staticmethod
    def _make_input(image_rgb: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("RegionCLIP expects an RGB image with shape HxWx3")
        image = torch.as_tensor(np.ascontiguousarray(image_rgb.transpose(2, 0, 1)))
        return {"image": image.to(device), "height": image_rgb.shape[0], "width": image_rgb.shape[1]}

    @torch.no_grad()
    def classify(self, image_rgb: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]) -> List[RegionCLIPPrediction]:
        """Classify YOLO boxes using official RegionCLIP region features.

        Score is a softmax probability over the configured concept pool, using
        REGIONCLIP_TEMPERATURE (default 0.01). It is a confidence heuristic,
        not a calibrated probability; tune the rejection threshold on validation data.
        """
        if not boxes:
            return []
        from detectron2.structures import Boxes

        inputs = [self._make_input(image_rgb, self.device)]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        proposal_boxes = [Boxes(boxes_tensor)]
        box_features = self.model.roi_heads._shared_roi_transform(
            [features[f] for f in self.model.roi_heads.in_features],
            proposal_boxes,
            self.model.backbone.layer4,
        )
        region_features = self.model.backbone.attnpool(box_features)
        region_features = region_features / region_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        temperature = float(os.getenv("REGIONCLIP_TEMPERATURE", "0.01"))
        probabilities = torch.softmax((region_features @ self.concept_emb.T) / temperature, dim=-1)
        scores, indices = probabilities.max(dim=-1)
        reject_threshold = float(os.getenv("REGIONCLIP_REJECT_THRESHOLD", "0.30"))
        predictions = []
        for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
            accepted = score >= reject_threshold
            predictions.append(RegionCLIPPrediction(
                label=self.labels[index] if accepted else None,
                score=float(score),
                index=int(index),
                accepted=accepted,
            ))
        return predictions

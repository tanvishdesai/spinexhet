"""Model factory."""

from __future__ import annotations

from typing import Any

from spine_xnet.models.baselines import BaselineClassifier, ConceptBottleneckBaseline
from spine_xnet.models.spinexnet import SpineXNet


def build_model(config: dict[str, Any], num_concepts: int | None = None):
    model_cfg = dict(config.get("model", config))
    model_type = model_cfg.pop("type", "spinexnet").lower()
    if num_concepts is not None:
        model_cfg["num_concepts"] = num_concepts

    if model_type in {"baseline", "blackbox", "classifier"}:
        allowed = {
            "backbone",
            "pretrained",
            "num_classes",
            "meta_dim",
            "hidden_dim",
            "dropout",
            "ordinal",
        }
        return BaselineClassifier(**{k: v for k, v in model_cfg.items() if k in allowed})
    if model_type in {"cbm", "concept_bottleneck"}:
        allowed = {
            "backbone",
            "pretrained",
            "num_concepts",
            "num_classes",
            "meta_dim",
            "hidden_dim",
            "dropout",
            "ordinal",
        }
        return ConceptBottleneckBaseline(**{k: v for k, v in model_cfg.items() if k in allowed})
    if model_type in {"spinexnet", "spine_xnet"}:
        return SpineXNet(**model_cfg)
    raise ValueError(f"Unknown model type: {model_type}")

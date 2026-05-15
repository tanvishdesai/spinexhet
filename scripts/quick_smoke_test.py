from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.models import build_model
from spine_xnet.models.losses import SpineLoss


def main() -> None:
    cfg = {
        "model": {
            "type": "spinexnet",
            "backbone": "convnext_tiny",
            "pretrained": False,
            "num_classes": 3,
            "num_concepts": 15,
            "num_prototypes": 10,
            "meta_dim": 8,
            "hidden_dim": 64,
            "dropout": 0.1,
            "residual_features": True,
            "residual_dim": 32,
            "ordinal": True,
        }
    }
    model = build_model(cfg, num_concepts=15)
    batch = {
        "image": torch.randn(2, 3, 224, 224),
        "condition_idx": torch.tensor([0, 1]),
        "level_idx": torch.tensor([2, 3]),
        "label": torch.tensor([0, 2]),
        "concept_targets": torch.rand(2, 15),
        "concept_mask": torch.ones(2, 15),
    }
    outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
    loss = SpineLoss()(outputs, batch, model=model)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["concept_probs"].shape == (2, 15)
    assert outputs["prototype_maps"].shape[1] == 10
    assert torch.isfinite(loss["total"])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()


"""CBM Concept Intervention Demo.

Demonstrates the practical clinical utility of the CBM by showing that
correcting individual concept predictions at test time can fix classification
errors — an advantage unique to ante-hoc interpretable models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBM concept intervention analysis.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    data_cfg = cfg.get("data", {})
    if args.crop_root is not None:
        data_cfg["crop_root"] = str(args.crop_root)
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = str(args.cache_dir)

    manifest = args.manifest or Path(data_cfg["manifest"])
    fold = args.fold if args.fold is not None else int(data_cfg.get("fold", 0))
    _, val_df = load_manifest_for_fold(manifest, fold=fold, fold_col=data_cfg.get("fold_col", "fold"))
    val_df = val_df.head(args.max_samples).reset_index(drop=True)

    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)
    dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_concepts=len(concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Verify model is a CBM with concept_head + classifier
    if not hasattr(model, "concept_head") or not hasattr(model, "classifier"):
        raise ValueError("Model must be a ConceptBottleneckBaseline for intervention analysis.")

    rows = []
    total, correct_before, correct_after_any = 0, 0, 0

    for batch in tqdm(loader, desc="intervention"):
        batch = to_device(batch, device)
        label = int(batch["label"].item())
        concept_targets = batch["concept_targets"].squeeze(0).cpu().numpy()

        # Forward pass: get concept predictions and final logits
        features = model.backbone(batch["image"])
        meta = model.meta(batch["condition_idx"], batch["level_idx"])
        concept_logits = model.concept_head(torch.cat([features, meta], dim=1))
        concept_probs = torch.sigmoid(concept_logits)

        # Original prediction (through concepts only, no residual)
        fused = torch.cat([concept_probs, meta], dim=1)
        logits = model.classifier(fused)
        pred_before = int(logits.argmax(dim=1).item())

        total += 1
        if pred_before == label:
            correct_before += 1

        # Try intervening on each concept
        any_fixed = False
        for c_idx, c_name in enumerate(concept_columns):
            gt_val = float(concept_targets[c_idx])
            pred_val = float(concept_probs[0, c_idx].item())

            # Replace concept c_idx with ground truth
            intervened = concept_probs.clone()
            intervened[0, c_idx] = gt_val
            fused_int = torch.cat([intervened, meta], dim=1)
            logits_int = model.classifier(fused_int)
            pred_after = int(logits_int.argmax(dim=1).item())

            was_wrong = pred_before != label
            now_correct = pred_after == label
            flipped = pred_before != pred_after

            if was_wrong and now_correct:
                any_fixed = True

            rows.append({
                "sample_id": batch["sample_id"][0],
                "condition": batch["condition"][0],
                "label": label,
                "pred_before": pred_before,
                "concept_name": c_name,
                "concept_pred": round(pred_val, 4),
                "concept_gt": round(gt_val, 4),
                "concept_error": round(abs(pred_val - gt_val), 4),
                "pred_after_intervention": pred_after,
                "was_wrong": was_wrong,
                "now_correct": now_correct,
                "prediction_flipped": flipped,
            })

        if any_fixed:
            correct_after_any += 1

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "concept_intervention.csv", index=False)

    wrong = total - correct_before
    summary = {
        "total_samples": total,
        "correct_before_intervention": correct_before,
        "accuracy_before": round(correct_before / max(total, 1), 4),
        "wrong_samples": wrong,
        "fixable_by_any_concept": correct_after_any,
        "fix_rate": round(correct_after_any / max(wrong, 1), 4),
    }

    # Per-concept fix rates
    if not df.empty:
        concept_summary = (
            df[df["was_wrong"]]
            .groupby("concept_name")["now_correct"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "fixes", "count": "attempts"})
        )
        concept_summary["fix_rate"] = (concept_summary["fixes"] / concept_summary["attempts"]).round(4)
        summary["per_concept_fix_rate"] = concept_summary["fix_rate"].to_dict()

    write_json(summary, args.output_dir / "intervention_summary.json")
    print(summary)


if __name__ == "__main__":
    main()

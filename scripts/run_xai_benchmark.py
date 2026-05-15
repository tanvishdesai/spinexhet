from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.evaluation.xai import (
    attribution_agreement,
    deletion_insertion_auc,
    proxy_roi_alignment,
    run_attribution_method,
)
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XAI faithfulness and agreement benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=["gradcam", "gradcam++", "integrated_gradients", "prototype"])
    parser.add_argument("--faithfulness-steps", type=int, default=20)
    parser.add_argument("--crop-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


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
    model = build_model(cfg, num_concepts=len(dataset.concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows = []
    skipped: dict[str, str] = {}
    for batch in tqdm(loader, desc="xai"):
        batch = to_device(batch, device)
        condition = batch["condition"][0] if "condition" in batch else None
        with torch.no_grad():
            outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
            target = int(torch.softmax(outputs["logits"], dim=1).argmax(dim=1).item())

        attributions = {}
        for method in args.methods:
            try:
                attr = run_attribution_method(
                    method,
                    model,
                    batch["image"],
                    int(batch["condition_idx"].item()),
                    int(batch["level_idx"].item()),
                    target,
                )
                attributions[method] = attr
                deletion = deletion_insertion_auc(
                    model,
                    batch["image"],
                    attr,
                    int(batch["condition_idx"].item()),
                    int(batch["level_idx"].item()),
                    target,
                    mode="deletion",
                    steps=args.faithfulness_steps,
                )
                insertion = deletion_insertion_auc(
                    model,
                    batch["image"],
                    attr,
                    int(batch["condition_idx"].item()),
                    int(batch["level_idx"].item()),
                    target,
                    mode="insertion",
                    steps=args.faithfulness_steps,
                )
                rows.append(
                    {
                        "sample_id": batch["sample_id"][0],
                        "method": method,
                        "target": target,
                        "condition": condition,
                        "deletion_auc": deletion,
                        "insertion_auc": insertion,
                        "proxy_roi_alignment": proxy_roi_alignment(attr, condition=condition),
                    }
                )
            except Exception as exc:
                skipped[method] = str(exc)

        if len(attributions) > 1:
            agreement = attribution_agreement(attributions)
            rows.append({"sample_id": batch["sample_id"][0], "method": "agreement", **agreement})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.output_dir / "xai_benchmark.csv", index=False)
    if not result_df.empty:
        summary = result_df.groupby("method").mean(numeric_only=True).to_dict(orient="index")
    else:
        summary = {}
    write_json({"summary": summary, "skipped": skipped}, args.output_dir / "xai_summary.json")
    print({"summary": summary, "skipped": skipped})


if __name__ == "__main__":
    main()

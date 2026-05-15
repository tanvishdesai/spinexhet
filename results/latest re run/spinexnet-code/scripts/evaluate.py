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
from spine_xnet.evaluation.metrics import classification_metrics, per_group_metrics, softmax_np
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--split", choices=["val", "train", "site_test"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


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
    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)

    if args.split in {"train", "val"}:
        train_df, val_df = load_manifest_for_fold(manifest, fold=fold, fold_col=data_cfg.get("fold_col", "fold"))
        df = train_df if args.split == "train" else val_df
    else:
        full = pd.read_csv(manifest)
        df = full.loc[full["site_split"] == "test"].reset_index(drop=True)

    dataset = RSNACropDataset(
        df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("training", {}).get("batch_size", 16)) * 2, shuffle=False, num_workers=int(cfg.get("training", {}).get("num_workers", 2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_concepts=len(dataset.concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows = []
    logits_all = []
    labels_all = []
    for batch in tqdm(loader, desc="eval"):
        batch = to_device(batch, device)
        outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
        logits = outputs["logits"].detach().float().cpu().numpy()
        probs = softmax_np(logits)
        labels = batch["label"].detach().cpu().numpy()
        logits_all.append(logits)
        labels_all.append(labels)
        for i, sample_id in enumerate(batch["sample_id"]):
            rows.append(
                {
                    "sample_id": sample_id,
                    "study_id": int(batch["study_id"][i].detach().cpu()),
                    "condition": batch["condition"][i],
                    "level": batch["level"][i],
                    "label": int(labels[i]),
                    "prob_normal_mild": float(probs[i, 0]),
                    "prob_moderate": float(probs[i, 1]),
                    "prob_severe": float(probs[i, 2]),
                    "pred": int(probs[i].argmax()),
                }
            )

    pred_df = pd.DataFrame(rows)
    probs = pred_df[["prob_normal_mild", "prob_moderate", "prob_severe"]].values
    labels = pred_df["label"].values
    metrics = classification_metrics(labels, probs)
    metrics.update(per_group_metrics(labels, probs, pred_df["condition"].values, "condition"))
    metrics.update(per_group_metrics(labels, probs, pred_df["level"].values, "level"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output_dir / f"predictions_{args.split}.csv", index=False)
    write_json(metrics, args.output_dir / f"metrics_{args.split}.json")
    print(metrics)


if __name__ == "__main__":
    main()

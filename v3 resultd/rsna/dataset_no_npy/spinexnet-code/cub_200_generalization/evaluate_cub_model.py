"""Evaluate a trained CUB-200 checkpoint on one fold's validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from cub_utils import (
    CUBDataset,
    cub_classification_metrics,
    load_model_from_checkpoint,
    make_cub_split,
    read_cub_metadata,
    softmax_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CUB-200 checkpoint.")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cub-root", type=Path, default=None)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = read_cub_metadata(args.cub_root)
    split = make_cub_split(df, fold=args.fold, n_splits=args.n_folds)
    dataset = CUBDataset(split.val, image_size=args.image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, _ = load_model_from_checkpoint(args.checkpoint, model_name=args.model, device=device)
    model.eval()

    logits_all = []
    labels_all = []
    rows = []
    for batch in tqdm(loader, desc="cub eval"):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(image).detach().float().cpu().numpy()
        probs = softmax_np(logits)
        pred = probs.argmax(axis=1)
        logits_all.append(logits)
        labels_all.append(label.detach().cpu().numpy())
        for i, sample_id in enumerate(batch["sample_id"]):
            top5 = np.argsort(probs[i])[-5:][::-1]
            rows.append(
                {
                    "sample_id": sample_id,
                    "class_name": batch["class_name"][i],
                    "label": int(label[i].detach().cpu()),
                    "pred": int(pred[i]),
                    "prob_pred": float(probs[i, pred[i]]),
                    "top5": " ".join(map(str, top5.tolist())),
                    "correct": bool(pred[i] == int(label[i].detach().cpu())),
                }
            )

    logits_arr = np.concatenate(logits_all)
    labels_arr = np.concatenate(labels_all)
    metrics = cub_classification_metrics(labels_arr, softmax_np(logits_arr))
    pd.DataFrame(rows).to_csv(args.output_dir / "predictions_val.csv", index=False)
    with (args.output_dir / "metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()

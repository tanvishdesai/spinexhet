"""Train one CUB-200 model/fold for the generalization experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
    build_cub_model,
    cub_classification_metrics,
    make_cub_split,
    read_cub_metadata,
    save_checkpoint,
    softmax_np,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CUB-200 model fold.")
    p.add_argument("--model", required=True)
    p.add_argument("--cub-root", type=Path, default=None)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--time-limit-minutes", type=float, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--progress-bar", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    progress_bar: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = 0.0
    n_seen = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    for batch in tqdm(loader, desc="train", leave=False, disable=not progress_bar):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits = model(image)
            loss = criterion(logits, label)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = image.shape[0]
        loss_meter += float(loss.detach().cpu()) * bs
        n_seen += bs
        correct += int((logits.detach().argmax(dim=1) == label).sum().cpu())
    return {"loss": loss_meter / max(n_seen, 1), "accuracy": correct / max(n_seen, 1)}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_bar: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    logits_all = []
    labels_all = []
    rows = []
    for batch in tqdm(loader, desc="val", leave=False, disable=not progress_bar):
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
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    df = read_cub_metadata(args.cub_root)
    split = make_cub_split(df, fold=args.fold, n_splits=args.n_folds, seed=args.seed)
    train_ds = CUBDataset(split.train, image_size=args.image_size, train=True)
    val_ds = CUBDataset(split.val, image_size=args.image_size, train=False)
    val_batch_size = args.val_batch_size or args.batch_size * 2

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cub_model(args.model, num_classes=200, pretrained=True).to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    cfg = {
        "model_name": args.model,
        "num_classes": 200,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "image_size": args.image_size,
        "seed": args.seed,
    }

    print(
        {
            "event": "cub_training_start",
            "model": args.model,
            "fold": args.fold,
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "device": str(device),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        flush=True,
    )

    start = time.monotonic()
    best_acc = -math.inf
    stale = 0
    history = []
    best_pred_df: pd.DataFrame | None = None
    best_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, amp_enabled, args.progress_bar
        )
        val_metrics, pred_df = evaluate(model, val_loader, device, args.progress_bar)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(row, flush=True)

        is_best = val_metrics["accuracy"] > best_acc
        if is_best:
            best_acc = val_metrics["accuracy"]
            stale = 0
            best_metrics = val_metrics
            best_pred_df = pred_df
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, val_metrics, cfg)
            pred_df.to_csv(args.output_dir / "predictions_val.csv", index=False)
            with (args.output_dir / "metrics_val.json").open("w", encoding="utf-8") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            stale += 1

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, val_metrics, cfg)
        if stale >= args.patience:
            print({"event": "early_stop", "epoch": epoch, "best_accuracy": best_acc}, flush=True)
            break
        if args.time_limit_minutes is not None and (time.monotonic() - start) / 60.0 >= args.time_limit_minutes:
            print({"event": "time_limit_stop", "epoch": epoch, "best_accuracy": best_acc}, flush=True)
            break

    if best_pred_df is None:
        metrics, pred_df = evaluate(model, val_loader, device, args.progress_bar)
        pred_df.to_csv(args.output_dir / "predictions_val.csv", index=False)
        with (args.output_dir / "metrics_val.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        best_metrics = metrics

    print({"event": "cub_training_done", "model": args.model, "fold": args.fold, **best_metrics}, flush=True)


if __name__ == "__main__":
    main()

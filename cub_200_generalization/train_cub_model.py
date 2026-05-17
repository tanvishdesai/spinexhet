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
    get_cub_training_recipe,
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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--drop-path-rate", type=float, default=None)
    p.add_argument("--label-smoothing", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--time-limit-minutes", type=float, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-model-recipe", dest="use_model_recipe", action="store_false")
    p.add_argument("--progress-bar", action="store_true")
    p.set_defaults(use_model_recipe=True)
    return p.parse_args()


def resolve_training_args(args: argparse.Namespace) -> argparse.Namespace:
    recipe = get_cub_training_recipe(args.model) if args.use_model_recipe else {}
    fallback = get_cub_training_recipe("__default__")
    recipe = {**fallback, **recipe}
    for key in [
        "epochs",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "drop_path_rate",
        "label_smoothing",
        "patience",
    ]:
        if getattr(args, key) is None:
            setattr(args, key, recipe.get(key))
    if args.backbone_lr is None:
        args.backbone_lr = recipe.get("backbone_lr")
    if args.head_lr is None:
        args.head_lr = recipe.get("head_lr")
    args.epochs = int(args.epochs)
    args.patience = int(args.patience)
    args.warmup_epochs = int(args.warmup_epochs or 0)
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    args.drop_path_rate = float(args.drop_path_rate or 0.0)
    args.label_smoothing = float(args.label_smoothing or 0.0)
    return args


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
    label_smoothing: float,
    progress_bar: bool,
) -> dict[str, float]:
    model.train()
    loss_meter = 0.0
    n_seen = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
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


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.backbone_lr is None and args.head_lr is None:
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    head_params: list[torch.nn.Parameter] = []
    if isinstance(classifier, torch.nn.Module):
        head_params = [p for p in classifier.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    if not head_params or not backbone_params:
        return torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(args.backbone_lr or args.lr)},
            {"params": head_params, "lr": float(args.head_lr or args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    warmup_epochs = int(args.warmup_epochs or 0)
    if warmup_epochs > 0 and int(args.epochs) > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.epochs) - warmup_epochs, 1),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))


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
    args = resolve_training_args(parse_args())
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
    model = build_cub_model(
        args.model,
        num_classes=200,
        pretrained=True,
        drop_path_rate=float(args.drop_path_rate or 0.0),
    ).to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    cfg = {
        "model_name": args.model,
        "num_classes": 200,
        "fold": args.fold,
        "n_folds": args.n_folds,
        "image_size": args.image_size,
        "seed": args.seed,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "drop_path_rate": args.drop_path_rate,
        "label_smoothing": args.label_smoothing,
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
            "lr": args.lr,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "drop_path_rate": args.drop_path_rate,
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
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            float(args.label_smoothing),
            args.progress_bar,
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

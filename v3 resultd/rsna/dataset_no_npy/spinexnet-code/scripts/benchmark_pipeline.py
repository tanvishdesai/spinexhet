from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader and train-step throughput.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--crop-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--model", action="store_true", help="Also benchmark forward/backward, not only data loading.")
    parser.add_argument("--no-pretrained", action="store_true")
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

    train_df, _ = load_manifest_for_fold(args.manifest, fold=args.fold, fold_col=data_cfg.get("fold_col", "fold"))
    dataset = RSNACropDataset(
        train_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=True,
        concept_columns=data_cfg.get("concept_columns", DEFAULT_CONCEPTS),
        crop_root=data_cfg.get("crop_root"),
        manifest_path=args.manifest,
        require_crops=True,
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    optimizer = None
    if args.model:
        if args.no_pretrained:
            cfg["model"]["pretrained"] = False
        model = build_model(cfg, num_concepts=len(dataset.concept_columns)).to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()

    start = time.perf_counter()
    data_time = 0.0
    step_time = 0.0
    n = 0
    last = start
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for n, batch in enumerate(loader, start=1):
        now = time.perf_counter()
        data_time += now - last
        if model is not None and optimizer is not None:
            batch = to_device(batch, device)
            if device.type == "cuda":
                batch["image"] = batch["image"].to(memory_format=torch.channels_last)
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                outputs = model(batch["image"], batch["condition_idx"], batch["level_idx"])
                loss = torch.nn.functional.cross_entropy(outputs["logits"], batch["label"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_time += time.perf_counter() - step_start
        if n >= args.batches:
            break
        last = time.perf_counter()

    elapsed = time.perf_counter() - start
    print(
        {
            "batches": n,
            "batch_size": args.batch_size,
            "samples": n * args.batch_size,
            "elapsed_sec": round(elapsed, 2),
            "sec_per_batch": round(elapsed / max(n, 1), 4),
            "data_sec_per_batch": round(data_time / max(n, 1), 4),
            "model_sec_per_batch": round(step_time / max(n, 1), 4) if args.model else None,
            "estimated_epoch_hours": round((elapsed / max(n, 1)) * len(loader) / 3600.0, 3),
            "train_batches_per_epoch": len(loader),
            "device": str(device),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()


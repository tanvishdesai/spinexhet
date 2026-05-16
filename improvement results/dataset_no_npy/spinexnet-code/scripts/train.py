from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base, save_config
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.models import build_model
from spine_xnet.training.trainer import Trainer
from spine_xnet.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline or SpineXNet model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--crop-root", type=Path, default=None, help="Directory containing crop PNGs, e.g. /kaggle/input/.../crops_224.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory from build_image_cache.py containing images_uint8.npy.")
    parser.add_argument("--require-crops", action="store_true", help="Fail fast if crop PNGs cannot be resolved.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-prototypes", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--time-limit-minutes", type=float, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-minutes", type=float, default=None)
    parser.add_argument("--heartbeat-every-steps", type=int, default=None)
    parser.add_argument("--heartbeat-every-minutes", type=float, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--save-val-predictions", action="store_true")
    parser.add_argument("--no-data-parallel", action="store_true")
    parser.add_argument("--progress-bar", action="store_true", help="Enable tqdm progress bars. Disabled by default to avoid huge Kaggle outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config_with_base(args.config)
    seed_everything(int(cfg.get("seed", 42)))

    data_cfg = cfg.setdefault("data", {})
    train_cfg = cfg.setdefault("training", {})
    if args.crop_root is not None:
        data_cfg["crop_root"] = str(args.crop_root)
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = str(args.cache_dir)
    if args.image_size is not None:
        data_cfg["image_size"] = int(args.image_size)
    if args.require_crops:
        data_cfg["require_crops"] = True
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.val_batch_size is not None:
        train_cfg["val_batch_size"] = int(args.val_batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    if args.grad_accum is not None:
        train_cfg["grad_accum"] = int(args.grad_accum)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.max_train_batches is not None:
        train_cfg["max_train_batches"] = int(args.max_train_batches)
    if args.max_val_batches is not None:
        train_cfg["max_val_batches"] = int(args.max_val_batches)
    if args.validate_every is not None:
        train_cfg["validate_every"] = int(args.validate_every)
    if args.skip_validation:
        train_cfg["skip_validation"] = True
    if args.time_limit_minutes is not None:
        train_cfg["time_limit_minutes"] = float(args.time_limit_minutes)
    if args.checkpoint_every_steps is not None:
        train_cfg["checkpoint_every_steps"] = int(args.checkpoint_every_steps)
    if args.checkpoint_every_minutes is not None:
        train_cfg["checkpoint_every_minutes"] = float(args.checkpoint_every_minutes)
    if args.heartbeat_every_steps is not None:
        train_cfg["heartbeat_every_steps"] = int(args.heartbeat_every_steps)
    if args.heartbeat_every_minutes is not None:
        train_cfg["heartbeat_every_minutes"] = float(args.heartbeat_every_minutes)
    if args.prefetch_factor is not None:
        train_cfg["prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers:
        train_cfg["persistent_workers"] = True
    if args.save_val_predictions:
        train_cfg["save_val_predictions"] = True
    if args.no_data_parallel:
        train_cfg["data_parallel"] = False
    if args.progress_bar:
        train_cfg["progress_bar"] = True
    if args.num_prototypes is not None:
        cfg.setdefault("model", {})["num_prototypes"] = int(args.num_prototypes)

    manifest = args.manifest or Path(data_cfg["manifest"])
    fold = args.fold if args.fold is not None else int(data_cfg.get("fold", 0))
    output_dir = args.output_dir or Path(cfg.get("output_dir", "outputs")) / cfg.get("experiment_name", args.config.stem) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    concept_columns = data_cfg.get("concept_columns", DEFAULT_CONCEPTS)
    train_df, val_df = load_manifest_for_fold(manifest, fold=fold, fold_col=data_cfg.get("fold_col", "fold"))
    train_dataset = RSNACropDataset(
        train_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=True,
        concept_columns=concept_columns,
        crop_size=int(data_cfg.get("crop_size", 224)),
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )
    val_dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=concept_columns,
        crop_size=int(data_cfg.get("crop_size", 224)),
        crop_root=data_cfg.get("crop_root"),
        manifest_path=manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )

    model = build_model(cfg, num_concepts=len(train_dataset.concept_columns))
    trainer = Trainer(model, train_dataset, val_dataset, cfg, output_dir=output_dir)
    result = trainer.fit(resume=args.resume or train_cfg.get("resume"))
    print(result)


if __name__ == "__main__":
    main()

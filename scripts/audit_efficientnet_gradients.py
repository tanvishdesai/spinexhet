"""Audit EfficientNet Integrated Gradients vs GradientSHAP on RSNA folds.

This is a targeted V4 repair utility. It does not train anything; it loads an
existing EfficientNet checkpoint, recomputes IG and GradientSHAP for a small
validation subset, and reports whether the two maps are actually identical,
baseline-sensitive, or merely similarly weak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.evaluation.metrics import normalize_map, spearman_corr, topk_iou
from spine_xnet.evaluation.xai import (
    deletion_insertion_auc,
    gradient_shap,
    integrated_gradients,
)
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything, to_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit EfficientNet IG/GradientSHAP behavior.")
    p.add_argument("--config", type=Path, default=Path("configs/baselines/efficientnet_b4.yaml"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--faithfulness-steps", type=int, default=20)
    p.add_argument("--baseline-modes", nargs="+", default=["mean", "black", "gray"])
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--examples", type=int, default=6)
    return p.parse_args()


def unnormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    arr = image[0].detach() * std + mean
    arr = arr.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2]):
        return arr[..., 0]
    return arr


def input_gradient_stats(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
) -> dict[str, float]:
    x = image.detach().clone().requires_grad_(True)
    cond = torch.tensor([condition_idx], dtype=torch.long, device=image.device)
    level = torch.tensor([level_idx], dtype=torch.long, device=image.device)
    logits = model(x, cond, level)["logits"][:, target].sum()
    model.zero_grad(set_to_none=True)
    logits.backward()
    grad = x.grad.detach()
    return {
        "input_grad_l1": float(grad.abs().mean().cpu()),
        "input_grad_l2": float(torch.sqrt(torch.mean(grad.square())).cpu()),
        "input_grad_max_abs": float(grad.abs().max().cpu()),
    }


def map_stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    m = normalize_map(arr)
    return {
        f"{prefix}_mean": float(m.mean()),
        f"{prefix}_std": float(m.std()),
        f"{prefix}_max": float(m.max()),
        f"{prefix}_nonzero_fraction": float((np.abs(m) > 1e-8).mean()),
    }


def save_example(
    output_dir: Path,
    fold: int,
    sample_id: str,
    image_arr: np.ndarray,
    mode_to_maps: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if not mode_to_maps:
        return
    rows = len(mode_to_maps)
    fig, axes = plt.subplots(rows, 4, figsize=(11, 2.8 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row_idx, (mode, (ig_map, gs_map)) in enumerate(mode_to_maps.items()):
        axes[row_idx, 0].imshow(image_arr, cmap="gray")
        axes[row_idx, 0].set_title(f"{sample_id}\ninput")
        axes[row_idx, 1].imshow(image_arr, cmap="gray")
        axes[row_idx, 1].imshow(normalize_map(ig_map), cmap="magma", alpha=0.55)
        axes[row_idx, 1].set_title(f"IG ({mode})")
        axes[row_idx, 2].imshow(image_arr, cmap="gray")
        axes[row_idx, 2].imshow(normalize_map(gs_map), cmap="magma", alpha=0.55)
        axes[row_idx, 2].set_title(f"GradientSHAP ({mode})")
        axes[row_idx, 3].imshow(np.abs(normalize_map(ig_map) - normalize_map(gs_map)), cmap="viridis")
        axes[row_idx, 3].set_title("|IG - GS|")
        for col in range(4):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
    plt.tight_layout()
    fig.savefig(output_dir / f"fold_{fold}_{sample_id}_ig_vs_gradshap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config_with_base(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    data_cfg = dict(cfg.get("data", {}))
    if args.crop_root is not None:
        data_cfg["crop_root"] = str(args.crop_root)
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = str(args.cache_dir)

    _, val_df = load_manifest_for_fold(args.manifest, fold=args.fold, fold_col=data_cfg.get("fold_col", "fold"))
    val_df = val_df.head(args.max_samples).reset_index(drop=True)
    dataset = RSNACropDataset(
        val_df,
        image_size=int(data_cfg.get("image_size", 224)),
        train=False,
        concept_columns=data_cfg.get("concept_columns", DEFAULT_CONCEPTS),
        crop_root=data_cfg.get("crop_root"),
        manifest_path=args.manifest,
        require_crops=bool(data_cfg.get("require_crops", False)),
        cache_dir=data_cfg.get("cache_dir"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_concepts=len(dataset.concept_columns)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    rows: list[dict] = []
    for sample_idx, batch in enumerate(loader):
        batch = to_device(batch, device)
        image = batch["image"]
        sample_id = str(batch["sample_id"][0])
        condition_idx = int(batch["condition_idx"].item())
        level_idx = int(batch["level_idx"].item())
        label = int(batch["label"].item())
        with torch.no_grad():
            logits = model(image, batch["condition_idx"], batch["level_idx"])["logits"]
            target = int(torch.softmax(logits, dim=1).argmax(dim=1).item())

        grad_stats = input_gradient_stats(model, image, condition_idx, level_idx, target)
        mode_to_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for baseline_mode in args.baseline_modes:
            ig_map = integrated_gradients(
                model,
                image,
                condition_idx,
                level_idx,
                target,
                baseline_mode=baseline_mode,
            )
            gs_map = gradient_shap(
                model,
                image,
                condition_idx,
                level_idx,
                target,
                baseline_mode=baseline_mode,
            )
            ig_ins = deletion_insertion_auc(
                model,
                image,
                ig_map,
                condition_idx,
                level_idx,
                target,
                mode="insertion",
                steps=args.faithfulness_steps,
            )
            gs_ins = deletion_insertion_auc(
                model,
                image,
                gs_map,
                condition_idx,
                level_idx,
                target,
                mode="insertion",
                steps=args.faithfulness_steps,
            )
            row = {
                "fold": args.fold,
                "sample_id": sample_id,
                "label": label,
                "target": target,
                "baseline_mode": baseline_mode,
                "map_spearman": spearman_corr(ig_map, gs_map),
                "top20_iou": topk_iou(ig_map, gs_map),
                "mean_abs_diff": float(np.mean(np.abs(normalize_map(ig_map) - normalize_map(gs_map)))),
                "max_abs_diff": float(np.max(np.abs(normalize_map(ig_map) - normalize_map(gs_map)))),
                "pixel_identical_atol_1e_6": bool(np.allclose(ig_map, gs_map, atol=1e-6)),
                "ig_insertion_auc": ig_ins,
                "gradient_shap_insertion_auc": gs_ins,
                "insertion_auc_delta_gs_minus_ig": gs_ins - ig_ins,
                **grad_stats,
                **map_stats("ig", ig_map),
                **map_stats("gradient_shap", gs_map),
            }
            rows.append(row)
            if sample_idx < args.examples:
                mode_to_maps[baseline_mode] = (ig_map, gs_map)
        if sample_idx < args.examples:
            save_example(args.output_dir, args.fold, sample_id, unnormalize_image(image), mode_to_maps)

    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / f"efficientnet_gradient_audit_fold_{args.fold}.csv", index=False)
    summary = (
        df.groupby("baseline_mode")
        .agg(
            samples=("sample_id", "nunique"),
            map_spearman_mean=("map_spearman", "mean"),
            top20_iou_mean=("top20_iou", "mean"),
            mean_abs_diff_mean=("mean_abs_diff", "mean"),
            identical_fraction=("pixel_identical_atol_1e_6", "mean"),
            ig_insertion_auc_mean=("ig_insertion_auc", "mean"),
            gradient_shap_insertion_auc_mean=("gradient_shap_insertion_auc", "mean"),
            insertion_auc_delta_mean=("insertion_auc_delta_gs_minus_ig", "mean"),
            input_grad_l2_mean=("input_grad_l2", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / f"efficientnet_gradient_audit_summary_fold_{args.fold}.csv", index=False)

    payload = {
        "fold": args.fold,
        "checkpoint": str(args.checkpoint),
        "baseline_modes": args.baseline_modes,
        "all_pixel_identical": bool(df["pixel_identical_atol_1e_6"].all()) if not df.empty else False,
        "summary_csv": str(args.output_dir / f"efficientnet_gradient_audit_summary_fold_{args.fold}.csv"),
    }
    with (args.output_dir / f"efficientnet_gradient_audit_fold_{args.fold}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

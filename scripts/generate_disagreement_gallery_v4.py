"""Generate a Figure-1 style qualitative disagreement gallery for RSNA.

The script computes saliency maps directly from checkpoints instead of relying
on cached attribution maps. This makes it useful after a targeted V4 repair run:
you can load DenseNet/ViT/DeiT checkpoints, pick a low-agreement sample from
existing XAI CSVs when available, and save a publication-facing visual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spine_xnet.config import load_config_with_base
from spine_xnet.constants import DEFAULT_CONCEPTS
from spine_xnet.data.dataset import RSNACropDataset, load_manifest_for_fold
from spine_xnet.evaluation.consensus import topk_faithfulness_consensus_map
from spine_xnet.evaluation.metrics import normalize_map
from spine_xnet.evaluation.xai import deletion_insertion_auc, run_attribution_method
from spine_xnet.models import build_model
from spine_xnet.utils import seed_everything


DEFAULT_METHODS = [
    "gradcam",
    "gradcam++",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
    "guided_backprop",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate RSNA qualitative XAI disagreement gallery.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--crop-root", type=Path, default=None)
    p.add_argument("--xai-root", type=Path, default=None, help="Optional existing XAI root used to pick sample IDs.")
    p.add_argument("--sample-id", default=None, help="Optional fixed sample ID. Overrides xai-root selection.")
    p.add_argument("--models", nargs="+", default=["densenet121", "vit_small"])
    p.add_argument("--configs", nargs="+", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--max-search-samples", type=int, default=300)
    p.add_argument("--gradient-baseline-mode", default="mean", choices=["mean", "black", "gray", "white", "blur"])
    p.add_argument("--faithfulness-steps", type=int, default=12)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def unnormalize_image(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
    arr = image[0].detach() * std + mean
    arr = arr.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    if np.allclose(arr[..., 0], arr[..., 1]) and np.allclose(arr[..., 1], arr[..., 2]):
        return arr[..., 0]
    return arr


def pick_sample_id(xai_root: Path | None, fold: int, preferred_model: str) -> str | None:
    if xai_root is None:
        return None
    path = xai_root / f"fold_{fold}" / preferred_model / "agreement_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "mean_spearman" not in df or "sample_id" not in df:
        return None
    df = df.dropna(subset=["mean_spearman"]).sort_values("mean_spearman")
    if df.empty:
        return None
    return str(df.iloc[0]["sample_id"])


def load_dataset_row(args: argparse.Namespace, cfg: dict, sample_id: str | None) -> tuple[RSNACropDataset, int]:
    data_cfg = dict(cfg.get("data", {}))
    if args.crop_root is not None:
        data_cfg["crop_root"] = str(args.crop_root)
    if args.cache_dir is not None:
        data_cfg["cache_dir"] = str(args.cache_dir)
    _, val_df = load_manifest_for_fold(args.manifest, fold=args.fold, fold_col=data_cfg.get("fold_col", "fold"))
    val_df = val_df.head(args.max_search_samples).reset_index(drop=True)
    if sample_id is not None and "sample_id" in val_df:
        match = val_df.index[val_df["sample_id"].astype(str) == str(sample_id)].tolist()
        if match:
            val_df = val_df.iloc[[match[0]]].reset_index(drop=True)
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
    return dataset, 0


def load_model(cfg: dict, checkpoint: Path, num_concepts: int, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg, num_concepts=num_concepts).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


def compute_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    methods: list[str],
    gradient_baseline_mode: str,
    faithfulness_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, float], str]:
    maps: dict[str, np.ndarray] = {}
    scores: dict[str, float] = {}
    skipped: list[str] = []
    for method in methods:
        try:
            attr = run_attribution_method(
                method,
                model,
                image,
                condition_idx,
                level_idx,
                target,
                gradient_baseline_mode=gradient_baseline_mode,
            )
            maps[method] = attr
            scores[method] = deletion_insertion_auc(
                model,
                image,
                attr,
                condition_idx,
                level_idx,
                target,
                mode="insertion",
                steps=faithfulness_steps,
            )
        except Exception as exc:
            skipped.append(f"{method}: {exc}")
    if len(maps) >= 3 and scores:
        try:
            consensus, weights = topk_faithfulness_consensus_map(maps, scores, top_k=3)
            maps["topk_consensus"] = consensus
            scores["topk_consensus"] = deletion_insertion_auc(
                model,
                image,
                consensus,
                condition_idx,
                level_idx,
                target,
                mode="insertion",
                steps=faithfulness_steps,
            )
        except Exception as exc:
            skipped.append(f"topk_consensus: {exc}")
    return maps, scores, "; ".join(skipped)


def plot_gallery(
    output_path: Path,
    input_image: np.ndarray,
    rows: list[dict],
    all_methods: list[str],
    dpi: int,
) -> None:
    n_rows = len(rows)
    n_cols = len(all_methods) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.25 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for row_idx, row in enumerate(rows):
        ax = axes[row_idx, 0]
        ax.imshow(input_image, cmap="gray")
        ax.set_ylabel(row["label"], fontsize=9, fontweight="bold")
        if row_idx == 0:
            ax.set_title("Input", fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for col_idx, method in enumerate(all_methods, start=1):
            ax = axes[row_idx, col_idx]
            ax.imshow(input_image, cmap="gray")
            if method in row["maps"]:
                ax.imshow(normalize_map(row["maps"][method]), cmap="magma", alpha=0.58)
                score = row["scores"].get(method)
                suffix = f"\nIns {score:.3f}" if score is not None else ""
                ax.set_title(method.replace("_", " ").title() + suffix, fontsize=7)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_title(method.replace("_", " ").title(), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.models) != len(args.configs) or len(args.models) != len(args.checkpoints):
        raise ValueError("--models, --configs, and --checkpoints must have the same length.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preferred = "vit_small" if "vit_small" in args.models else args.models[-1]
    sample_id = args.sample_id or pick_sample_id(args.xai_root, args.fold, preferred)
    first_cfg = load_config_with_base(args.configs[0])
    dataset, row_idx = load_dataset_row(args, first_cfg, sample_id)
    batch = dataset[row_idx]
    image = batch["image"].unsqueeze(0).to(device)
    condition_idx = int(batch["condition_idx"])
    level_idx = int(batch["level_idx"])
    input_image = unnormalize_image(image)
    resolved_sample_id = str(batch["sample_id"])

    rows = []
    all_methods: list[str] = list(args.methods)
    if any("vit" in m or "deit" in m for m in args.models):
        all_methods = [*all_methods, "attention_rollout"]
    all_methods = [*all_methods, "topk_consensus"]

    summary_rows = []
    for model_name, cfg_path, ckpt_path in zip(args.models, args.configs, args.checkpoints):
        cfg = load_config_with_base(cfg_path)
        model = load_model(cfg, ckpt_path, num_concepts=len(dataset.concept_columns), device=device)
        with torch.no_grad():
            cond = torch.tensor([condition_idx], dtype=torch.long, device=device)
            level = torch.tensor([level_idx], dtype=torch.long, device=device)
            logits = model(image, cond, level)["logits"]
            target = int(torch.softmax(logits, dim=1).argmax(dim=1).item())

        methods = list(args.methods)
        if "vit" in model_name or "deit" in model_name:
            methods.append("attention_rollout")
        maps, scores, skipped = compute_maps(
            model,
            image,
            condition_idx,
            level_idx,
            target,
            methods,
            args.gradient_baseline_mode,
            args.faithfulness_steps,
        )
        rows.append(
            {
                "label": f"{model_name}\ntarget={target}",
                "maps": maps,
                "scores": scores,
            }
        )
        for method, score in scores.items():
            summary_rows.append(
                {
                    "sample_id": resolved_sample_id,
                    "model": model_name,
                    "target": target,
                    "method": method,
                    "insertion_auc": score,
                    "skipped": skipped,
                }
            )

    output_png = args.output_dir / f"figure1_rsna_disagreement_gallery_fold_{args.fold}_{resolved_sample_id}.png"
    plot_gallery(output_png, input_image, rows, all_methods, args.dpi)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "figure1_rsna_disagreement_gallery_scores.csv", index=False)
    print({"figure": str(output_png), "sample_id": resolved_sample_id})


if __name__ == "__main__":
    main()

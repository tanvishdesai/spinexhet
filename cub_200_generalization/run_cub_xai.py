"""Run attribution agreement on CUB-200 validation images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from cub_utils import CUBDataset, load_model_from_checkpoint, make_cub_split, read_cub_metadata
from spine_xnet.evaluation.metrics import normalize_map
from spine_xnet.evaluation.stats import bootstrap_ci
from spine_xnet.evaluation.xai import attention_rollout, attribution_agreement, find_last_conv


DEFAULT_METHODS = [
    "gradcam",
    "gradcam++",
    "integrated_gradients",
    "gradient_shap",
    "occlusion",
    "guided_backprop",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CUB-200 XAI agreement benchmark.")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cub-root", type=Path, default=None)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--faithfulness-steps", type=int, default=20)
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--save-maps", action="store_true")
    p.add_argument("--enable-attention-rollout", action="store_true")
    return p.parse_args()


class ImageOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class SpineCompatibleImageModel(torch.nn.Module):
    """Adapter so the shared ViT attention rollout can call image-only models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.backbone = model

    def forward(self, image: torch.Tensor, condition_idx: torch.Tensor, level_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.model(image)}


def _to_numpy_map(attr: torch.Tensor, image_hw: tuple[int, int]) -> np.ndarray:
    if attr.ndim == 4:
        attr = attr.detach().abs().mean(dim=1, keepdim=True)
        attr = F.interpolate(attr, size=image_hw, mode="bilinear", align_corners=False)
        arr = attr[0, 0].detach().cpu().numpy()
    elif attr.ndim == 3:
        arr = attr[0].detach().cpu().numpy()
    else:
        arr = attr.detach().cpu().numpy()
    return normalize_map(arr)


def integrated_gradients(model: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    from captum.attr import IntegratedGradients

    wrapped = ImageOnlyWrapper(model).eval()
    ig = IntegratedGradients(wrapped)
    attr = ig.attribute(image, baselines=torch.zeros_like(image), target=target, n_steps=32)
    return _to_numpy_map(attr, image.shape[-2:])


def gradient_shap(model: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    from captum.attr import GradientShap

    wrapped = ImageOnlyWrapper(model).eval()
    baseline_dist = torch.randn(50, *image.shape[1:], device=image.device) * 0.001
    attr = GradientShap(wrapped).attribute(
        image,
        baselines=baseline_dist,
        target=target,
        n_samples=50,
        stdevs=0.09,
    )
    return _to_numpy_map(attr, image.shape[-2:])


def guided_backprop(model: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    from captum.attr import GuidedBackprop

    wrapped = ImageOnlyWrapper(model).eval()
    attr = GuidedBackprop(wrapped).attribute(image, target=target)
    return _to_numpy_map(attr, image.shape[-2:])


def occlusion_attribution(model: torch.nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    from captum.attr import Occlusion

    wrapped = ImageOnlyWrapper(model).eval()
    attr = Occlusion(wrapped).attribute(
        image,
        sliding_window_shapes=(3, 32, 32),
        strides=(3, 16, 16),
        target=target,
        baselines=0,
    )
    return _to_numpy_map(attr, image.shape[-2:])


def grad_cam_variant(model: torch.nn.Module, image: torch.Tensor, target: int, variant: str) -> np.ndarray:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam_cls = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}[variant.lower()]
    wrapped = ImageOnlyWrapper(model).eval()
    target_layer = find_last_conv(model)
    with cam_cls(model=wrapped, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(target)])
    return normalize_map(grayscale_cam[0])


def run_attribution_method(
    method: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    target: int,
) -> np.ndarray:
    key = method.lower()
    if key in {"gradcam", "gradcam++"}:
        return grad_cam_variant(model, image, target, key)
    if key in {"integrated_gradients", "ig"}:
        return integrated_gradients(model, image, target)
    if key in {"gradient_shap", "gradientshap"}:
        return gradient_shap(model, image, target)
    if key == "occlusion":
        return occlusion_attribution(model, image, target)
    if key in {"guided_backprop", "guidedbackprop", "gbp"}:
        return guided_backprop(model, image, target)
    if key in {"attention_rollout", "rollout"}:
        compat = SpineCompatibleImageModel(model)
        return attention_rollout(compat, image, 0, 0, target)
    raise ValueError(f"Unsupported CUB attribution method: {method}")


@torch.no_grad()
def target_confidence(model: torch.nn.Module, image: torch.Tensor, target: int) -> float:
    probs = torch.softmax(model(image), dim=1)
    return float(probs[:, target].mean().detach().cpu())


def deletion_insertion_auc(
    model: torch.nn.Module,
    image: torch.Tensor,
    attribution: np.ndarray,
    target: int,
    mode: str,
    steps: int,
) -> float:
    device = image.device
    _, _, h, w = image.shape
    attr = torch.as_tensor(normalize_map(attribution), dtype=torch.float32, device=device)
    attr = F.interpolate(attr[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    order = torch.argsort(attr.flatten(), descending=True)
    total = h * w
    baseline = torch.zeros_like(image)
    scores = []
    fractions = np.linspace(0.0, 1.0, steps + 1)
    for frac in fractions:
        k = int(total * frac)
        mask = torch.zeros(total, dtype=torch.bool, device=device)
        if k > 0:
            mask[order[:k]] = True
        mask = mask.reshape(1, 1, h, w)
        if mode == "deletion":
            perturbed = image.masked_fill(mask, 0.0)
        elif mode == "insertion":
            perturbed = torch.where(mask, image, baseline)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        scores.append(target_confidence(model, perturbed, target))
    return float(np.trapz(scores, fractions))


def _save_df(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {len(rows)} rows to {path}", flush=True)


def _summary(faithfulness_rows: list[dict], agreement_rows: list[dict], skipped: dict[str, str]) -> dict:
    summary: dict = {"skipped": skipped}
    if agreement_rows:
        df = pd.DataFrame(agreement_rows)
        for col in ["mean_spearman", "mean_top20_iou"]:
            vals = df[col].dropna().to_numpy(dtype=float)
            summary[col] = float(vals.mean()) if len(vals) else float("nan")
            if len(vals) >= 2:
                lo, hi = bootstrap_ci(vals)
                summary[f"{col}_ci"] = [round(lo, 4), round(hi, 4)]
    if faithfulness_rows:
        df = pd.DataFrame(faithfulness_rows)
        summary["faithfulness"] = (
            df.groupby("method")[["deletion_auc", "insertion_auc"]]
            .mean()
            .to_dict(orient="index")
        )
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.output_dir / "attribution_maps" if args.save_maps else None
    if maps_dir:
        maps_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = read_cub_metadata(args.cub_root)
    split = make_cub_split(df, fold=args.fold, n_splits=args.n_folds)
    val_df = split.val.head(args.max_samples).reset_index(drop=True)
    dataset = CUBDataset(val_df, image_size=args.image_size, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model, _ = load_model_from_checkpoint(args.checkpoint, model_name=args.model, device=device)
    model.eval()

    methods = list(args.methods)
    if args.enable_attention_rollout and "attention_rollout" not in methods:
        methods.append("attention_rollout")

    faithfulness_rows: list[dict] = []
    agreement_rows: list[dict] = []
    skipped: dict[str, str] = {}
    start = time.monotonic()

    for sample_idx, batch in enumerate(tqdm(loader, desc="cub xai")):
        image = batch["image"].to(device)
        label = int(batch["label"].item())
        sample_id = str(batch["sample_id"][0])
        with torch.no_grad():
            target = int(torch.softmax(model(image), dim=1).argmax(dim=1).item())

        attributions: dict[str, np.ndarray] = {}
        for method in methods:
            try:
                attr = run_attribution_method(method, model, image, target)
                attributions[method] = attr
                if maps_dir:
                    np.save(maps_dir / f"{sample_id}_{method}.npy", attr)
            except Exception as exc:
                if method not in skipped:
                    skipped[method] = str(exc)

        if len(attributions) > 1:
            agreement = attribution_agreement(attributions)
            agreement_rows.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "target": target,
                    **agreement,
                }
            )

        if not args.skip_faithfulness:
            for method, attr in attributions.items():
                try:
                    faithfulness_rows.append(
                        {
                            "sample_id": sample_id,
                            "method": method,
                            "label": label,
                            "target": target,
                            "deletion_auc": deletion_insertion_auc(
                                model, image, attr, target, mode="deletion", steps=args.faithfulness_steps
                            ),
                            "insertion_auc": deletion_insertion_auc(
                                model, image, attr, target, mode="insertion", steps=args.faithfulness_steps
                            ),
                        }
                    )
                except Exception:
                    pass

        if (sample_idx + 1) % 50 == 0:
            print(
                {
                    "event": "cub_xai_progress",
                    "samples": sample_idx + 1,
                    "elapsed_min": round((time.monotonic() - start) / 60.0, 1),
                },
                flush=True,
            )

    _save_df(agreement_rows, args.output_dir / "agreement_metrics.csv")
    _save_df(faithfulness_rows, args.output_dir / "faithfulness_metrics.csv")
    summary = _summary(faithfulness_rows, agreement_rows, skipped)
    payload = {"summary": summary, "args": {k: str(v) for k, v in vars(args).items()}}
    with (args.output_dir / "xai_summary_cub.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(payload)


if __name__ == "__main__":
    main()

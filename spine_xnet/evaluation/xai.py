"""Attribution methods and faithfulness metrics for the XAI benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from spine_xnet.evaluation.metrics import normalize_map, spearman_corr, topk_iou


@dataclass
class AttributionResult:
    method: str
    attribution: np.ndarray
    score: float


class FixedMetaModel(torch.nn.Module):
    """Wrap a model so CAM libraries see a standard image -> logits module."""

    def __init__(self, model: torch.nn.Module, condition_idx: int, level_idx: int) -> None:
        super().__init__()
        self.model = model
        self.condition_idx = int(condition_idx)
        self.level_idx = int(level_idx)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        cond = torch.full((b,), self.condition_idx, dtype=torch.long, device=image.device)
        level = torch.full((b,), self.level_idx, dtype=torch.long, device=image.device)
        return self.model(image, cond, level)["logits"]


def find_last_conv(module: torch.nn.Module) -> torch.nn.Module:
    convs = [m for m in module.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise ValueError("No Conv2d layer found for CAM target layer.")
    return convs[-1]


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


def integrated_gradients(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    steps: int = 32,
) -> np.ndarray:
    from captum.attr import IntegratedGradients

    model.eval()
    wrapped = FixedMetaModel(model, condition_idx, level_idx)

    def forward_fn(x):
        return wrapped(x)[:, target]

    ig = IntegratedGradients(forward_fn)
    baseline = torch.zeros_like(image)
    attr = ig.attribute(image, baselines=baseline, n_steps=steps)
    return _to_numpy_map(attr, image.shape[-2:])


def lrp_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
) -> np.ndarray:
    from captum.attr import LRP

    wrapped = FixedMetaModel(model, condition_idx, level_idx).eval()
    attr = LRP(wrapped).attribute(image, target=target)
    return _to_numpy_map(attr, image.shape[-2:])


def _grid_feature_mask(image: torch.Tensor, grid_size: int = 14) -> torch.Tensor:
    _, _, h, w = image.shape
    yy = torch.arange(h, device=image.device).view(h, 1)
    xx = torch.arange(w, device=image.device).view(1, w)
    gy = torch.clamp((yy * grid_size) // h, max=grid_size - 1)
    gx = torch.clamp((xx * grid_size) // w, max=grid_size - 1)
    mask = gy * grid_size + gx
    return mask.view(1, 1, h, w).long()


def perturbation_attribution(
    method: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    n_samples: int = 128,
    grid_size: int = 14,
) -> np.ndarray:
    if method == "lime":
        from captum.attr import Lime

        attr_cls = Lime
    elif method in {"shap", "kernelshap", "kernel_shap"}:
        from captum.attr import KernelShap

        attr_cls = KernelShap
    else:
        raise ValueError(f"Unsupported perturbation attribution method: {method}")

    wrapped = FixedMetaModel(model, condition_idx, level_idx).eval()
    feature_mask = _grid_feature_mask(image, grid_size=grid_size)
    attr = attr_cls(wrapped).attribute(
        image,
        target=target,
        feature_mask=feature_mask,
        n_samples=n_samples,
        perturbations_per_eval=8,
    )
    return _to_numpy_map(attr, image.shape[-2:])


def grad_cam_variant(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    variant: str = "gradcam",
    target_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam_cls = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
    }[variant.lower()]
    wrapped = FixedMetaModel(model, condition_idx, level_idx).eval()
    layer = target_layer or find_last_conv(model)
    with cam_cls(model=wrapped, target_layers=[layer]) as cam:
        grayscale_cam = cam(input_tensor=image, targets=[ClassifierOutputTarget(target)])
    return normalize_map(grayscale_cam[0])


def builtin_prototype_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int | None = None,
) -> np.ndarray:
    model.eval()
    cond = torch.tensor([condition_idx], dtype=torch.long, device=image.device)
    level = torch.tensor([level_idx], dtype=torch.long, device=image.device)
    with torch.no_grad():
        outputs = model(image, cond, level)
        maps = outputs["prototype_maps"]
        if maps.shape[1] == 0:
            return np.zeros(image.shape[-2:], dtype=np.float32)
        activations = outputs["prototype_activations"].unsqueeze(-1).unsqueeze(-1)
        weighted = (maps * activations).max(dim=1, keepdim=True).values
        weighted = F.interpolate(weighted, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return normalize_map(weighted[0, 0].detach().cpu().numpy())


def run_attribution_method(
    method: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
) -> np.ndarray:
    method_key = method.lower()
    if method_key in {"gradcam", "gradcam++", "scorecam"}:
        return grad_cam_variant(model, image, condition_idx, level_idx, target, variant=method_key)
    if method_key in {"ig", "integrated_gradients"}:
        return integrated_gradients(model, image, condition_idx, level_idx, target)
    if method_key == "lrp":
        return lrp_attribution(model, image, condition_idx, level_idx, target)
    if method_key in {"lime", "shap", "kernelshap", "kernel_shap"}:
        return perturbation_attribution(method_key, model, image, condition_idx, level_idx, target)
    if method_key in {"prototype", "builtin", "spinexnet"}:
        return builtin_prototype_attribution(model, image, condition_idx, level_idx, target)
    raise ValueError(f"Unsupported attribution method: {method}")


@torch.no_grad()
def target_confidence(model: torch.nn.Module, image: torch.Tensor, condition_idx: int, level_idx: int, target: int) -> float:
    b = image.shape[0]
    cond = torch.full((b,), int(condition_idx), dtype=torch.long, device=image.device)
    level = torch.full((b,), int(level_idx), dtype=torch.long, device=image.device)
    probs = torch.softmax(model(image, cond, level)["logits"], dim=1)
    return float(probs[:, target].mean().detach().cpu())


def deletion_insertion_auc(
    model: torch.nn.Module,
    image: torch.Tensor,
    attribution: np.ndarray,
    condition_idx: int,
    level_idx: int,
    target: int,
    mode: str = "deletion",
    steps: int = 20,
) -> float:
    """Faithfulness curve AUC.

    Deletion masks the most-attributed pixels first, so lower is better.
    Insertion reveals the most-attributed pixels first, so higher is better.
    """

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
        scores.append(target_confidence(model, perturbed, condition_idx, level_idx, target))
    return float(np.trapz(scores, fractions))


def attribution_agreement(attributions: dict[str, np.ndarray]) -> dict[str, float]:
    methods = sorted(attributions)
    out: dict[str, float] = {}
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            out[f"spearman_{left}_vs_{right}"] = spearman_corr(attributions[left], attributions[right])
            out[f"top20_iou_{left}_vs_{right}"] = topk_iou(attributions[left], attributions[right], top_fraction=0.20)
    if out:
        out["mean_spearman"] = float(np.mean([v for k, v in out.items() if k.startswith("spearman_")]))
        out["mean_top20_iou"] = float(np.mean([v for k, v in out.items() if k.startswith("top20_iou_")]))
    return out


def proxy_roi_alignment(
    attribution: np.ndarray,
    radius_fraction: float = 0.20,
    top_fraction: float = 0.20,
    condition: str | None = None,
) -> float:
    """IoU between top-attributed pixels and an anatomically-informed proxy ROI.

    Preprocessed crops are centered on RSNA label coordinates. The ROI shape
    varies by condition type to better match the expected pathology location:
      - spinal_canal_stenosis: horizontal central band (canal runs centrally)
      - *_foraminal_*: lateral ellipses on the appropriate side
      - *_subarticular_*: lateral bands shifted toward the affected side
      - default: circular ROI centred on the crop
    """

    attr = normalize_map(attribution)
    h, w = attr.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0

    if condition is not None and "spinal_canal_stenosis" in condition:
        # Horizontal band across the centre (canal is a central structure)
        band_height = h * radius_fraction
        roi = np.abs(yy - cy) <= band_height
    elif condition is not None and "foraminal" in condition:
        # Ellipse shifted left or right
        rx = w * radius_fraction * 0.8
        ry = h * radius_fraction * 1.2
        shift = w * 0.15 * (-1.0 if "left" in condition else 1.0)
        roi = ((yy - cy) / ry) ** 2 + ((xx - cx - shift) / rx) ** 2 <= 1.0
    elif condition is not None and "subarticular" in condition:
        # Lateral band shifted toward the affected side
        band_width = w * radius_fraction
        shift = w * 0.15 * (-1.0 if "left" in condition else 1.0)
        roi = np.abs(xx - cx - shift) <= band_width
    else:
        # Default circular ROI
        radius = min(h, w) * radius_fraction
        roi = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    k = max(1, int(attr.size * top_fraction))
    th = np.partition(attr.reshape(-1), -k)[-k]
    hot = attr >= th
    return float(np.logical_and(roi, hot).sum() / max(np.logical_or(roi, hot).sum(), 1))

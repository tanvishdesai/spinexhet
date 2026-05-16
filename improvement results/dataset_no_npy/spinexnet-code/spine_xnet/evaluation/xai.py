"""Attribution methods and faithfulness metrics for the XAI benchmark.

Supported method families:
  (1) CAM-based:     GradCAM, GradCAM++, ScoreCAM
  (2) Gradient-based: IntegratedGradients, GuidedBackpropagation, GradientSHAP
  (3) Perturbation:   Occlusion, LIME, KernelSHAP
  (4) Attention:       AttentionRollout (ViT-only)
  (5) Prototype:       Built-in SpineXNet prototype maps
"""

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


# ── New XAI Methods (Tier 1 Fix 2) ───────────────────────────────────────────


def gradient_shap(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    n_samples: int = 50,
    stdevs: float = 0.09,
) -> np.ndarray:
    """GradientSHAP — SHAP-motivated gradient method with random baselines."""
    from captum.attr import GradientShap

    model.eval()
    wrapped = FixedMetaModel(model, condition_idx, level_idx)
    gs = GradientShap(wrapped)
    baseline_dist = torch.randn(n_samples, *image.shape[1:], device=image.device) * 0.001
    attr = gs.attribute(
        image,
        baselines=baseline_dist,
        target=target,
        n_samples=n_samples,
        stdevs=stdevs,
    )
    return _to_numpy_map(attr, image.shape[-2:])


def occlusion_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    sliding_window_shapes: tuple[int, int, int] = (3, 15, 15),
    strides: tuple[int, int, int] = (1, 8, 8),
) -> np.ndarray:
    """Occlusion sensitivity — perturbation-based, model-agnostic."""
    from captum.attr import Occlusion

    model.eval()
    wrapped = FixedMetaModel(model, condition_idx, level_idx)
    occ = Occlusion(wrapped)
    attr = occ.attribute(
        image,
        sliding_window_shapes=sliding_window_shapes,
        strides=strides,
        target=target,
        baselines=0,
    )
    return _to_numpy_map(attr, image.shape[-2:])


def guided_backprop(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
) -> np.ndarray:
    """Guided Backpropagation — different gradient modification from IG."""
    from captum.attr import GuidedBackprop

    model.eval()
    wrapped = FixedMetaModel(model, condition_idx, level_idx)
    gbp = GuidedBackprop(wrapped)
    attr = gbp.attribute(image, target=target)
    return _to_numpy_map(attr, image.shape[-2:])


def attention_rollout(
    model: torch.nn.Module,
    image: torch.Tensor,
    condition_idx: int,
    level_idx: int,
    target: int,
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Attention Rollout (Abnar & Zuidema, 2020) for ViT architectures.

    Works with timm ViT models that expose ``model.backbone.blocks``.
    Falls back gracefully for non-ViT models.
    """
    model.eval()

    # Navigate to the ViT backbone in either BaselineClassifier or raw timm model
    backbone = getattr(model, "backbone", model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError(
            "attention_rollout requires a ViT model with `.blocks` attribute. "
            f"Got: {type(backbone).__name__}"
        )

    attention_maps: list[torch.Tensor] = []
    hooks = []

    def _hook_fn(module, inp, output):
        # timm ViT Attention returns (B, heads, seq, seq) from qkv split
        # We capture the attention weights *before* dropout
        if isinstance(output, tuple):
            attn = output[0]
        else:
            attn = output
        # Some timm versions need the softmax tensor directly;
        # the hook below captures the *output* of Attention.forward.
        # We compute attention from q @ k manually if needed.
        attention_maps.append(attn.detach())

    # timm ViTs typically have Block -> Attention -> attn_drop
    # We hook into each block's attn module
    # The hook captures the attention weight tensor from get_attn()
    _attn_weights: list[torch.Tensor] = []
    _attn_hooks: list = []

    # Use the internal _attn_map capture approach:
    # Register hooks on each Block that capture the attention matrix
    for block in blocks:
        attn_module = block.attn
        # timm stores attention as self.attn_drop(attn) where attn = q @ k.T * scale
        # We register a hook that manually captures it
        _attn_weights.clear()  # Reset for safety

    # Alternative: use timm's built-in attention capture if available
    # For robustness, we compute rollout from scratch using forward hooks
    # on the softmax inside each attention block
    import types

    original_forwards = []
    captured_attns: list[torch.Tensor] = []

    for i, block in enumerate(blocks):
        attn_mod = block.attn
        original_forwards.append(attn_mod.forward)

        def make_new_forward(original_fwd, idx):
            def new_forward(self_attn, x):
                B, N, C = x.shape
                qkv = self_attn.qkv(x).reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * self_attn.scale
                attn = attn.softmax(dim=-1)
                captured_attns.append(attn.detach())
                attn = self_attn.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self_attn.proj(x)
                x = self_attn.proj_drop(x)
                return x
            return new_forward

        attn_mod.forward = types.MethodType(make_new_forward(attn_mod.forward, i), attn_mod)

    try:
        with torch.no_grad():
            # Run full model forward (we don't need the output)
            wrapped = FixedMetaModel(model, condition_idx, level_idx)
            _ = wrapped(image)
    finally:
        # Restore original forwards
        for i, block in enumerate(blocks):
            block.attn.forward = original_forwards[i]

    if not captured_attns:
        raise RuntimeError("No attention weights captured. Check ViT architecture.")

    # Rollout computation
    device = image.device
    result = torch.eye(captured_attns[0].shape[-1], device=device)
    for attn in captured_attns:
        attn_avg = attn.mean(dim=1)  # Average over heads: (B, seq, seq)
        # Discard lowest attention values
        flat = attn_avg.view(attn_avg.shape[0], -1)
        k_keep = max(1, int(flat.shape[-1] * (1 - discard_ratio)))
        _, idx = flat.topk(k_keep, dim=-1)
        mask = torch.zeros_like(flat)
        mask.scatter_(-1, idx, 1)
        attn_avg = attn_avg * mask.view_as(attn_avg)
        # Add identity (residual connection)
        I = torch.eye(attn_avg.shape[-1], device=device)
        attn_avg = attn_avg + I
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_avg[0], result)

    # Extract CLS → patch attention, reshape to spatial grid
    seq_len = result.shape[-1] - 1  # minus CLS
    grid_size = int(seq_len ** 0.5)
    rollout_map = result[0, 1:].reshape(grid_size, grid_size)

    # Upsample to input size
    rollout_map = F.interpolate(
        rollout_map.unsqueeze(0).unsqueeze(0).float(),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    return normalize_map(rollout_map)


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
    """Dispatch to the appropriate attribution method.

    Supported methods (7+ methods across 4 families):
      CAM:          gradcam, gradcam++, scorecam
      Gradient:     ig / integrated_gradients, guided_backprop, gradient_shap
      Perturbation: occlusion, lime, shap / kernelshap
      Attention:    attention_rollout (ViT only)
      Built-in:     prototype / builtin / spinexnet
    """
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
    # ── New methods (Tier 1 Fix 2) ──
    if method_key in {"gradient_shap", "gradientshap"}:
        return gradient_shap(model, image, condition_idx, level_idx, target)
    if method_key == "occlusion":
        return occlusion_attribution(model, image, condition_idx, level_idx, target)
    if method_key in {"guided_backprop", "guidedbackprop", "gbp"}:
        return guided_backprop(model, image, condition_idx, level_idx, target)
    if method_key in {"attention_rollout", "rollout"}:
        return attention_rollout(model, image, condition_idx, level_idx, target)
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


# ── Expert ROI Alignment (Tier 1 Fix 1) ──────────────────────────────────────


def make_expert_roi_mask(
    rel_x: float,
    rel_y: float,
    radius_fraction: float = 0.15,
    H: int = 224,
    W: int = 224,
) -> np.ndarray:
    """Circular mask around an expert-annotated coordinate.

    Parameters
    ----------
    rel_x, rel_y : float in [0, 1]
        Centre of the expert annotation relative to the crop.
    radius_fraction : float
        Radius as a fraction of ``min(H, W)``.
    """
    Y, X = np.ogrid[:H, :W]
    cx, cy = rel_x * W, rel_y * H
    r_px = radius_fraction * min(H, W)
    mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= r_px ** 2
    return mask.astype(np.float32)


def expert_roi_alignment(
    attribution: np.ndarray,
    expert_mask: np.ndarray,
    threshold_pct: float = 75.0,
) -> float:
    """IoU between top-K% saliency pixels and the expert-annotated ROI.

    Parameters
    ----------
    attribution : (H, W) saliency map.
    expert_mask : (H, W) binary mask from ``make_expert_roi_mask``.
    threshold_pct : float
        Percentile threshold — top ``(100 - threshold_pct)%`` pixels are 'salient'.
    """
    smap = normalize_map(attribution)
    threshold = np.percentile(smap, threshold_pct)
    pred_mask = (smap >= threshold).astype(np.float32)
    intersection = (pred_mask * expert_mask).sum()
    union = np.clip(pred_mask + expert_mask, 0, 1).sum()
    return float(intersection / (union + 1e-8))


def coord_to_crop_roi(
    abs_x: float,
    abs_y: float,
    crop_x0: float,
    crop_y0: float,
    crop_w: float,
    crop_h: float,
) -> tuple[float, float]:
    """Convert absolute RSNA annotation coordinates to crop-relative [0,1] space."""
    rel_x = (abs_x - crop_x0) / max(crop_w, 1e-8)
    rel_y = (abs_y - crop_y0) / max(crop_h, 1e-8)
    return float(np.clip(rel_x, 0, 1)), float(np.clip(rel_y, 0, 1))

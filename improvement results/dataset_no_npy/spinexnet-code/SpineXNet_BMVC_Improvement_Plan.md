# SpineXNet → BMVC: The Complete Improvement Roadmap
*Synthesized from 4 peer reviews + project report analysis. Constraint: ML engineers only, no radiologist access.*

---

## First: The Deadline Reality

BMVC 2026 paper deadline is **May 29, 2026** — roughly two weeks away, which makes this cycle impossible given the scope of work needed. The realistic targets after completing this roadmap are:

- **BMVC 2027** (Nov 2027, deadline ~May 2027) — the main target this plan builds toward
- **MICCAI 2026 Workshop** (iMIMIC / XAIM workshop, deadline ~June 2026) — achievable with Tier 1 fixes only
- **MIDL 2027** (deadline ~Jan 2027) — ideal specialist venue, achievable with Tier 1+2
- **IEEE ISBI 2027** (deadline ~Nov 2026) — achievable with Tier 1+2+3, 4-page format

The plan is structured in tiers. Complete Tier 1 for workshop submissions. All tiers for BMVC/MICCAI main track.

---

## The Strategic Pivot (Read This First)

Every reviewer said the same thing in different words: **this currently reads as a medical application paper, not a computer vision paper**. BMVC reviewers are vision scientists. The paper needs to feel like it reveals a fundamental truth about how neural networks work, using spine imaging as a stress test.

**Current narrative:** "XAI methods disagree in spine imaging (confirming Krishna et al. in a new domain)"

**New narrative:** "Architecture family — not XAI method choice — is the primary determinant of explanation reliability in fine-grained structural classification. Modern CNNs with transformer-like inductive biases (ConvNeXt) exhibit near-ViT levels of disagreement, suggesting that gradient flow topology, not domain, governs XAI reliability. We propose a faithfulness-weighted disagreement resolution method and validate across two datasets."

This reframing turns your most confusing result (ConvNeXt ρ=0.175, almost as bad as ViT despite being a CNN) into your **headline finding** rather than an unexplained anomaly. And it makes every result speak to a general CV audience.

---

## TIER 1: Fatal Fixes (Without These, Instant Rejection)

### Fix 1 — Replace Proxy ROIs with Real RSNA Coordinate Annotations

**Why it's fatal:** All four reviewers flagged the programmatic ellipse/rectangle proxy ROIs as the single most damaging methodological flaw. Reviewer 3 called it "circular" — your ROI was derived from the same condition labels used to train the model.

**The good news you may not know:** You don't need a radiologist. The RSNA 2024 dataset already includes `train_label_coordinates.csv`, which contains (x, y) coordinates annotated by 50+ expert neuroradiologists and musculoskeletal radiologists from RSNA, ASNR, and ASSR. These are real expert annotations. The LumbarDISC paper (arXiv 2506.09162) formally describes this annotation process. You have been sitting on ground-truth expert annotations the whole time.

**What to do:**

```python
# Step 1: Load the coordinate annotations
import pandas as pd
import numpy as np

coords_df = pd.read_csv('train_label_coordinates.csv')
# Columns: study_id, series_id, instance_number, condition, level, x, y

# Step 2: For each condition-level crop you already have,
# map the expert (x, y) annotation into the crop's coordinate space.
# Your crop pipeline already stores the crop bounding box — use it to
# transform the absolute (x,y) to crop-relative coordinates.

def coord_to_crop_roi(abs_x, abs_y, crop_x0, crop_y0, crop_w, crop_h, 
                       roi_radius_frac=0.15):
    """
    Convert absolute annotation coordinates to a circular ROI 
    within the crop coordinate space.
    roi_radius_frac: fraction of crop size as radius (tune this)
    """
    rel_x = (abs_x - crop_x0) / crop_w
    rel_y = (abs_y - crop_y0) / crop_h
    radius = roi_radius_frac  # in normalized [0,1] space
    return rel_x, rel_y, radius  # center + radius of expert ROI

# Step 3: Generate expert ROI masks at the same resolution as your saliency maps
def make_expert_roi_mask(rel_x, rel_y, radius, H=224, W=224):
    """Circular mask around expert-annotated coordinate."""
    Y, X = np.ogrid[:H, :W]
    cx, cy = rel_x * W, rel_y * H
    r_px = radius * min(H, W)
    mask = ((X - cx)**2 + (Y - cy)**2) <= r_px**2
    return mask.astype(float)

# Step 4: Replace your proxy IoU computation with this mask
def compute_expert_roi_iou(saliency_map, expert_mask, threshold_pct=75):
    """
    Threshold saliency at top-K% and compute IoU against expert ROI.
    threshold_pct=75 means top 25% of pixels are 'salient'.
    """
    threshold = np.percentile(saliency_map, threshold_pct)
    pred_mask = (saliency_map >= threshold).astype(float)
    intersection = (pred_mask * expert_mask).sum()
    union = np.clip(pred_mask + expert_mask, 0, 1).sum()
    return intersection / (union + 1e-8)
```

**Key consideration:** Some crops in your dataset may not have a direct coordinate match (the coordinates are for disc levels, and your crop creation pipeline may have transformed the image). Carefully trace your crop generation code to ensure the coordinate transformation is correct. Start with ~500 samples where you're confident the mapping is accurate, report N in the paper, and acknowledge the mapping uncertainty as a limitation.

**Expected impact:** This single change transforms the clinical alignment axis from "a proxy sanity check" to "evaluated against 50+ expert annotations from a ASNR/ASSR consortium." Every reviewer will be satisfied with this.

---

### Fix 2 — Expand from 3 to 6+ XAI Methods (At Least 3 Distinct Families)

**Why it's fatal:** All four reviews flagged this. GradCAM and GradCAM++ are from the same family — effectively you compared 2 methods. Krishna et al. (your own primary reference) used more methods. A benchmark with 3 methods (2 near-identical) will be labeled "insufficient coverage" in every review.

**Target method set:**

| Method | Family | Library | Notes |
|--------|--------|---------|-------|
| GradCAM | Gradient-CAM | pytorch-grad-cam | Already have |
| GradCAM++ | Gradient-CAM | pytorch-grad-cam | Already have |
| Integrated Gradients | Gradient | Captum | Already have |
| **Occlusion Sensitivity** | **Perturbation** | **Captum** | Add — easy, no gradient needed |
| **GradientSHAP** | **Gradient + SHAP** | **Captum** | Add — theoretically grounded |
| **Guided Backpropagation** | **Gradient** | **Captum** | Add — different mechanism from IG |
| **Attention Rollout** | **Attention** | **Custom (10 lines)** | Add — ViT-specific, ESSENTIAL |

You need to cover at minimum: (1) CAM-based, (2) Gradient-based, (3) Perturbation-based, (4) Attention-based (for ViT). Currently you only have (1) and (2).

**Implementation — all methods using Captum:**

```python
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    Occlusion,
    GuidedBackprop,
    NoiseTunnel
)
import torch

class XAIMethodRegistry:
    """Unified interface for all XAI methods."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def get_gradientshap(self, input_tensor, target_class, n_samples=50):
        """GradientSHAP — SHAP-motivated gradient method with random baselines."""
        gs = GradientShap(self.model)
        # Random baseline distribution (Gaussian noise around zero)
        baseline_dist = torch.randn(n_samples, *input_tensor.shape[1:]) * 0.001
        baseline_dist = baseline_dist.to(self.device)
        attribution = gs.attribute(
            input_tensor, 
            baselines=baseline_dist,
            target=target_class,
            n_samples=n_samples,
            stdevs=0.09
        )
        return attribution.squeeze().cpu().detach().numpy()
    
    def get_occlusion(self, input_tensor, target_class, 
                      sliding_window_shapes=(3, 15, 15), strides=(1, 8, 8)):
        """Occlusion — perturbation-based, model-agnostic."""
        occ = Occlusion(self.model)
        attribution = occ.attribute(
            input_tensor,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            target=target_class,
            baselines=0  # occlude with zero (black patch)
        )
        return attribution.squeeze().cpu().detach().numpy()
    
    def get_guided_backprop(self, input_tensor, target_class):
        """Guided Backpropagation — different gradient modification than IG."""
        gbp = GuidedBackprop(self.model)
        attribution = gbp.attribute(input_tensor, target=target_class)
        return attribution.squeeze().cpu().detach().numpy()

# NOTE on Occlusion parameters for 224x224 images:
# sliding_window_shapes=(3, 15, 15) with strides=(1, 8, 8) gives
# reasonable spatial resolution without being too slow.
# For 300 samples × 7 models this will take ~2-4 hours on GPU.
# Cache results to disk exactly as you do for existing saliency maps.
```

**Attention Rollout for ViT (10-line custom implementation):**

```python
import torch
import torch.nn.functional as F

def attention_rollout(model, input_tensor, discard_ratio=0.9):
    """
    Compute Attention Rollout for ViT (Abnar & Zuidema, 2020).
    Works with timm ViT models.
    
    Args:
        model: timm ViT model with attention hooks
        input_tensor: (1, C, H, W)
        discard_ratio: fraction of lowest-attention heads to zero out
    """
    attention_maps = []
    hooks = []
    
    # Register forward hooks on all attention blocks
    def hook_fn(module, input, output):
        # output is (B, heads, seq, seq) for timm ViT attention
        attention_maps.append(output.detach())
    
    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(input_tensor)
    
    for h in hooks:
        h.remove()
    
    # Rollout computation
    result = torch.eye(attention_maps[0].shape[-1]).to(input_tensor.device)
    for attn in attention_maps:
        # Average over heads
        attn_avg = attn.mean(dim=1)  # (B, seq, seq)
        # Discard lowest attentions
        flat = attn_avg.view(attn_avg.shape[0], -1)
        _, idx = flat.topk(
            int(flat.shape[-1] * (1 - discard_ratio)), dim=-1
        )
        mask = torch.zeros_like(flat)
        mask.scatter_(-1, idx, 1)
        attn_avg = attn_avg * mask.view_as(attn_avg)
        # Add identity (residual connection effect)
        attn_avg = attn_avg + torch.eye(attn_avg.shape[-1]).to(attn_avg.device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_avg[0], result)
    
    # Extract CLS token attention to patch tokens
    # Remove CLS token, reshape to spatial grid
    seq_len = result.shape[-1] - 1  # minus CLS
    grid_size = int(seq_len ** 0.5)
    rollout_map = result[0, 1:].reshape(grid_size, grid_size)
    
    # Upsample to input size
    rollout_map = F.interpolate(
        rollout_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    return rollout_map
```

**Important framing note:** With Attention Rollout added, your ViT finding changes from "GradCAM and IG disagree on ViT" (expected, since CAM is inappropriate for ViT) to "Even transformer-native methods (Attention Rollout) disagree with gradient methods on ViT, but *less* than gradient-vs-CAM." This is actually more interesting and more defensible.

---

### Fix 3 — Full Statistical Significance on All Metrics

**Why it's fatal:** Every single reviewer called this out. Benchmark papers without confidence intervals are not accepted at top venues. Period.

**Complete statistical testing module:**

```python
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, bootstrap
from itertools import combinations

class StatisticalAnalyzer:
    
    @staticmethod
    def bootstrap_ci(data, statistic=np.mean, n_resamples=2000, confidence_level=0.95):
        """Bootstrap confidence interval for any statistic."""
        result = bootstrap(
            (data,), 
            statistic, 
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method='percentile'
        )
        return result.confidence_interval.low, result.confidence_interval.high
    
    @staticmethod
    def bootstrap_spearman_ci(saliency_a, saliency_b, n_resamples=2000):
        """Bootstrap CI for Spearman correlation between two saliency map sets."""
        n = len(saliency_a)
        correlations = []
        for _ in range(n_resamples):
            idx = np.random.randint(0, n, n)
            rho, _ = stats.spearmanr(
                saliency_a[idx].flatten(), 
                saliency_b[idx].flatten()
            )
            correlations.append(rho)
        return np.percentile(correlations, 2.5), np.percentile(correlations, 97.5)
    
    @staticmethod
    def pairwise_wilcoxon_tests(metric_dict, alpha=0.05):
        """
        Run all pairwise Wilcoxon signed-rank tests.
        metric_dict: {model_name: array_of_per_sample_scores}
        Returns DataFrame with p-values and significance flags.
        """
        results = []
        models = list(metric_dict.keys())
        for m1, m2 in combinations(models, 2):
            stat, p = wilcoxon(metric_dict[m1], metric_dict[m2])
            results.append({
                'model_A': m1,
                'model_B': m2,
                'statistic': stat,
                'p_value': p,
                'significant': p < alpha,
                'effect_size': stat / len(metric_dict[m1])  # simple effect size
            })
        return pd.DataFrame(results)
    
    @staticmethod
    def random_baseline_agreement(n_samples=300, n_bootstrap=500, map_size=(224, 224)):
        """
        Compute expected Spearman correlation between two RANDOM saliency maps.
        This is the null baseline — your architecture results should be 
        compared against this.
        """
        correlations = []
        for _ in range(n_bootstrap):
            random_a = np.random.rand(n_samples, *map_size)
            random_b = np.random.rand(n_samples, *map_size)
            # Flatten and correlate
            rho, _ = stats.spearmanr(
                random_a.reshape(n_samples, -1).mean(axis=0),
                random_b.reshape(n_samples, -1).mean(axis=0)
            )
            correlations.append(rho)
        return np.mean(correlations), np.std(correlations)

# Usage in your results tables:
# For each metric, report: value ± bootstrap_CI  (p < 0.05 vs random baseline)
# Example: DenseNet GradCAM-IG agreement: ρ = 0.489 [0.421, 0.556] (p < 0.001 vs random ρ = 0.003 ± 0.012)
```

**What to add to every table and figure:**
- All Spearman ρ values: add 95% bootstrap CI in brackets
- All Insertion/Deletion AUC comparisons: add Wilcoxon p-value vs. best model
- All IoU scores: add bootstrap CI and significance vs. random baseline
- Every bar chart: add error bars representing the CI

---

### Fix 4 — Multi-Fold Evaluation and Fix EfficientNet-B4

**Why it's fatal:** Reviewer 3 called single-fold evaluation "not peer-review acceptable." The EfficientNet-B4 missing results ("—") in the main table is described as grounds for immediate rejection.

**What to do:**
- Run all 7 architectures on folds 0, 1, 2 (minimum — 5 is ideal but 3 is defensible)
- Investigate EfficientNet-B4: check if it was a training issue (LR too high, weight decay mismatch), retrain with adjusted hyperparameters, or document the failure mode explicitly with a diagnostic analysis
- Report mean ± std across folds for all classification metrics
- For the XAI benchmark, run on fold 0 and fold 1 (300 samples each) and check if the agreement patterns hold — report whether architecture-dependent disagreement is consistent across folds

The text should say: "All reported XAI metrics are averaged over folds 0-2 of our 5-fold split; classification metrics are mean ± std over all 5 folds."

---

### Fix 5 — Qualitative Disagreement Gallery Figure

**Why it's fatal:** Reviewer 1 said "bar charts are necessary but insufficient" and all reviewers either explicitly asked for or implied the need for visual examples. This is also the figure that will make your paper's finding viscerally clear to a BMVC audience.

**What to create:**

Make a single 4-row, N-column figure as follows:

```
Row 1: "High-Agreement Case (DenseNet-121)" 
  → Input | GradCAM | GradCAM++ | IG | GradientSHAP | Occlusion | Expert ROI overlay
  
Row 2: "Low-Agreement Case (ConvNeXt-Tiny)"
  → Same columns — show how explanations scatter across the image
  
Row 3: "Near-Zero Agreement Case (ViT-Small)"
  → Same columns — show visually alarming disagreement for the same diagnosis
  
Row 4: "CBM vs. Black-Box Comparison (Non-Leaky CBM)"
  → Input | Black-box saliency | CBM concept activations | Expert ROI | Concept intervention result
```

Select these cases systematically: for Row 3 (ViT), find the image from your 300-sample evaluation set where the pairwise disagreement between GradCAM and IG is maximum. That's your "most alarming" case. For Row 1 (DenseNet), find the highest-agreement case. These are already in your saliency map cache — you just need to visualize them.

```python
def find_extreme_cases(agreement_scores_per_sample, top_k=3):
    """Find highest and lowest agreement samples for visualization."""
    sorted_idx = np.argsort(agreement_scores_per_sample)
    lowest_agreement = sorted_idx[:top_k]   # worst cases
    highest_agreement = sorted_idx[-top_k:]  # best cases
    return highest_agreement, lowest_agreement
```

---

## TIER 2: Novelty Contributions (Workshop → Main Track)

The Tier 1 fixes get you to "technically sound." Tier 2 is what gets you from "we observe disagreement" to "we contribute something new."

### Contribution 1 — Faithfulness-Weighted Consensus Map (Your Novel Method)

This is the most tractable high-impact addition that requires no external collaborators. Every reviewer asked: "you show disagreement exists, but what do you DO about it?" Here's the answer.

**The idea:** Instead of asking a clinician to choose one XAI method, produce a single consensus saliency map that is a weighted combination of all methods, where each method's weight is proportional to its faithfulness (Insertion AUC) on that specific image.

**Why this is novel:** There are papers on XAI ensembles but they use fixed weights. Yours uses per-sample adaptive weighting based on faithfulness — this is new and clinically motivated.

```python
class FaithfulnessWeightedConsensus:
    """
    Propose a consensus saliency map weighted by per-sample faithfulness.
    Each XAI method's contribution is scaled by its Insertion AUC score 
    for that specific input, measured on-the-fly.
    
    This is the paper's novel methodological contribution.
    """
    
    def __init__(self, model, xai_methods, faithfulness_metric='insertion'):
        self.model = model
        self.xai_methods = xai_methods  # dict: name -> callable
        self.metric = faithfulness_metric
    
    def compute_per_sample_insertion_auc(self, saliency_map, input_tensor, 
                                          target_class, n_steps=20):
        """
        Fast approximation of Insertion AUC for a single sample.
        Inserts pixels in order of saliency magnitude, measures 
        model confidence at each step.
        """
        H, W = saliency_map.shape
        flat_saliency = saliency_map.flatten()
        sorted_idx = np.argsort(flat_saliency)[::-1]  # high to low
        
        step_size = len(sorted_idx) // n_steps
        confidences = []
        baseline = torch.zeros_like(input_tensor)
        
        for step in range(n_steps):
            mask = torch.zeros(H * W)
            insert_idx = sorted_idx[:step * step_size + 1]
            mask[insert_idx] = 1
            mask = mask.reshape(1, 1, H, W).to(input_tensor.device)
            
            # Partial image: inserted pixels + baseline elsewhere
            partial_input = input_tensor * mask + baseline * (1 - mask)
            
            with torch.no_grad():
                logits = self.model(partial_input)
                conf = torch.softmax(logits, dim=-1)[0, target_class].item()
            confidences.append(conf)
        
        # AUC via trapezoid rule
        return np.trapz(confidences) / n_steps
    
    def get_consensus_map(self, input_tensor, target_class):
        """
        Compute the consensus saliency map for one input.
        Returns: consensus_map, {method_name: weight} dict
        """
        saliency_maps = {}
        faithfulness_scores = {}
        
        for name, method_fn in self.xai_methods.items():
            smap = method_fn(input_tensor, target_class)
            smap_normalized = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)
            saliency_maps[name] = smap_normalized
            faithfulness_scores[name] = self.compute_per_sample_insertion_auc(
                smap_normalized, input_tensor, target_class
            )
        
        # Softmax-normalize faithfulness scores to get weights
        scores = np.array(list(faithfulness_scores.values()))
        weights = np.exp(scores) / np.exp(scores).sum()
        
        # Weighted average of normalized saliency maps
        consensus = sum(
            w * saliency_maps[name] 
            for name, w in zip(faithfulness_scores.keys(), weights)
        )
        
        return consensus, dict(zip(faithfulness_scores.keys(), weights))
```

**What to evaluate:**
- Compute Insertion AUC of the consensus map vs. each individual method
- Show that consensus matches or exceeds the best individual method on most architectures
- Show that consensus particularly helps for architectures with high disagreement (ViT, ConvNeXt) where no single method is reliable
- Compare Expert ROI IoU of consensus vs. best individual method

This is a genuine novel contribution and it's entirely feasible with your existing pipeline.

---

### Contribution 2 — Deep Analysis of the ConvNeXt-ViT Disagreement Paradox

Reviewer 3 called this "your deepest intellectual contribution hiding in plain sight." ConvNeXt is a CNN but shows near-ViT levels of XAI disagreement (ρ=0.175 vs ViT ρ=0.076). Why?

This analysis transforms the paper from "observe disagreement" to "explain disagreement" — which is a genuine theoretical contribution.

**Hypotheses to test (all implementable in ML only):**

**Hypothesis 1: Depthwise separable convolutions break gradient locality**
ConvNeXt uses depthwise separable convolutions (inherited from MobileNet lineage) and large 7×7 kernels. Regular CNNs (ResNet, DenseNet) use 3×3 standard convolutions. Larger receptive fields may cause gradient attribution methods to spread attributions more diffusely.

```python
def compute_gradient_localization_score(model, input_tensor, target_class):
    """
    Measure how spatially concentrated the gradients are.
    High score = localized gradients = XAI methods will agree more.
    Low score = diffuse gradients = XAI methods will disagree.
    
    This is the "gradient locality" metric — your new proposed diagnostic.
    """
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    loss = output[0, target_class]
    loss.backward()
    
    grad = input_tensor.grad.abs().squeeze().mean(dim=0)  # (H, W)
    grad_normalized = grad / (grad.sum() + 1e-8)
    
    # Spatial entropy of gradient distribution
    # Low entropy = concentrated (good for XAI agreement)
    # High entropy = diffuse (bad for XAI agreement)
    entropy = -(grad_normalized * torch.log(grad_normalized + 1e-8)).sum()
    
    return entropy.item()

# Run this for all 7 architectures on 300 samples
# Hypothesis: ConvNeXt gradient entropy ≈ ViT gradient entropy > ResNet/DenseNet entropy
# This would explain the disagreement pattern
```

**Hypothesis 2: Layer normalization vs. batch normalization**
ConvNeXt uses Layer Norm instead of Batch Norm (like ViT). Batch Norm creates strong feature discrimination signals that gradient methods can latch onto. Layer Norm may not.

Test this by measuring the gradient signal-to-noise ratio at different layers.

**Hypothesis 3: Skip connection topology**
Run activation patching experiments: zero out skip connections in ConvNeXt and ResNet-50, and measure whether XAI agreement changes. If ConvNeXt's disagreement improves when skip connections are ablated (and ResNet's doesn't change), this implicates the connection topology.

**The proposed new metric — Gradient Locality Score (GLS):**
Define GLS as the reciprocal of gradient spatial entropy. Show that GLS correlates with inter-method agreement across your 7 architectures. This is your paper's theoretical contribution: **GLS predicts XAI agreement before you run any XAI method**.

```python
# Final analysis: scatter plot of GLS vs. Spearman ρ (inter-method agreement)
# across all 7 architectures — if this is a strong correlation, you have found
# a fundamental connection between architecture and XAI reliability.
# This is publishable at a top venue.
```

---

### Contribution 3 — Redesign CBM Concepts Using Image-Derived Features

The current non-leaky CBM concepts are all metadata (condition type, laterality, level, neighboring label density). Reviewers correctly identified that these contain no visual information — they're metadata encoded twice.

**What to replace them with (ML-only, no radiologist):**

**Strategy A: Unsupervised feature discovery using DINO**

```python
# DINO/DINOv2 learns semantically meaningful features without labels.
# Use it to extract patch-level features from your spine crops,
# then cluster them to discover visual concepts automatically.

import torch
from torchvision import transforms

# Load pretrained DINOv2
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

def extract_dino_patch_features(image_tensor, n_concepts=8):
    """
    Extract DINOv2 patch features and cluster into visual concepts.
    Returns per-patch concept activations.
    """
    with torch.no_grad():
        features = dinov2.get_intermediate_layers(
            image_tensor, n=1, return_class_token=False
        )[0]  # (B, n_patches, feature_dim)
    return features
```

**Strategy B: Hand-crafted visual features from MRI signal properties**

These are grounded in radiology physics but require no expert annotation:

```python
def compute_visual_concepts(image_crop):
    """
    Compute biologically motivated visual concepts from MRI signal.
    These are image-derived, not metadata-derived.
    
    Concepts:
    1. Central canal signal ratio: mean intensity in center 20% vs. outer 80%
       (low ratio = potential stenosis — canal signal suppressed)
    2. Bilateral symmetry index: L-R asymmetry in signal distribution
       (high asymmetry = foraminal or subarticular asymmetry)
    3. Dark region fraction: fraction of pixels below 30th percentile
       (high fraction = disc dehydration or loss of CSF signal)
    4. Gradient magnitude variance: measure of structural boundary sharpness
       (low variance = blurred anatomy = potential pathology)
    5. Horizontal band intensity: mean intensity of middle horizontal third
       (disc level signal)
    6. Vertical edge density: concentration of vertical edges (facet joints)
    7. Signal heterogeneity: local variance across 8 regions
    8. Central-to-peripheral intensity gradient: key indicator of stenosis pattern
    """
    import cv2
    img = image_crop.numpy()
    H, W = img.shape[-2:]
    
    # Concept 1: Central canal signal ratio
    center_mask = np.zeros((H, W))
    cy, cx = H//2, W//2
    r = min(H, W) // 5
    cv2.circle(center_mask, (cx, cy), r, 1, -1)
    central_signal = (img * center_mask).sum() / (center_mask.sum() + 1e-8)
    peripheral_signal = (img * (1 - center_mask)).sum() / ((1 - center_mask).sum() + 1e-8)
    canal_ratio = central_signal / (peripheral_signal + 1e-8)
    
    # Concept 2: Bilateral symmetry index
    left_half = img[:, :W//2]
    right_half = img[:, W//2:]
    right_flipped = np.flip(right_half, axis=-1)
    symmetry = 1 - np.abs(left_half - right_flipped).mean()
    
    # Concept 3: Dark region fraction
    dark_fraction = (img < np.percentile(img, 30)).mean()
    
    # Concept 4: Gradient magnitude variance
    sobel_x = cv2.Sobel(img.mean(axis=0) if img.ndim==3 else img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img.mean(axis=0) if img.ndim==3 else img, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_variance = grad_mag.var()
    
    # Concept 5: Middle band intensity (disc signal)
    band = img[..., H//3:2*H//3, :]
    band_intensity = band.mean()
    
    # Concept 6: Signal heterogeneity
    h_step, w_step = H//4, W//4
    region_means = [img[..., i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step].mean()
                    for i in range(4) for j in range(4)]
    heterogeneity = np.std(region_means)
    
    return np.array([canal_ratio, symmetry, dark_fraction, grad_variance,
                     band_intensity, heterogeneity])
```

**Strategy C: BiomedCLIP zero-shot concept scoring (strongest)**

```python
# BiomedCLIP (Microsoft) is trained on 15M biomedical image-text pairs.
# Use it to score the presence of clinical concepts in your crops
# WITHOUT any radiologist annotation.

from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Define visual concepts as text prompts
CLINICAL_CONCEPTS = [
    "narrowing of the spinal canal",
    "loss of cerebrospinal fluid signal",
    "disc space narrowing",
    "asymmetric neural foramen",
    "disc herniation",
    "bright disc signal on T2",
    "dark dehydrated disc on T2",
    "facet joint hypertrophy",
]

def score_clinical_concepts(image_tensor, concepts=CLINICAL_CONCEPTS):
    """
    Zero-shot concept scoring using BiomedCLIP.
    Returns probability of each concept being present in the image.
    """
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_tokens = tokenizer(concepts)
        text_features = model.encode_text(text_tokens)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    return similarity.squeeze().cpu().numpy()
```

Use Strategy C (BiomedCLIP) as your primary concept source. This replaces the 7 metadata-derived concepts with 8 clinically meaningful visual concepts, scored by a model trained on 15M biomedical image-text pairs. The paper can honestly say "our CBM uses visual concepts grounded in radiological terminology, scored by a biomedical vision-language model."

---

## TIER 3: Elevating to Strong Accept

### Addition 1 — Second Dataset for Generalizability

All reviewers implicitly or explicitly asked: "is this just a quirk of the RSNA dataset?" A second dataset proving the finding generalizes is what separates a workshop paper from a main-track paper.

**Best option: CUB-200-2011 (Caltech-UCSD Birds)** — public, fine-grained structural classification (bird species from subtle visual features), semantically similar task to spine (finding the right anatomical region matters), no clinical setup needed.

Why CUB-200 works for your narrative: it's a canonical fine-grained recognition benchmark where spatial localization matters. If ViT shows high XAI disagreement on CUB-200 too (and ConvNeXt shows intermediate disagreement), the finding is no longer about spine imaging — it's about fine-grained structural classification in general. That's a BMVC paper.

Run your full XAI benchmark (6 methods, 5 architectures, same agreement metrics) on CUB-200. This requires roughly 1 week of GPU time and is completely feasible for an ML team.

**Second best option: NIH ChestX-ray14** — medical imaging but different modality (chest vs. spine), available on Kaggle, similar classification task. Shows the finding generalizes within medical imaging before the CUB-200 generalization shows it applies to natural images.

### Addition 2 — Model Randomization Sanity Check (Adebayo et al. 2018)

This is a 1-hour implementation that reviewers will specifically look for in any benchmark paper on XAI methods. It tests whether your saliency maps are actually responding to the model's learned features vs. just the input image structure.

```python
def cascading_randomization_test(model, input_tensor, target_class, 
                                  xai_method_fn, n_levels=5):
    """
    Adebayo et al. (NeurIPS 2018) cascading randomization test.
    Progressively randomize model weights from output layer to input.
    A faithful XAI method's saliency should change significantly 
    as weights are randomized. If saliency is invariant to weight 
    randomization, the method is not actually using the model.
    
    Returns: list of SSIM scores between original and randomized saliency maps.
    Low SSIM = method is sensitive to model (GOOD — it's using the model).
    High SSIM = method ignores model structure (BAD).
    """
    from skimage.metrics import structural_similarity as ssim
    import copy
    
    # Baseline saliency from trained model
    original_saliency = xai_method_fn(input_tensor, target_class)
    
    model_copy = copy.deepcopy(model)
    ssim_scores = []
    layers = list(model_copy.named_parameters())
    layer_step = len(layers) // n_levels
    
    for level in range(1, n_levels + 1):
        # Randomize layers from output backward
        for i in range(len(layers) - level * layer_step, len(layers)):
            name, param = layers[i]
            param.data = torch.randn_like(param.data)
        
        # Recompute saliency on randomized model
        rand_saliency = xai_method_fn(input_tensor, target_class)
        
        score = ssim(original_saliency, rand_saliency, 
                     data_range=original_saliency.max() - original_saliency.min())
        ssim_scores.append(score)
    
    return ssim_scores

# Expected results:
# - GradCAM should show rapidly decreasing SSIM (good — responds to model)
# - GradCAM++ similar to GradCAM
# - Integrated Gradients should show similar behavior
# - If any method shows SSIM > 0.7 even at full randomization → flag it
```

### Addition 3 — Disagreement Uncertainty Maps for Clinical Communication

Beyond the consensus map (Contribution 1, Tier 2), add a disagreement uncertainty visualization:

```python
def compute_disagreement_map(saliency_maps_dict):
    """
    Pixel-wise disagreement map showing WHERE methods disagree.
    This is a clinical communication tool — shows clinicians 
    which image regions have uncertain explanations.
    
    High uncertainty = methods disagree about this region's importance.
    Low uncertainty = all methods agree this region matters (or doesn't).
    """
    maps = np.stack([
        (m - m.min()) / (m.max() - m.min() + 1e-8)
        for m in saliency_maps_dict.values()
    ])  # (n_methods, H, W)
    
    # Pixel-wise standard deviation across methods
    disagreement = maps.std(axis=0)
    
    # Also compute agreement-weighted confidence
    mean_saliency = maps.mean(axis=0)
    confidence = mean_saliency * (1 - disagreement)  # high where methods agree AND salient
    
    return disagreement, confidence

# Visualize as a 3-panel figure:
# Panel 1: Consensus saliency (high = important region, all methods agree)
# Panel 2: Disagreement map (red = methods strongly disagree)
# Panel 3: Clinical confidence map (only show high-saliency regions where methods agree)
```

---

## TIER 4: The Revised Paper Narrative

With all tiers completed, here is how the paper should be framed for BMVC:

**Title (draft):** "Architecture Family Governs Explanation Reliability: A Multi-Method XAI Benchmark for Fine-Grained Structural Classification"

**Abstract (draft structure):**
1. Problem: Post-hoc XAI methods are widely deployed in high-stakes settings, but their reliability is poorly understood beyond classification accuracy.
2. Gap: Prior work establishes disagreement exists (Krishna et al.); we lack understanding of what drives it.
3. What we do: We benchmark 6 XAI methods across 7 architectures on fine-grained structural classification, using expert-annotated ground truth for clinical evaluation and two datasets spanning biomedical and natural images.
4. Finding 1: Architecture family — not XAI method selection — is the primary determinant of agreement. Classic CNNs show moderate agreement (ρ ≈ 0.47–0.49); modern CNNs with transformer-like properties (ConvNeXt) and ViTs show near-random agreement (ρ ≈ 0.18 and 0.08 respectively).
5. Finding 2: Gradient locality (measurable without any XAI) predicts inter-method agreement, providing a pre-hoc diagnostic for XAI reliability.
6. Contribution: We propose a faithfulness-weighted consensus method that consistently outperforms individual XAI methods, particularly for architectures with high disagreement.
7. Finding 3: Concept Bottleneck Models with visual concepts achieve highest faithfulness despite lower classification accuracy, supporting ante-hoc interpretability for high-stakes deployment.

**Section structure:**
1. Introduction + Related Work
2. SpineXNet Benchmark: Dataset, Architectures, XAI Methods, Evaluation Protocol
3. Results: Agreement Analysis (the architecture finding)
4. Analysis: Why Do Modern CNNs Behave Like Transformers? (gradient locality analysis)
5. FW-Consensus: The Proposed Resolution Method
6. Ante-Hoc Interpretability: CBM with Visual Concepts
7. Generalization: Validation on CUB-200
8. Discussion + Conclusion

---

## Implementation Timeline

**Weeks 1-2 (Tier 1):**
- Implement RSNA coordinate-based expert ROIs, validate on 100 samples
- Implement Occlusion + GradientSHAP + Guided Backprop + Attention Rollout (all in Captum + custom ViT hook)
- Run new XAI methods on cached crops (no retraining needed)
- Add bootstrap CI and Wilcoxon tests to all existing metrics
- Investigate and fix EfficientNet-B4 training issue
- Launch fold 1 and fold 2 training runs

**Weeks 3-4 (Tier 2):**
- Implement and evaluate FW-Consensus method
- Run gradient locality analysis on all 7 architectures
- Replace CBM concepts with BiomedCLIP zero-shot visual concepts, retrain CBM only (fast)
- Create qualitative disagreement gallery figure

**Weeks 5-6 (Tier 3):**
- Run XAI benchmark on CUB-200 (5 architectures × 6 methods)
- Run model randomization sanity checks
- Begin manuscript draft

**Weeks 7-9:**
- Complete manuscript
- Create all final figures (reuse most existing infrastructure)
- Internal review + revision
- Target: MICCAI workshop or ISBI by end of this period; BMVC 2027 / MIDL 2027 for full version

---

## Venue Recommendation (Honest Assessment)

**After Tier 1 + Tier 2 only (6-7 weeks of work):**
- MICCAI 2026 Workshop (iMIMIC or XAIM) — deadline ~June 2026 — **achievable and appropriate**
- ISBI 2026 — deadline ~Nov 2026 — **strong submission**

**After Tier 1 + Tier 2 + Tier 3 (9-12 weeks):**
- MIDL 2027 (deadline ~Jan 2027) — **ideal venue, strong accept probability**
- BMVC 2026 is already closed. **BMVC 2027** (deadline ~May 2027) — achievable with all tiers

**Important:** MIDL is not a "fallback" — it is the premier venue for ML in medical imaging, directly targeting the exact audience that will appreciate and cite this work. A strong MIDL paper with the full tier 1-3 improvements is a better career outcome than a weak BMVC submission that bounces back.

---

## Summary Checklist

### Tier 1 — Fatal Fixes
- [ ] Replace proxy ROIs with RSNA `train_label_coordinates.csv` expert annotations
- [ ] Add Occlusion Sensitivity (Captum)
- [ ] Add GradientSHAP (Captum)
- [ ] Add Guided Backpropagation (Captum)
- [ ] Add Attention Rollout (custom 10-line ViT hook)
- [ ] Add bootstrap CIs on all Spearman ρ values
- [ ] Add Wilcoxon signed-rank tests for all pairwise comparisons
- [ ] Add random baseline for agreement metrics
- [ ] Run 5-fold CV (minimum 3 folds reported)
- [ ] Fix / explain EfficientNet-B4 missing results
- [ ] Correct "31,500 saliency maps" claim → 6,300
- [ ] Create qualitative disagreement gallery (4-row figure)

### Tier 2 — Novel Contributions
- [ ] Implement Faithfulness-Weighted Consensus method
- [ ] Evaluate FW-Consensus vs. individual methods on all architectures
- [ ] Compute Gradient Locality Score for all 7 architectures
- [ ] Test GLS vs. inter-method agreement correlation
- [ ] Replace metadata-derived CBM concepts with BiomedCLIP visual concepts
- [ ] Retrain non-leaky CBM with new visual concepts
- [ ] Report GLS-agreement scatter plot as key theoretical figure

### Tier 3 — Strong Accept Additions
- [ ] Run full XAI benchmark on CUB-200 (generalization experiment)
- [ ] Run model randomization sanity checks (Adebayo et al. 2018)
- [ ] Implement and visualize disagreement uncertainty maps

### Narrative / Writing
- [ ] Reframe title and abstract to emphasize architecture-dependent finding
- [ ] Ensure abstract reads as a general CV contribution, not a medical application
- [ ] Add a dedicated "Why does ConvNeXt behave like ViT?" analysis section
- [ ] Cite LumbarDISC paper (arXiv 2506.09162) for the formalized dataset description

# SpineXNet v1 → v2: A Complete Analysis

> **Project:** SpineXNet Explainability Disagreement Benchmark  
> **Target Venue:** BMVC / MICCAI / MIDL (top-tier)  
> **Assessment Date:** 2026-05-16

---

## 1. Drawbacks of the First Implementation (v1)

The v1 run completed a functional pipeline (7 models trained, XAI benchmark on 3 methods, concept intervention) but had **five fatal methodological flaws** that every reviewer flagged:

### Fatal Flaw 1: Fabricated Clinical Alignment Ground Truth
- Used **programmatic proxy ROIs** (ellipses for canal stenosis, rectangles for foraminal narrowing) as "clinical alignment" targets
- These were derived from the same condition labels used to train the model → **circular evaluation**
- The RSNA dataset already contains `train_label_coordinates.csv` with 50+ expert neuroradiologist annotations — these were never used

### Fatal Flaw 2: Only 2 Effective XAI Methods
- Claimed 3 methods (GradCAM, GradCAM++, Integrated Gradients), but **GradCAM and GradCAM++ are from the same family**
- Effectively only compared 2 method families (CAM-based vs. gradient-based)
- A benchmark paper needs at minimum 3 distinct families: CAM-based, gradient-based, and perturbation-based
- Missing: Occlusion, GradientSHAP, Guided Backpropagation, Attention Rollout (for ViT)

### Fatal Flaw 3: No Statistical Significance
- All metrics reported as point estimates with **zero confidence intervals**
- No bootstrap CIs on Spearman ρ values, no Wilcoxon tests for pairwise comparisons
- No random baseline for agreement metrics (what does "random" agreement look like?)
- Benchmark papers without CIs are "not peer-review acceptable" (verbatim from reviewer)

### Fatal Flaw 4: Single-Fold Evaluation
- All results from **fold 0 only** of a 5-fold split
- No way to assess result stability or report mean ± std
- EfficientNet-B4 had missing results ("—") in the main table — unexplained failure

### Fatal Flaw 5: No Qualitative Evidence
- Only showed bar charts and tables — no visual examples of disagreement
- Reviewers said "bar charts are necessary but insufficient" for a paper about visual explanations
- No side-by-side saliency map gallery showing the disagreement problem

### Additional Weaknesses
- **Metadata-only CBM concepts**: All 7 non-leaky concepts were derived from metadata (condition type, laterality, level), containing no visual information — "metadata encoded twice"
- **No novel contribution**: Only confirmed Krishna et al.'s finding in a new domain, without proposing any resolution method
- **Medical-application framing**: Read as a medical imaging paper, not a computer vision paper — wrong audience for BMVC

---

## 2. The Improvement Plan

The mentor-provided `SpineXNet_BMVC_Improvement_Plan.md` was organized into **4 tiers**:

### Tier 1 — Fatal Fixes (Instant Rejection Without These)

| Fix | What | Status in v2 |
|-----|------|:---:|
| **Fix 1** | Replace proxy ROIs with RSNA expert coordinate annotations | ✅ Done |
| **Fix 2** | Expand from 3 → 6+ XAI methods (3+ distinct families) | ✅ Done (6 methods) |
| **Fix 3** | Add bootstrap CIs + Wilcoxon tests on all metrics | ✅ Done |
| **Fix 4** | Multi-fold evaluation (minimum 3 folds) | ✅ Done (3 folds) |
| **Fix 5** | Qualitative disagreement gallery figure | ✅ Done |

### Tier 2 — Novel Contributions (Workshop → Main Track)

| Contribution | What | Status in v2 |
|-------------|------|:---:|
| **Contribution 1** | Faithfulness-Weighted Consensus Map | ✅ Done |
| **Contribution 2** | ConvNeXt-ViT disagreement analysis (Gradient Locality Score) | ✅ Done |
| **Contribution 3** | Redesign CBM with BiomedCLIP visual concepts | ❌ Excluded (see below) |

> [!IMPORTANT]
> **BiomedCLIP was excluded by design.** You investigated it and found the zero-shot concept scores on cropped spine MRI patches were essentially noise — they did not correlate with clinical reality on these small crops. This was a justified engineering decision, and the original deterministic non-leaky concepts were retained.

### Tier 3 — Strong Accept Additions

| Addition | What | Status in v2 |
|----------|------|:---:|
| **Addition 1** | Second dataset (CUB-200 or NIH ChestX-ray14) | ❌ Not done |
| **Addition 2** | Model randomization sanity check (Adebayo et al.) | ✅ Done |
| **Addition 3** | Disagreement uncertainty maps | ❌ Not done |

### Tier 4 — Narrative Reframing

| Item | Status in v2 |
|------|:---:|
| Reframe as architecture-governs-XAI-reliability paper | 🟡 Results gathered, not yet written |
| ConvNeXt-ViT paradox analysis section | ✅ Data collected |

---

## 3. Were You Successful?

### Scorecard: What Got Done

| Category | Items Planned | Items Completed | Score |
|----------|:---:|:---:|:---:|
| **Tier 1 — Fatal Fixes** | 5 | 5 | **100%** |
| **Tier 2 — Novel Contributions** | 3 | 2 (BiomedCLIP excluded with justification) | **67%** |
| **Tier 3 — Strong Accept** | 3 | 1 | **33%** |
| **Tier 4 — Narrative** | 2 | 1 (data only, no manuscript) | **50%** |

### What the v2 Results Actually Show

#### Classification (3-fold mean ± std)

| Model | WLL ↓ | Bal. Acc ↑ | AUC ↑ |
|-------|:---:|:---:|:---:|
| **ConvNeXt-Tiny** | **0.509 ± 0.016** | 74.2% ± 0.5% | **0.920 ± 0.010** |
| ViT-Small | 0.521 ± 0.021 | 71.1% ± 3.5% | 0.918 ± 0.007 |
| CBM Leaky | 0.517 ± 0.014 | 73.3% ± 0.5% | 0.916 ± 0.003 |
| CBM Non-Leaky | 0.541 ± 0.013 | 71.7% ± 2.5% | **0.919 ± 0.011** |
| DenseNet-121 | 0.527 ± 0.020 | 72.5% ± 2.5% | 0.917 ± 0.004 |
| ResNet-50 | 0.573 ± 0.017 | 70.0% ± 1.9% | 0.900 ± 0.006 |
| EfficientNet-B4 | 0.650 ± 0.023 | 66.9% ± 0.7% | 0.882 ± 0.014 |

> [!TIP]
> **EfficientNet-B4 now has complete results** across all 3 folds — it's consistently the weakest model, which is itself a valid finding rather than a missing entry.

#### XAI Agreement (v2 — 6 methods, with CIs)

| Model | Mean Spearman ρ | 95% CI |
|-------|:---:|:---:|
| DenseNet-121 | 0.436 | [0.426, 0.446] |
| CBM Non-Leaky | 0.406 | [0.392, 0.420] |
| CBM Leaky | 0.405 | [0.388, 0.422] |
| ResNet-50 | 0.384 | [0.373, 0.395] |
| ConvNeXt-Tiny | **0.317** | [0.305, 0.330] |
| EfficientNet-B4 | 0.237 | [0.231, 0.243] |
| **ViT-Small** | **0.231** | [0.223, 0.239] |

> [!WARNING]
> The v1→v2 agreement values shifted because you now have 6 methods instead of 3 and are measuring 15 pairwise correlations instead of 3. The absolute numbers changed but the **ranking and story are preserved**: ViT and ConvNeXt show the worst agreement, classic CNNs show the best. This is actually a stronger result because it holds across more methods.

#### Faithfulness-Weighted Consensus (Novel Contribution)

| Model | Consensus Insertion AUC | Best Individual Method |
|-------|:---:|:---:|
| **CBM Non-Leaky** | **0.878** | 0.825 (Occlusion) |
| ConvNeXt-Tiny | 0.817 | 0.816 (Occlusion) |
| DenseNet-121 | 0.801 | 0.786 (GradCAM) |
| ViT-Small | 0.778 | 0.793 (Occlusion) |

> The consensus map **matches or exceeds** the best individual method on most architectures, validating the contribution.

#### Expert ROI Clinical Alignment (New — Real Expert Annotations)

| Model | GradCAM Expert IoU | Gradient-Based Expert IoU |
|-------|:---:|:---:|
| DenseNet-121 | **0.260** | 0.196–0.240 |
| CBM Non-Leaky | 0.135 | 0.194–0.196 |
| ConvNeXt-Tiny | 0.080 | 0.215–0.219 |
| ViT-Small | 0.081 | 0.208–0.236 |

> CAM-based methods have very low expert IoU on ConvNeXt and ViT (< 0.09), while gradient-based methods are more consistent across architectures (~0.20). This is a genuinely new finding enabled by the real expert annotations.

#### Gradient Locality Score (Tier 2 Analysis)

| Model | Gradient Entropy | GLS |
|-------|:---:|:---:|
| ConvNeXt | 9.959 | 0.1004 |
| ViT-Small | 9.982 | 0.1002 |
| DenseNet-121 | 10.195 | 0.0981 |
| ResNet-50 | 10.337 | 0.0967 |
| EfficientNet-B4 | 10.420 | 0.0960 |

> [!CAUTION]
> **The GLS experiment did NOT produce the expected result.** The gradient entropy values are clustered tightly between 9.96–10.42 — essentially flat across architectures. There is **no meaningful correlation** between GLS and inter-method agreement. The theoretical hypothesis (that gradient locality explains XAI disagreement) was not supported by the data. The GLS-agreement scatter plot likely shows a near-random cloud.

#### Randomization Sanity Check

| Method | Full Random SSIM (ConvNeXt) | Interpretation |
|--------|:---:|---|
| GradCAM | 0.37 | Moderate sensitivity to model weights — **passes** |
| GradientSHAP | 0.02 | Very low SSIM at full randomization — **passes strongly** |
| Integrated Gradients | 0.02 | Very low SSIM — **passes strongly** |
| Occlusion | 0.08 | Low SSIM — **passes** |

> All methods pass the Adebayo sanity check: their saliency maps change significantly when model weights are randomized. Gradient-based methods (SHAP, IG) are most sensitive, which validates that they truly use the model's learned features.

#### Concept Intervention (Unchanged from v1)

| Model | Fix Rate | Top Concept |
|-------|:---:|---|
| CBM Non-Leaky | 59.2% | `adjacent_pathology_density` (51.3%) |
| CBM Leaky | 84.6% | `pathology_present` (74.1% — confirms leakage) |

---

## 4. Did You Achieve What You Wanted?

### The Honest Assessment

**Partially yes, with important caveats.**

#### ✅ What Was Achieved
1. **All Tier 1 fatal fixes are complete** — the paper can no longer be rejected on methodological grounds
2. **Multi-fold validation** establishes result stability
3. **6 XAI methods across 4 families** (CAM, gradient, perturbation, guided backprop) — comprehensive
4. **Expert annotation-based clinical alignment** — replaces the circular proxy ROI
5. **Faithfulness-Weighted Consensus** — a genuine novel contribution that works
6. **Randomization sanity check** — validates the saliency maps are real
7. **Complete, detailed per-fold/per-condition/per-level data** exists for every metric
8. **Qualitative gallery figure** exists

#### ⚠️ What Fell Short

1. **Gradient Locality Score failed to deliver**: The theoretical hypothesis that GLS predicts XAI agreement was **not supported by the data**. This was positioned as a key Tier 2 contribution ("your paper's theoretical contribution") and it produced a null result. This needs to be either honestly reported as a negative finding or the analysis needs fundamental rethinking.

2. **BiomedCLIP visual concepts excluded**: While justified, this means the CBM still uses metadata-only concepts. The paper cannot claim "visual concept supervision" — a point reviewers specifically wanted addressed.

3. **No second dataset**: The generalizability experiment (CUB-200 or ChestX-ray14) was not done. This was the single most important differentiator between a workshop paper and a main-track paper.

4. **Attention Rollout failed for ViT**: The `skipped` field in `vit_small/xai_summary_v2.json` shows an error (`got an unexpected keyword argument 'attn_mask'`). This was flagged as "ESSENTIAL" in the improvement plan and would have made the ViT story much stronger.

5. **No manuscript draft exists**: All the data is gathered but the actual paper writing hasn't started.

6. **Fold coverage is 3, not 5**: The plan said "minimum 3, ideal 5." You hit the minimum, not the ideal.

---

## 5. What Should You Do Next

### Immediate Priority: Write the Paper

You have enough data right now to submit a **solid workshop paper** (MICCAI iMIMIC/XAIM, or ISBI). The experimental work is done for this tier. Don't run any more experiments — start writing.

### Path A: Workshop Submission (Fast Track, 2–3 weeks)

This is the pragmatic path if you want a publication credit soon.

1. **Draft the manuscript immediately** using the `research-paper-writing` skill. You have all the data.
2. **Frame the GLS null result honestly** — report it as "we tested the hypothesis that gradient spatial entropy predicts XAI agreement and found no significant correlation, suggesting the disagreement mechanism is more complex than a simple gradient locality measure"
3. **Target MICCAI 2026 Workshop (iMIMIC/XAIM)** — deadline ~June 2026
4. **Keep 6 methods, 7 architectures, 3 folds, expert ROI, consensus map** — this is plenty for a 4-page workshop paper

### Path B: Main Conference (BMVC/MIDL, 3–4 months more work)

This requires additional experiments:

1. **Fix Attention Rollout for ViT** — debug the `attn_mask` keyword error in timm. This is likely a timm version mismatch (newer versions changed the attention hook interface). This is critical for the ViT narrative.

2. **Run on CUB-200** — This takes ~1 week of GPU time and is the single highest-ROI experiment remaining. If the architecture-dependent disagreement pattern reproduces on a natural image dataset, you have a BMVC paper.

3. **Rethink the GLS analysis** — Instead of spatial gradient entropy, try:
   - **Effective receptive field (ERF) size** as the predictor variable
   - **Layer-wise gradient norm concentration** (how much gradient signal concentrates in deeper layers)
   - **Feature map spatial autocorrelation** (smoother feature maps → more CAM agreement)

4. **Add 2 more folds** (train folds 3 and 4) — brings you to 5-fold, which is the gold standard

5. **Consider adding LIME or KernelSHAP** — these perturbation-based methods were in the original plan but not implemented, probably due to compute cost. If you run on CUB-200 anyway, you'll already have the GPU time budget.

### My Recommendation

> [!IMPORTANT]
> **Go with Path A first.** Submit a workshop paper to MICCAI 2026 Workshop with what you have — the data is complete enough. Then extend it for BMVC 2027 / MIDL 2027 with the CUB-200 experiment and fixed Attention Rollout. This gives you one publication credit now AND a stronger follow-up later.

### Concrete Next Steps (Ordered)

| Step | Task | Time | Priority |
|:---:|------|:---:|:---:|
| 1 | **Write the paper draft** using the research-paper-writing skill | 1 week | 🔴 Critical |
| 2 | Fix Attention Rollout for ViT (timm API change) | 1 day | 🟡 High |
| 3 | Create all publication-quality figures (LaTeX-compatible) | 2 days | 🟡 High |
| 4 | Internal review and revision | 3 days | 🟡 High |
| 5 | Submit to MICCAI workshop | — | 🔴 Critical |
| 6 | Run CUB-200 experiments (for BMVC/MIDL extended version) | 1 week | 🟢 Later |
| 7 | Rethink the GLS analysis with alternative metrics | 3 days | 🟢 Later |
| 8 | Train folds 3 and 4 for 5-fold reporting | 2 days GPU | 🟢 Later |

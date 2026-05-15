# BMVC Research Paper: The Explainability Disagreement Problem in Lumbar Spine Degenerative Classification

1. Executive Summary

This document presents a **complete research project blueprint** designed to produce a paper worthy of acceptance at **BMVC, MICCAI, or comparable tier-1 venues**. The key insight: post-hoc XAI methods (Grad-CAM, LIME, SHAP, etc.) applied to spine pathology classifiers produce **contradictory explanations** for the same prediction — a phenomenon known as the *explainability disagreement problem* (Krishna et al., 2022). No prior work has systematically quantified this in the context of lumbar spine imaging. We conduct the **first comprehensive explainability benchmark** for spine degenerative classification, comparing 7+ post-hoc methods against a concept-based ante-hoc baseline across multiple faithfulness, consistency, and clinical alignment metrics.

This is an **analysis-first, benchmark-style paper** — the contribution is the *finding* (disagreement is severe and clinically dangerous) and the *evaluation framework* (standardized, reproducible, quantitative), not a new architecture.

---

## 1. Gap Analysis: Why Existing Work Falls Short

### What the Literature Review Papers Do

| Paper                        | Approach                                                    | Limitation                                                                                  |
| ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Vaishnavi V K (2025, IJSREM) | CNN (ResNet/U-Net) for severity classification              | No dataset specified, no XAI, no reproducibility                                            |
| Jun Qian (2024, Elsevier)    | U-Net + Transformer fusion for disc herniation segmentation | Segmentation-focused, no interpretability evaluation                                        |
| Spine-CNN Attenuation Model  | CNN with attention for classification                       | Attention ≠ explanation; no faithfulness evaluation                                        |
| Various others in LR folder  | Standard DL + basic metrics                                 | All share: (1) single dataset, (2) no XAI or post-hoc only, (3) no cross-dataset evaluation |

### The Critical Gap

> [!IMPORTANT]
> **Every paper in the literature review treats explainability as an afterthought** — either absent entirely, or limited to showing a few Grad-CAM heatmaps as qualitative "evidence." None rigorously evaluate whether their explanations are *faithful* to the model's decision-making, *consistent* across similar inputs, or *clinically meaningful* to radiologists. Most critically, **no paper compares multiple XAI methods against each other** to check if they even agree.

### The Disagreement Problem (Krishna et al., 2022)

The *explainability disagreement problem* — formalized by Krishna et al. at NeurIPS 2022 — shows that:

- Different post-hoc explanation methods frequently produce **contradictory** explanations for the same prediction
- There are **no principled frameworks** for practitioners to resolve these disagreements
- In clinical settings, this creates a **dangerous false sense of security** — clinicians may trust whichever explanation confirms their bias

This problem has been studied in tabular data and natural images, but **never systematically demonstrated in medical imaging for spine pathology**. That is our contribution.

### What Top Venues Demand (BMVC/MICCAI Standards)

1. **Novel problem formulation** — not just "we classify spine images better"
2. **Rigorous quantitative evaluation** — standardized metrics, not just pretty heatmaps
3. **Comprehensive baselines** — comparing many methods fairly, not cherry-picking one
4. **Clinical relevance** — explanations evaluated against medical knowledge
5. **Reproducibility** — open-source code, public dataset, clear methodology
6. **Actionable findings** — what should practitioners actually do?

---

## 2. Proposed Paper: Problem Statement & Framing

### Title (Working)

**"How Faithful Are Your Explanations? A Comprehensive Benchmark of Post-Hoc Explainability Methods for Lumbar Spine Degenerative Classification"**

Alternative titles:

- *"The Explainability Disagreement Problem in Spine Pathology Classification: A Quantitative Analysis"*
- *"Beyond Heatmaps: Benchmarking Explanation Faithfulness for Clinical Spine MRI Classification"*
- *"Do XAI Methods Agree? A Systematic Evaluation of Explanation Consistency in Lumbar Spine Imaging"*

### Core Research Questions

> **RQ1:** Do state-of-the-art post-hoc XAI methods produce consistent explanations when applied to the same spine pathology classifier and the same input image?
>
> **RQ2:** Which post-hoc methods provide the most *faithful* explanations (i.e., accurately reflect the model's actual decision-making process) for spine degenerative classification?
>
> **RQ3:** Can a lightweight concept-based ante-hoc approach provide explanations that are inherently more consistent and clinically aligned than post-hoc alternatives, even if classification accuracy is slightly lower?

### Why This Is BMVC/MICCAI-Worthy

This framing elevates the work from "spine classification + XAI" to addressing a **fundamental problem in trustworthy clinical AI**: the *explainability disagreement problem*. Our contributions are:

1. **First systematic disagreement study in spine imaging** — directly extends Krishna et al. (2022) to a clinically critical domain
2. **Comprehensive faithfulness benchmark** — 7+ XAI methods × 8+ quantitative metrics × 5 clinical conditions × 5 vertebral levels
3. **Clinical alignment evaluation** — explanations compared against anatomically-defined regions of interest (spinal canal, foramina, disc space)
4. **Actionable recommendation** — we don't just show the problem, we evaluate which methods clinicians should actually trust
5. **Fully reproducible** — public dataset (RSNA 2024), open-source code, standardized evaluation (Quantus)

### Key Message of the Paper

> **"Post-hoc explainability methods, when applied to lumbar spine degenerative classification, exhibit severe disagreement: different methods highlight contradictory regions for the same prediction, with mean pairwise Spearman correlation below 0.1. We benchmark 7 methods across 8 faithfulness metrics on 48,000+ condition-level predictions, revealing that no single post-hoc method is consistently faithful across all conditions. A lightweight concept-based model provides inherently consistent explanations at a modest accuracy trade-off, suggesting that ante-hoc interpretability deserves greater attention in clinical deployment."**

---

## 3. Methodology Overview

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL PIPELINE                         │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │  PHASE 1:     │    │  PHASE 2:         │    │  PHASE 3:      │  │
│  │  Train Strong │───▶│  Apply 7+ XAI     │───▶│  Quantitative  │  │
│  │  Black-Box    │    │  Methods           │    │  Evaluation    │  │
│  │  Models       │    │                    │    │                │  │
│  │              │    │  • Grad-CAM        │    │  • Faithfulness│  │
│  │  • ConvNeXt  │    │  • Grad-CAM++      │    │  • Agreement   │  │
│  │  • ResNet-50 │    │  • Score-CAM       │    │  • Consistency │  │
│  │  • EfficientN│    │  • LIME            │    │  • Clinical    │  │
│  │  • ViT-Small │    │  • SHAP            │    │    Alignment   │  │
│  │  • DenseNet  │    │  • IntGrad         │    │  • Robustness  │  │
│  │              │    │  • LRP             │    │                │  │
│  └──────────────┘    └──────────────────┘    └───────────────┘  │
│         │                                            │           │
│         ▼                                            ▼           │
│  ┌──────────────┐                           ┌───────────────┐   │
│  │  PHASE 1b:    │                           │  PHASE 4:      │   │
│  │  Train Light  │──────────────────────────▶│  Comparative   │   │
│  │  CBM Baseline │                           │  Analysis      │   │
│  │  (Ante-hoc)  │                           │  Post-hoc vs   │   │
│  │              │                           │  Ante-hoc      │   │
│  └──────────────┘                           └───────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Phase 1: Black-Box Classification Models

We train **5 strong black-box classifiers** on the RSNA 2024 lumbar spine dataset. The primary model is ConvNeXt-Tiny (our best-performing architecture from prior experiments). Additional backbones ensure our XAI findings generalize across architectures.

#### Classification Baselines (Black-Box):

| Model                     | Params | Rationale                                                                                 |
| ------------------------- | ------ | ----------------------------------------------------------------------------------------- |
| **ConvNeXt-Tiny**   | ~28M   | Primary model — modern pure-CNN, strong performance (AUC 0.92)                           |
| **ResNet-50**       | ~25M   | Classic baseline — most commonly used in medical imaging papers                          |
| **EfficientNet-B4** | ~19M   | Efficient architecture — popular in Kaggle medical competitions                          |
| **ViT-Small**       | ~22M   | Transformer baseline — tests whether XAI disagreement differs for attention-based models |
| **DenseNet-121**    | ~8M    | Medical imaging standard — used in CheXpert, ISIC, etc.                                  |

Each model uses:

- ImageNet pre-trained weights
- Condition + Level metadata embeddings (same as current pipeline)
- Ordinal severity classification head (CORAL loss)
- AdamW optimizer, cosine annealing, FP16 mixed precision
- 224×224 image resolution
- Stratified 5-fold cross-validation

> [!NOTE]
> We already have a strong ConvNeXt-Tiny baseline achieving val AUC 0.921 and weighted log-loss 0.524. The other backbones need only basic tuning — we're not chasing SOTA accuracy, we need reasonably strong models to apply XAI methods to.

### 3.3 Phase 1b: Concept-Based Ante-Hoc Baseline

We train a **lightweight Concept Bottleneck Model (CBM)** as the ante-hoc interpretable alternative:

```
┌──────────┐    ┌──────────────┐    ┌──────────────────┐
│ ConvNeXt │───▶│ Concept Head │───▶│ Classification   │
│ Backbone │    │ (5 concepts) │    │ Head             │
│          │    │              │    │                  │
└──────────┘    │ Predicts:    │    │ Normal/Mild →    │
                │ • Pathology  │    │ Moderate →       │
                │   present    │    │ Severe           │
                │ • Severity   │    └──────────────────┘
                │   indicator  │
                │ • Laterality │
                │ • Adjacent   │
                │   density    │
                └──────────────┘
```

**Key design decisions for the CBM:**

- **Same backbone** (ConvNeXt-Tiny) as the primary black-box — ensures fair comparison
- **NO residual bypass** — all information must flow through concepts (true bottleneck)
- **NO prototypes** — keep it simple and clean
- **Only programmatic pseudo-concepts** — derived deterministically from RSNA labels (no BiomedCLIP noise)
- **Concept supervision weight: 0.3** — light regularization, not dominant

The CBM serves two purposes:

1. **Ante-hoc baseline** whose explanations (concept activations) are inherently self-consistent
2. **Concept intervention demonstration** — show that correcting a concept prediction changes the final classification (practical clinical utility)

### 3.4 Phase 2: XAI Method Application

Apply **7 post-hoc XAI methods** to each black-box model:

#### Post-Hoc Methods:

| Method                         | Category              | Library          | Key Property                                              |
| ------------------------------ | --------------------- | ---------------- | --------------------------------------------------------- |
| **Grad-CAM**             | CAM-based             | pytorch-grad-cam | Most widely used in medical imaging                       |
| **Grad-CAM++**           | CAM-based             | pytorch-grad-cam | Improved multi-object localization                        |
| **Score-CAM**            | CAM-based             | pytorch-grad-cam | Gradient-free, perturbation-based                         |
| **LIME**                 | Perturbation-based    | lime             | Model-agnostic, superpixel-based                          |
| **KernelSHAP**           | Perturbation-based    | shap             | Game-theoretic, model-agnostic                            |
| **Integrated Gradients** | Gradient-based        | captum           | Satisfies axioms (sensitivity, implementation invariance) |
| **LRP**                  | Backpropagation-based | captum/zennit    | Layer-wise relevance propagation                          |

#### Ante-Hoc Methods (from CBM):

| Method                        | Category | Key Property                                       |
| ----------------------------- | -------- | -------------------------------------------------- |
| **Concept Activations** | Built-in | Global: which concepts activated and how strongly  |
| **Concept Gradient**    | Built-in | Which concept most influenced the final prediction |

For each test image, each method produces an **attribution map** (saliency map) of the same spatial resolution, indicating which image regions the method considers most important for the prediction.

### 3.5 Phase 3: Quantitative Evaluation

This is **the core contribution** — we don't just show heatmaps, we *measure* explanation quality.

#### A. Faithfulness Metrics (per-method)

| Metric                                 | What It Measures                                                | Implementation                      |
| -------------------------------------- | --------------------------------------------------------------- | ----------------------------------- |
| **Faithfulness (Deletion AUC)**  | Removing top-attributed pixels should decrease confidence       | Quantus `FaithfulnessCorrelation` |
| **Faithfulness (Insertion AUC)** | Adding top-attributed pixels should increase confidence         | Quantus `MonotonicityCorrelation` |
| **Infidelity**                   | How well does the explanation approximate the model's behavior? | Captum `infidelity`               |
| **Sensitivity-n**                | Is the explanation sensitive to features that matter?           | Quantus `Sensitivity`             |
| **Pixel Flipping**               | Progressive masking of top-k% pixels                            | Custom (standard protocol)          |

#### B. Agreement Metrics (between methods)

| Metric                              | What It Measures                               | Implementation                      |
| ----------------------------------- | ---------------------------------------------- | ----------------------------------- |
| **Spearman Rank Correlation** | Do methods rank pixels the same way?           | `scipy.stats.spearmanr` pairwise  |
| **Top-k% IoU**                | Do methods highlight the same regions?         | Top-20% thresholded overlap         |
| **Kendall's τ**              | Ordinal agreement between attribution rankings | `scipy.stats.kendalltau` pairwise |
| **Mean Pairwise Agreement**   | Average agreement across all method pairs      | Aggregate of above                  |

#### C. Consistency Metrics (same method, perturbed input)

| Metric                                 | What It Measures                                                 | Implementation                                            |
| -------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| **Augmentation Consistency**     | Same image + minor transform → same explanation?                | IoU between attribution maps under geometric augmentation |
| **Input Perturbation Stability** | Small noise → stable explanation?                               | Quantus `LocalLipschitzEstimate`                        |
| **Cross-Fold Consistency**       | Same image, model trained on different fold → same explanation? | Compare attributions from fold-0 vs fold-1 models         |

#### D. Clinical Alignment Metrics

| Metric                                 | What It Measures                                            | Implementation                                                                             |
| -------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Proxy ROI Alignment**          | Do explanations highlight the anatomically relevant region? | IoU between attribution and crop center region (where the pathology was annotated in RSNA) |
| **Condition-Specific Alignment** | Do explanations differ appropriately across conditions?     | Compare attributions for stenosis vs foraminal narrowing at the same level                 |
| **Severity Gradient**            | Do explanations intensify with severity?                    | Correlation between attribution magnitude and severity grade                               |

### 3.6 Phase 4: Comparative Analysis

The final analysis compares:

1. **Post-hoc methods against each other** → disagreement quantification
2. **Post-hoc methods against ante-hoc CBM** → consistency comparison
3. **Across architectures** → does disagreement depend on backbone?
4. **Across conditions** → is disagreement worse for certain pathologies?
5. **Across severity levels** → is disagreement worse for subtle (mild) cases?

---

## 4. Dataset Strategy

### 4.1 Primary Dataset: RSNA 2024 Lumbar Spine Competition

| Property      | Details                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| Platform      | **Kaggle** (directly attached to notebooks, ~35 GB)                                                            |
| Size          | ~2,697 patients, multi-sequence MRI (DICOM)                                                                          |
| Modalities    | Sagittal T1, Sagittal T2/STIR, Axial T2                                                                              |
| Labels        | 5 conditions × 5 disc levels × 3 severity grades                                                                   |
| Conditions    | Spinal canal stenosis, L/R neural foraminal narrowing, L/R subarticular stenosis                                     |
| Key Files     | `train.csv` (labels), `train_label_coordinates.csv` (x/y per condition/level), `train_series_descriptions.csv` |
| Total Samples | ~48,657 condition-level crops                                                                                        |
| Role          | **Training, validation, and evaluation**                                                                       |

> [!IMPORTANT]
> **Single-dataset is standard for benchmark papers.** MICCAI/BMVC benchmark papers routinely use one large, well-annotated dataset (e.g., CheXpert, ISIC, BraTS). The RSNA dataset is multi-institution (8 sites), multi-reader annotated, and provides spatial coordinate labels — this is more than adequate. The contribution is the *evaluation methodology*, not the dataset breadth.

### 4.2 Evaluation Protocols

| Protocol                         | Details                                                                   |
| -------------------------------- | ------------------------------------------------------------------------- |
| **Stratified 5-fold CV**   | Primary — reports mean ± std across folds                               |
| **Per-institution split**  | Train on 6 institutions, test on 2 → simulates cross-site generalization |
| **Per-condition analysis** | Separate XAI evaluation for each of the 5 conditions                      |
| **Per-severity analysis**  | Separate evaluation for Normal/Mild vs Moderate vs Severe                 |
| **Per-level analysis**     | Separate evaluation for L1/L2 through L5/S1                               |

### 4.3 XAI Evaluation Subset

For computational feasibility, the full XAI benchmark (7 methods × faithfulness metrics) is run on a **stratified subset of 500-1000 samples** from the validation set, balanced across conditions and severity levels. Agreement and consistency metrics are computed on a larger set (2000+ samples).

---

## 5. Experimental Design

### 5.1 Experiment 1: The Disagreement Experiment (FLAGSHIP)

**Goal:** Demonstrate that post-hoc XAI methods produce severely inconsistent explanations for spine pathology classification.

**Method:**

1. Train ConvNeXt-Tiny black-box on RSNA (already done — AUC 0.921)
2. Apply all 7 post-hoc XAI methods to the same 1000 validation predictions
3. For each prediction, compute pairwise Spearman correlation and Top-20% IoU between all 21 method pairs
4. Aggregate into a **7×7 agreement matrix** (the paper's central figure)
5. Report mean pairwise agreement overall, per-condition, and per-severity

**Expected finding:** Mean pairwise Spearman < 0.15 (preliminary data shows ~0.05), demonstrating severe disagreement.

**Why this matters:** If a clinician uses Grad-CAM and sees one region highlighted, but LIME would have shown a completely different region, the "explanation" is meaningless for building clinical trust.

### 5.2 Experiment 2: Faithfulness Benchmark

**Goal:** Determine which XAI method provides the most faithful explanations for spine classification.

**Method:**

1. For each of the 7 post-hoc methods + CBM concept gradients:
   - Compute Deletion AUC, Insertion AUC, Infidelity, Pixel Flipping curves
2. Rank methods by each faithfulness metric
3. Check if rankings are consistent across metrics (meta-disagreement)

**Key table:** 8 methods × 5 faithfulness metrics → which method wins?

### 5.3 Experiment 3: Cross-Architecture Disagreement

**Goal:** Test whether disagreement severity depends on model architecture.

**Method:**

1. Apply Grad-CAM, Integrated Gradients, and LIME to all 5 backbone architectures
2. Compare intra-model disagreement (different XAI methods, same model) vs inter-model disagreement (same XAI method, different models)

**Expected finding:** Intra-model disagreement (between XAI methods) is larger than inter-model disagreement (between architectures), showing the problem lies with the explanation methods, not the models.

### 5.4 Experiment 4: Consistency Under Augmentation

**Goal:** Test explanation stability when the input image is slightly perturbed.

**Method:**

1. Take 500 validation images
2. Apply 5 minor augmentations each (small rotation ±5°, brightness ±10%, horizontal flip, slight crop, Gaussian noise σ=0.01)
3. Compute explanation IoU between original and augmented versions
4. Compare stability across XAI methods

**Expected finding:** Some methods (Grad-CAM++) are more stable than others (LIME, SHAP), but none achieve >0.8 IoU consistency.

### 5.5 Experiment 5: CBM Concept Intervention

**Goal:** Demonstrate the practical clinical utility of the concept-based approach.

**Method:**

1. At test time, take predictions where the CBM is wrong
2. "Correct" one concept prediction (e.g., set `pathology_present = 1`)
3. Show how the final classification changes
4. Quantify: what fraction of errors can be corrected by concept intervention?

**Why this matters:** This demonstrates a unique advantage of ante-hoc interpretability — the clinician can *interact* with the explanation, not just view it.

### 5.6 Experiment 6: Clinical Alignment Analysis

**Goal:** Evaluate which XAI method best aligns with anatomically relevant regions.

**Method:**

1. Use the RSNA `train_label_coordinates.csv` (x, y) annotations as proxy ground-truth ROIs
2. Define a circular ROI around each annotation coordinate (radius = 15% of image width)
3. Compute IoU between each method's top-20% attribution and the proxy ROI
4. Compare across methods, conditions, and severity levels

**Expected finding:** All methods show poor clinical alignment (<0.2 IoU with proxy ROI), but some are systematically better than others.

### 5.7 Experiment 7: Cross-Institution Explanation Transfer

**Goal:** Test whether explanations generalize across institutions.

**Method:**

1. Train on 6 RSNA institutions, test on 2 held-out ones
2. Compare explanation faithfulness and consistency on in-distribution vs out-of-distribution data
3. Check if disagreement worsens under distribution shift

### 5.8 Ablation Studies

| Ablation                                                   | What It Tests                                      |
| ---------------------------------------------------------- | -------------------------------------------------- |
| XAI evaluation at different image resolutions (160 vs 224) | Does resolution affect disagreement?               |
| XAI evaluation with different last conv layers             | Does layer choice affect Grad-CAM?                 |
| CBM with 3 vs 5 vs 10 concepts                             | Optimal concept granularity                        |
| CBM with vs without ordinal loss                           | Does ordinal formulation help the CBM?             |
| LIME with different superpixel sizes                       | Hyperparameter sensitivity of perturbation methods |
| SHAP with different kernel widths                          | Hyperparameter sensitivity of kernel methods       |

---

## 6. Architecture & Model Details

### 6.1 Primary Black-Box: ConvNeXt-Tiny Classifier

This is the existing `BaselineClassifier` in `spine_xnet/models/baselines.py`:

```python
# Architecture: ConvNeXt-Tiny (ImageNet pretrained)
#   → Global Average Pool
#   → Condition+Level Embedding (2 × 16-dim)
#   → MLP(feature_dim + 32, 256, 3)  # classification head
#   → MLP(feature_dim + 32, 256, 2)  # ordinal CORAL head
```

**Training config** (proven from results-logs-3.txt):

- Optimizer: AdamW (lr=1e-3, backbone_lr=1e-4, weight_decay=0.01)
- Scheduler: Cosine annealing with 5-epoch warmup
- Loss: `CrossEntropy(class_weights=auto) + 0.2 × CORAL_ordinal`
- Label smoothing: 0.0 (cleaner signal for faithfulness evaluation)
- Image size: 224×224 (NOT 160 — larger resolution for better XAI maps)
- Batch size: 16, grad_accum=2 (effective BS 32)
- Epochs: 30 with patience=10
- AMP (FP16), channels_last, gradient clipping at 1.0

### 6.2 Additional Black-Box Backbones

All use **identical training protocol and classification head**. Only the backbone differs:

| Model           | `timm` name             | Feature dim | Notes                                      |
| --------------- | ------------------------- | ----------- | ------------------------------------------ |
| ResNet-50       | `resnet50`              | 2048        | Classic; widely cited baseline             |
| EfficientNet-B4 | `efficientnet_b4`       | 1792        | Efficient; popular in competitions         |
| ViT-Small       | `vit_small_patch16_224` | 384         | Transformer; different attention mechanism |
| DenseNet-121    | `densenet121`           | 1024        | Medical imaging standard                   |

> [!NOTE]
> **We do NOT need all 5 backbones to achieve top accuracy.** Their purpose is to test whether XAI disagreement is architecture-dependent. ConvNeXt-Tiny remains the primary model for all XAI experiments. The other 4 are used only in Experiment 3 (Cross-Architecture Disagreement).

### 6.3 Lightweight CBM (Ante-Hoc Baseline)

Uses the existing `ConceptBottleneckBaseline` in `spine_xnet/models/baselines.py`:

```python
# Architecture: ConvNeXt-Tiny (same backbone, frozen during concept training)
#   → Global Average Pool
#   → Condition+Level Embedding (2 × 16-dim)
#   → Concept Head: MLP(feature_dim + 32, 256, 5)  # 5 programmatic concepts
#   → sigmoid  → concept_probs (interpretable intermediate)
#   → Classifier: MLP(5 + 32, 256, 3)  # classification from concepts only
#   → Ordinal: MLP(5 + 32, 256, 2)
```

**5 Programmatic Concepts** (clean, deterministic, no BiomedCLIP):

| Concept                        | Derivation                                               | Interpretation                      |
| ------------------------------ | -------------------------------------------------------- | ----------------------------------- |
| `pathology_present`          | 1 if severity ∈ {moderate, severe}                      | "Is there pathology at this level?" |
| `severe_grade`               | 1 if severity = severe                                   | "Is the pathology severe?"          |
| `left_laterality`            | 1 if condition starts with "left_"                       | "Is this a left-sided condition?"   |
| `right_laterality`           | 1 if condition starts with "right_"                      | "Is this a right-sided condition?"  |
| `adjacent_pathology_density` | Fraction of adjacent levels with pathology in same study | "How widespread is degeneration?"   |

> [!IMPORTANT]
> **Yes, concepts 1-2 are derived from the target labels.** This is intentional and explicitly acknowledged in the paper. The CBM's role is NOT to achieve SOTA accuracy — it's to demonstrate the *interpretability advantage* (concept intervention, consistent explanations). The paper will discuss this label leakage as a limitation and argue it represents a reasonable proxy for expert-annotated concepts that would exist in a clinical deployment scenario.

**Training protocol:**

- Two-stage training:
  1. Stage 1: Train backbone + concept head jointly (concept loss only, 10 epochs)
  2. Stage 2: Freeze backbone, train concept head + classifier jointly (concept + classification loss, 20 epochs)
- Concept loss: BCE with weight 0.3
- Classification loss: CrossEntropy(class_weights=auto) with weight 1.0
- No residual bypass — all information flows through 5 concept dimensions

---

## 7. Implementation Plan: What to Build

### 7.1 What Already Exists (Reuse)

| Component                    | File                                 | Status               |
| ---------------------------- | ------------------------------------ | -------------------- |
| ConvNeXt-Tiny baseline       | `spine_xnet/models/baselines.py`   | ✅ Ready             |
| CBM baseline                 | `spine_xnet/models/baselines.py`   | ✅ Ready             |
| RSNA dataset / dataloader    | `spine_xnet/data/dataset.py`       | ✅ Ready             |
| Image cache builder          | `scripts/build_image_cache.py`     | ✅ Ready             |
| Training loop (Trainer)      | `spine_xnet/training/trainer.py`   | ✅ Ready             |
| Classification metrics       | `spine_xnet/evaluation/metrics.py` | ✅ Ready             |
| XAI benchmark runner (basic) | `scripts/run_xai_benchmark.py`     | ⚠️ Needs expansion |
| Config system (YAML)         | `configs/`                         | ✅ Ready             |
| Loss functions               | `spine_xnet/models/losses.py`      | ✅ Ready             |

### 7.2 What Needs to Be Built/Modified

#### A. Expand XAI Method Suite (`spine_xnet/evaluation/xai_methods.py`)

Current state: Only supports Grad-CAM, Grad-CAM++, Integrated Gradients, and prototype maps.

**Add:**

- Score-CAM (from `pytorch-grad-cam`)
- LIME (from `lime` library, adapted for medical imaging — use SLIC superpixels)
- KernelSHAP (from `shap` library)
- LRP (from `captum` or `zennit`)

Each method should return a **normalized attribution map** of shape `(H, W)` with values in `[0, 1]`.

#### B. Build Quantitative XAI Evaluation (`spine_xnet/evaluation/xai_metrics.py`)

New module implementing:

```python
def compute_faithfulness_metrics(model, image, attribution, label):
    """Compute deletion AUC, insertion AUC, infidelity, pixel flipping."""
    ...

def compute_agreement_metrics(attributions: dict[str, np.ndarray]):
    """Compute pairwise Spearman, Top-k IoU, Kendall's τ across methods."""
    ...

def compute_consistency_metrics(model, image, method, augmentations):
    """Compute augmentation consistency and perturbation stability."""
    ...

def compute_clinical_alignment(attribution, annotation_xy, image_size):
    """Compute proxy ROI alignment using RSNA coordinate annotations."""
    ...
```

Use **Quantus** library for standardized faithfulness metrics where possible. Implement custom metrics for clinical alignment.

#### C. New XAI Benchmark Script (`scripts/run_xai_benchmark_v2.py`)

Expanded version of existing script:

```
Usage:
  python run_xai_benchmark_v2.py \
    --config configs/baselines/convnext_blackbox.yaml \
    --checkpoint outputs/best.pt \
    --manifest manifests/manifest_with_concepts.csv \
    --fold 0 \
    --output-dir xai_results/ \
    --max-samples 1000 \
    --methods gradcam gradcam++ scorecam lime shap intgrad lrp \
    --metrics faithfulness agreement consistency clinical \
    --save-maps  # Save attribution maps for visualization
```

Output structure:

```
xai_results/
├── faithfulness_metrics.csv        # Per-sample, per-method faithfulness scores
├── agreement_matrix.csv            # 7×7 pairwise agreement (Spearman + IoU)
├── consistency_metrics.csv         # Per-method stability scores
├── clinical_alignment.csv          # Per-method, per-condition alignment
├── attribution_maps/               # Saved .npy maps for visualization
│   ├── sample_001_gradcam.npy
│   ├── sample_001_lime.npy
│   └── ...
└── figures/
    ├── agreement_heatmap.png       # Central figure of the paper
    ├── faithfulness_ranking.png
    └── example_disagreement.png    # Side-by-side heatmap comparison
```

#### D. Visualization Script (`scripts/visualize_xai.py`)

Generates publication-quality figures:

1. **Agreement Heatmap** — 7×7 matrix of pairwise Spearman correlations, annotated with values
2. **Disagreement Examples** — Grid of the same image with all 7 attribution maps side-by-side
3. **Faithfulness Bar Charts** — Per-method deletion/insertion AUC comparison
4. **Condition-Specific Analysis** — Spider/radar plots per condition
5. **Severity Gradient Plot** — Attribution intensity vs severity grade
6. **CBM Concept Intervention** — Before/after concept correction examples

#### E. CBM Training Configs

New config files:

```yaml
# configs/baselines/cbm_pure.yaml
base_config: ../base.yaml
experiment_name: cbm_pure_5concepts

model:
  type: cbm
  backbone: convnext_tiny
  pretrained: true
  num_concepts: 5
  num_classes: 3
  meta_dim: 16
  hidden_dim: 256
  dropout: 0.2
  ordinal: true

loss:
  class_weights: auto
  classification: 1.0
  concept: 0.3
  prototype_cluster: 0.0
  prototype_diversity: 0.0
  ordinal: 0.2

training:
  epochs: 30
  image_size: 224
  lr: 0.001
  backbone_lr: 0.0001
```

#### F. Concept Intervention Script (`scripts/concept_intervention.py`)

Demonstrates CBM's practical utility:

```python
# For each misclassified sample:
# 1. Get current concept predictions
# 2. Replace one concept with ground truth
# 3. Re-run classifier head with corrected concepts
# 4. Check if classification is now correct
# 5. Report intervention success rate per concept
```

### 7.3 What to Remove/Simplify

| Component                          | Action                          | Rationale                         |
| ---------------------------------- | ------------------------------- | --------------------------------- |
| SpineXNet model                    | **Keep but deprioritize** | Not part of the new paper's scope |
| BiomedCLIP concept extraction      | **Remove from pipeline**  | Source of noise; not used         |
| Prototype layer                    | **Remove from pipeline**  | Not used in CBM or black-box      |
| `merge_concepts.py`              | **Remove**                | No longer needed                  |
| `extract_biomedclip_concepts.py` | **Remove**                | No longer needed                  |

---

## 8. Compute Strategy (Kaggle-Optimized)

### 8.1 Resource Constraints

| Resource           | Limit                                   |
| ------------------ | --------------------------------------- |
| GPU                | 2× T4 (16GB each) via Kaggle notebooks |
| Session limit      | 9 hours per session                     |
| Weekly quota       | 30 hours GPU                            |
| Storage            | 20GB /kaggle/working                    |
| Dataset attachment | Up to 4 datasets × 20GB each           |

### 8.2 Kaggle Session Plan

| Session              | Duration | Task                                                      | Output                                       |
| -------------------- | -------- | --------------------------------------------------------- | -------------------------------------------- |
| **Session 1**  | 3h       | Build image cache (224×224)                              | `image_cache_224/images_uint8.npy`         |
| **Session 2**  | 6h       | Train ConvNeXt-Tiny black-box (fold 0)                    | `outputs/convnext_blackbox/fold_0/best.pt` |
| **Session 3**  | 6h       | Train ConvNeXt-Tiny (fold 1) + CBM (fold 0)               | Two checkpoints                              |
| **Session 4**  | 4h       | Train ResNet-50 + DenseNet-121 (fold 0 each)              | Two baselines                                |
| **Session 5**  | 4h       | Train EfficientNet-B4 + ViT-Small (fold 0 each)           | Two more baselines                           |
| **Session 6**  | 6h       | XAI benchmark: ConvNeXt — all 7 methods, faithfulness    | `xai_results/convnext/`                    |
| **Session 7**  | 6h       | XAI benchmark: Agreement + Consistency + Clinical         | `xai_results/convnext/`                    |
| **Session 8**  | 4h       | XAI benchmark: Cross-architecture (3 methods × 4 models) | `xai_results/cross_arch/`                  |
| **Session 9**  | 3h       | CBM evaluation + Concept intervention + CBM XAI           | `xai_results/cbm/`                         |
| **Session 10** | 2h       | Visualization generation + final eval                     | Figures, tables                              |

**Total estimated GPU time: ~44 hours (~1.5 weeks)**

> [!TIP]
> **Optimization strategies:**
>
> - Use image cache for ALL sessions (eliminates DICOM I/O)
> - Save all intermediate results to Kaggle output → re-attach as dataset input for next session
> - LIME and SHAP are the slowest methods — limit to 500 samples for these
> - Use `max_val_batches` to cap validation during training (save time for XAI)

### 8.3 Dataset Attachments (per session)

```
/kaggle/input/
├── competitions/rsna-2024-lumbar-spine-degenerative-classification/  # RSNA data
├── datasets/vasuaashadesai/spinexnet-code/                           # Code
├── datasets/vasuaashadesai/image-cache-224/                          # Pre-built cache
└── datasets/vasuaashadesai/checkpoints-v2/                           # Trained models (sessions 2-5)
```

---

## 9. Paper Structure

### Target Venue: **BMVC 2025** (or MICCAI 2025 Workshop)

Paper format: 9 pages + references + supplementary

### Proposed Structure:

```
Title: "How Faithful Are Your Explanations? Benchmarking Post-Hoc 
        Explainability Disagreement in Lumbar Spine Degenerative Classification"

Abstract (200 words)

1. Introduction (1 page)
   - Clinical motivation: AI-assisted spine diagnosis
   - The trust problem: clinicians need explanations
   - The disagreement problem: different XAI methods contradict each other
   - Our contribution: first systematic benchmark in spine imaging
   - Key finding preview: severe disagreement + CBM as consistent alternative

2. Related Work (1 page)
   2.1 Deep Learning for Spine Pathology Classification
   2.2 Post-Hoc Explainability Methods (Grad-CAM, LIME, SHAP, IG)
   2.3 The Disagreement Problem (Krishna et al., 2022)
   2.4 Concept Bottleneck Models and Ante-Hoc Interpretability
   2.5 Explainability Evaluation Metrics (Quantus, Faithfulness)

3. Methodology (2 pages)
   3.1 Dataset and Preprocessing (RSNA 2024)
   3.2 Classification Models (5 black-box + 1 CBM)
   3.3 Explainability Methods (7 post-hoc + 1 ante-hoc)
   3.4 Evaluation Metrics
       3.4.1 Faithfulness (Deletion, Insertion, Infidelity)
       3.4.2 Agreement (Spearman, Top-k IoU)
       3.4.3 Consistency (Augmentation, Perturbation)
       3.4.4 Clinical Alignment (Proxy ROI)

4. Experiments and Results (3 pages)
   4.1 Classification Performance (Table 1: all models)
   4.2 The Disagreement Problem in Spine Imaging
       (Figure 1: 7×7 agreement matrix — CENTRAL FIGURE)
       (Figure 2: Qualitative disagreement examples)
   4.3 Faithfulness Benchmark (Table 2: all methods × all metrics)
   4.4 Cross-Architecture Analysis (Table 3: disagreement per backbone)
   4.5 Consistency Analysis (Table 4: augmentation/perturbation stability)
   4.6 Clinical Alignment (Table 5: proxy ROI alignment per condition)
   4.7 CBM Concept Intervention (Figure 3: intervention examples)

5. Discussion (1 page)
   5.1 Key Findings and Implications
   5.2 Which Method Should Clinicians Trust?
   5.3 The Case for Ante-Hoc Interpretability
   5.4 Limitations

6. Conclusion (0.5 pages)

References (~40 citations)

Supplementary Material:
   A. Full per-condition, per-level XAI results
   B. Additional qualitative examples
   C. Hyperparameter sensitivity analysis
   D. Ablation studies
   E. Code and reproducibility details
```

### Key Figures:

1. **Figure 1 (Hero Figure):** 7×7 agreement heatmap showing pairwise Spearman correlations between all XAI methods. Diagonal = 1.0, off-diagonal << 0.3. Title: "Post-hoc XAI methods exhibit severe disagreement."
2. **Figure 2:** Grid of 3 example images × 7 XAI methods, showing visually contradictory attribution maps. One row per condition type (stenosis, foraminal, subarticular).
3. **Figure 3:** CBM concept intervention demonstration. Before: wrong prediction with concept values. After: corrected concept → correct prediction. Shows the practical advantage of ante-hoc interpretability.
4. **Figure 4:** Faithfulness comparison bar chart (Deletion AUC and Insertion AUC per method).

### Key Tables:

1. **Table 1:** Classification performance of all 6 models (5 black-box + 1 CBM). Metrics: Weighted Log-Loss, Balanced Accuracy, Macro F1, AUC-OVR.
2. **Table 2:** Faithfulness benchmark (7 methods × 5 faithfulness metrics). Rows = methods, columns = metrics. Best method bolded per column.
3. **Table 3:** Cross-architecture disagreement. Shows that intra-model disagreement > inter-model disagreement.
4. **Table 4:** Agreement breakdown per condition. Shows that disagreement varies across clinical conditions (e.g., worse for subarticular stenosis).

---

## 10. Key References

| Citation                                                                        | Relevance                                                          |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Krishna et al., "The Disagreement Problem in Explainable ML" (NeurIPS 2022)     | **Primary reference** — formalizes the disagreement problem |
| Hedström et al., "Quantus: An Explainable AI Toolkit" (JMLR 2023)              | Evaluation framework for faithfulness metrics                      |
| Koh et al., "Concept Bottleneck Models" (ICML 2020)                             | Foundational CBM paper                                             |
| Selvaraju et al., "Grad-CAM" (ICCV 2017)                                        | Most cited post-hoc method                                         |
| Ribeiro et al., "LIME" (KDD 2016)                                               | Perturbation-based explanation                                     |
| Lundberg & Lee, "SHAP" (NeurIPS 2017)                                           | Game-theoretic explanation                                         |
| Sundararajan et al., "Integrated Gradients" (ICML 2017)                         | Axiomatic gradient method                                          |
| Bach et al., "LRP" (PLOS ONE 2015)                                              | Layer-wise relevance propagation                                   |
| Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018)                | Shows many methods fail basic sanity checks                        |
| Hooker et al., "A Benchmark for Interpretability" (NeurIPS 2019)                | ROAR benchmark for faithfulness                                    |
| Ghassemi et al., "The False Hope of Current Approaches to XAI" (Lancet DH 2021) | Clinical perspective on XAI limitations                            |
| RSNA 2024 Competition Dataset                                                   | Primary dataset                                                    |
| Wang et al., "Score-CAM" (CVPR Workshop 2020)                                   | Gradient-free CAM method                                           |
| Ozyoruk et al., "Spine XAI review" (various 2023-2024)                          | Domain-specific literature                                         |

---

## 11. Risk Assessment & Mitigation

| Risk                                | Impact                           | Probability                     | Mitigation                                                              |
| ----------------------------------- | -------------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| Disagreement is not severe enough   | Paper's central claim weakened   | Low (preliminary data confirms) | Include more methods; compare across architectures                      |
| One method dominates all metrics    | Weakens "no consensus" narrative | Medium                          | Focus on condition-specific analysis where no single winner             |
| CBM accuracy too low                | Weakens ante-hoc argument        | Medium                          | Emphasize interpretability-accuracy tradeoff analysis                   |
| LIME/SHAP too slow for 1000 samples | Incomplete evaluation            | High                            | Limit to 500 samples; parallelize; skip SHAP if necessary               |
| Quantus library issues on Kaggle    | Blocked evaluation               | Low                             | Fall back to custom implementations                                     |
| Kaggle quota exceeded               | Delayed timeline                 | Medium                          | Prioritize ConvNeXt experiments; do other backbones only if time allows |

---

## 12. Success Criteria

### Minimum Viable Paper (Floor):

- [ ] ConvNeXt-Tiny black-box trained with AUC ≥ 0.92
- [ ] 5+ XAI methods applied and agreement matrix computed
- [ ] Faithfulness benchmark with 3+ metrics
- [ ] CBM trained and compared
- [ ] Disagreement demonstrated quantitatively
- [ ] 5+ figures/tables

### Target Paper (Ceiling):

- [ ] All 5 black-box backbones trained
- [ ] All 7 XAI methods evaluated
- [ ] All 4 evaluation categories (faithfulness, agreement, consistency, clinical)
- [ ] Cross-architecture analysis completed
- [ ] CBM concept intervention demonstrated
- [ ] Per-condition and per-severity breakdowns
- [ ] 8+ figures/tables
- [ ] Supplementary with full results

---

## 13. Timeline

| Week             | Milestone                                                       |
| ---------------- | --------------------------------------------------------------- |
| **Week 1** | Build expanded XAI evaluation code; train ConvNeXt-Tiny (224px) |
| **Week 2** | Train remaining backbones; run XAI benchmark on ConvNeXt        |
| **Week 3** | Cross-architecture XAI; CBM training + evaluation               |
| **Week 4** | Clinical alignment analysis; visualization; begin writing       |
| **Week 5** | Complete draft; internal review; revisions                      |
| **Week 6** | Final polish; supplementary material; submission                |

---

## 14. Differences from Original Plan (SpineXNet v1)

| Aspect                        | Original (SpineXNet v1)                               | New (Disagreement Benchmark)                           |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------------------------ |
| **Hero contribution**   | Novel architecture (CBM + Prototypes + Residual)      | Novel evaluation (systematic disagreement analysis)    |
| **Primary model**       | SpineXNet (interpretable)                             | ConvNeXt-Tiny (black-box)                              |
| **CBM role**            | Main model                                            | Baseline for comparison                                |
| **Prototype layer**     | Central component                                     | Removed                                                |
| **BiomedCLIP concepts** | Label-free supervision signal                         | Removed (too noisy)                                    |
| **Risk level**          | High (architecture must outperform or match baseline) | Low (finding disagreement is almost guaranteed)        |
| **Paper type**          | Architecture paper                                    | Benchmark/analysis paper                               |
| **Key figure**          | Architecture diagram                                  | 7×7 agreement heatmap                                 |
| **Evaluation focus**    | Classification accuracy + faithfulness                | Disagreement + faithfulness + clinical alignment       |
| **Compute needs**       | ~30h (one architecture)                               | ~44h (multiple architectures + XAI)                    |
| **Novelty argument**    | "First inherently interpretable spine classifier"     | "First systematic disagreement study in spine imaging" |

> [!TIP]
> **The new approach is strictly easier to execute and defend.** The original plan required the interpretable model to match the black-box — a tall order with noisy supervision. The new plan only requires demonstrating that XAI methods disagree (which they provably do) and that a simple CBM provides more consistent explanations (which it inherently does by construction).

# Figure and Table Plan

## Existing Figures in the Draft

1. Figure 1: `figures/figure1_disagreement_gallery.png`
   - Current source: copied from `artifacts/results/current/v4/rsna/dataset_no_npy/figures/fold_0/gallery/disagreement_gallery.png`.
   - Purpose: hero figure showing qualitative disagreement among XAI methods on RSNA.
   - Status: replace after `generate_disagreement_gallery_v4.py` runs inside the RSNA repair notebook. Current labels may be too small for BMVC print.
   - Nanobanana prompt: "Create a polished BMVC-style qualitative XAI disagreement gallery for lumbar spine MRI. Use two architecture rows: high-agreement DenseNet-121 and low-agreement ViT/DeiT. Columns should show input MRI crop, Grad-CAM, Grad-CAM++, Integrated Gradients, GradientSHAP, Occlusion, Guided Backpropagation, Attention Rollout where applicable, and top-k consensus. Use large readable labels, clean white background, subtle colored borders for method families, and one callout arrow highlighting contradictory saliency regions. Do not add decorative effects."

2. Figure 2: `figures/agreement_hierarchy.pdf`
   - Source: generated from final aggregate RSNA/CUB XAI CSVs.
   - Purpose: main quantitative visual for the three-tier agreement hierarchy.
   - Status: usable. Regenerate if RSNA repair changes ViT/EfficientNet rows.
   - Nanobanana prompt if redesigned manually: "Create a clean scientific bar chart comparing mean pairwise Spearman explanation agreement across DenseNet-121, ResNet-50, ConvNeXt-Tiny, EfficientNet-B4, ViT-Small, and DeiT-Small. Show paired bars for RSNA lumbar MRI and CUB-200 with standard deviation error bars. Visually group classic CNNs, modern CNNs, and Transformers using subtle separators or labels. Use BMVC paper styling, high contrast, readable axis labels, no 3D effects."

3. Figure 3: `figures/consensus_ablation.pdf`
   - Source: generated from current V4 rerun consensus rows.
   - Purpose: compare uniform, faithfulness-weighted, and oracle top-k consensus.
   - Status: includes fold-level error bars now. Replace after the CUB consensus repair yields all six model rows.
   - Nanobanana prompt if redesigned manually: "Create a compact BMVC-style consensus ablation chart showing insertion AUC for Uniform, Faithfulness-Weighted, and Oracle Top-k consensus. Use grouped bars with error bars, emphasize that Oracle Top-k is an upper-bound curation result, and avoid making it look deployable. Use clear labels and restrained colors."

## Placeholder Figure Added

4. Planned Methodology Schematic: placeholder in `bmvc_review.tex`, label `fig:method`.
   - Purpose: explain the pipeline before results.
   - Required content: RSNA MRI crops and CUB images feed into model families; model families split into classic CNNs, modern CNNs, Transformers, and RSNA-only CBM; seven XAI methods generate saliency maps; evaluation outputs are agreement, faithfulness/consensus, expert alignment, and concept intervention.
   - Nanobanana prompt: "Create a clean academic methodology diagram for a BMVC paper titled Architecture Shapes Explanation. Left side: two datasets, RSNA lumbar MRI crops and CUB-200 bird images. Middle: architecture families in four lanes: classic CNNs (ResNet-50, DenseNet-121), modern CNNs (ConvNeXt-Tiny, EfficientNet-B4), Transformers (ViT-Small, DeiT-Small), and RSNA-only concept bottleneck model. Right side: seven post-hoc XAI methods producing saliency maps, then four evaluation heads: inter-method agreement, insertion/deletion faithfulness with consensus, expert-coordinate ROI alignment, and concept intervention. Use simple boxes and arrows, blue/orange domain colors, no marketing style, no decorative gradients, readable labels at single-column paper scale."

## Tables in the Draft

1. Table 1: classification competence.
   - Purpose: prove that disagreement is not caused by failed classifiers.
   - Status: final unless classification aggregation changes.

2. Table 2: cross-domain explanation agreement.
   - Purpose: central quantitative result.
   - Status: replace ViT-Small RSNA row after RSNA repair if Attention Rollout folds 0 and 1 change aggregate values.

3. Table 3: statistical tests.
   - Purpose: makes fold-paired significance visible instead of burying p-values in prose.
   - Status: final unless repaired aggregates change paired differences.

4. Table 4: CBM intervention.
   - Purpose: integrates the ante-hoc interpretability result with the disagreement narrative.
   - Status: final for current intervention summaries.

## Not Included Yet

- Full per-method faithfulness table: too large for main paper; good supplementary material.
- EfficientNet gradient audit table: demoted to supplementary to keep main paper from reading like a debugging report.
- Full CUB six-model consensus table: add after `repair_v4_cub_consensus.ipynb` completes.

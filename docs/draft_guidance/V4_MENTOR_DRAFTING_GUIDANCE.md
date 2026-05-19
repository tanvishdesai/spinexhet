# V4 Mentor Drafting Guidance

Last updated: 2026-05-17

This file consolidates the writing, framing, and paper-structure advice from
`v4-mentor-review.md`. It is meant to guide the manuscript draft after the
remaining numerical repair notebooks have been run.

## Core Manuscript Position

Use the title and thesis:

> Architecture Shapes Explanation: A Cross-Domain Benchmark of Post-Hoc XAI Disagreement

Do not use "governs" as the central verb. The data support a strong shaping
effect, not a deterministic architecture ranking. The cleanest claim is a
three-tier hierarchy:

| Tier | Models | Draft framing |
|---|---|---|
| High agreement | DenseNet-121, ResNet-50 | Classic CNNs produce comparatively stable explanations. |
| Domain-sensitive middle | ConvNeXt-Tiny, EfficientNet-B4 | Modern CNNs shift with domain and should not be forced into a strict ranking. |
| Low agreement | ViT-Small, DeiT-Small | Patch-based Transformers consistently show the lowest inter-method agreement. |

The decisive V4 upgrade is DeiT-Small. Phrase the Transformer claim around
ViT and DeiT together, not ViT alone:

> The low-agreement pattern appears across two independently trained
> Transformer architectures, including DeiT-Small, which is trained through
> distillation from a CNN teacher. This makes the result difficult to dismiss
> as a ViT-specific artifact.

## Figure Strategy

Figure 1 must be a qualitative disagreement gallery. Mentors repeatedly framed
this as the highest-ROI presentation fix for BMVC.

Required visual content:

- One RSNA example showing the same input, same prediction, and multiple XAI
  methods.
- A high-agreement CNN row, preferably DenseNet-121 or ResNet-50.
- A low-agreement Transformer row, preferably ViT-Small after fold 0/1 rollout
  is repaired.
- Include Top-k consensus as the final column or a separate panel.
- If space allows, add one CUB bird example to show the phenomenon is visually
  recognizable outside MRI.

The caption should be standalone. It should say that the maps are generated for
the same prediction and that disagreement is architectural rather than
sample- or label-specific.

## Abstract Guidance

The abstract should be concrete and number-driven:

- Name both datasets: RSNA lumbar spine MRI and CUB-200.
- State the scale: 8 architectures, 7 XAI methods, 5-fold CV.
- State the headline contrast: Transformers around rho = 0.20 versus classic
  CNNs around rho = 0.41-0.52.
- Mention DeiT as the second Transformer.
- Mention Top-k consensus beating full faithfulness-weighted consensus.
- Frame expert IoU <= 0.28 as clinical inadequacy of post-hoc XAI, not success.
- End with a practical recommendation: use curated consensus or ante-hoc
  interpretability for safety-critical settings.

Avoid opening with "largest benchmark." LATEC makes that claim unsafe.

## Introduction Outline

Recommended length: about 1.5 pages.

1. Start with the clinical trust problem: radiology models need explanations,
   but post-hoc saliency can create false confidence.
2. Define disagreement plainly: different XAI methods explain the same model
   prediction with contradictory regions.
3. State the gap: prior work studies XAI reliability, but not whether
   disagreement is shaped by architecture class and reproduced across domains.
4. Introduce the two-domain design: RSNA as safety-critical clinical evidence,
   CUB-200 as natural-image generalization.
5. End with contributions as specific evidence claims, not broad promises.

Contribution bullets should include:

- Cross-domain architecture-disagreement benchmark across RSNA and CUB-200.
- Two-Transformer validation with ViT-Small and DeiT-Small.
- Three-tier architecture hierarchy rather than a brittle strict ranking.
- Consensus ablation: uniform, faithfulness-weighted, and top-k.
- Expert-coordinate clinical alignment and CBM intervention analysis.

## Related Work Positioning

The LATEC paragraph must be written defensively and respectfully. Use
"orthogonal" rather than "better" or "larger."

Suggested wording:

> Recent large-scale XAI benchmarks such as LATEC evaluate explanation methods
> and metrics across broad method-metric grids. Our focus is different: we ask
> whether inter-method explanation disagreement is shaped by architecture class
> and whether the resulting hierarchy reproduces across clinical and natural
> image domains. Thus, while LATEC studies metric reliability at scale, our work
> studies architecture-conditioned disagreement, cross-domain stability, and
> clinical expert-coordinate alignment.

For "Hypothesis Class Determines Explanation", the distinction should be:

- They emphasize how model class affects explanations.
- This project emphasizes inter-method disagreement within the same model and
  tests whether that disagreement hierarchy reproduces across domains.
- Do not pretend there is no overlap. Acknowledge it and then narrow the gap.

Related work subsections recommended by mentors:

- XAI disagreement and saliency reliability.
- XAI benchmarks and metric reliability, including LATEC.
- Architecture and inductive-bias effects on explanations.
- Concept bottleneck and ante-hoc interpretability.
- Clinical explainability and the false-hope critique.

## Methods Section

Recommended length: about 2 pages.

The Methods section should be calm and reproducible rather than promotional.
Include:

- Datasets: RSNA and CUB-200, with folds and image size.
- Architectures: group as classic CNN, modern CNN, Transformer, CBM.
- XAI methods: GradCAM, GradCAM++, Integrated Gradients, GradientSHAP,
  Occlusion, Guided Backpropagation, and Attention Rollout for Transformers.
- Consensus strategies: uniform, faithfulness-weighted, top-k.
- Metrics: agreement, insertion/deletion faithfulness, expert IoU, consensus
  metrics, significance tests.

CBM defense wording:

> We use deterministic metadata-derived concepts to isolate the effect of
> architecture and explanation method without introducing concept-prediction
> noise. We therefore interpret the CBM as an ante-hoc controlled alternative,
> not as proof that visual concept discovery has been solved.

This avoids overstating the CBM as a fully learned semantic concept model.

## Results Section

Recommended length: about 3.5 pages.

Order the results like this:

1. Classification competence: short table proving all models are valid.
2. Main agreement hierarchy: this is the central table.
3. Cross-domain reproduction: RSNA vs CUB rank correlation and tier stability.
4. Consensus resolution: Top-k vs FW vs uniform, with p-values.
5. Clinical reality: low expert IoU for post-hoc XAI, then CBM intervention.
6. Supplementary/exploratory: feature coherence, EfficientNet diagnostic if
   needed, k-sensitivity if run.

Do not bury p-values. Mentors specifically noted that "significance tests
claimed but not shown" is worse than not claiming them. Include a compact table
with DenseNet/ResNet vs ViT/DeiT, RSNA-CUB rank correlation, Top-k vs FW, and FW
vs uniform.

## Consensus Claims

The clean claim is:

> Top-k consensus, which averages only the most faithful methods, consistently
> outperforms uniform averaging and full faithfulness-weighted averaging in the
> tested RSNA setting.

If the optional CUB consensus repair is run, extend the claim to CUB. If it is
not run, explicitly restrict CUB consensus language:

> CUB consensus results are reported for the newly rerun models and treated as
> preliminary; the full consensus ablation is evaluated on RSNA.

Do not imply every CUB model has consensus variants until the optional repair
notebook has been executed.

## EfficientNet Language

EfficientNet IG/GradientSHAP must not remain ambiguous.

If the audit shows the maps are not identical:

> The EfficientNet IG/GradientSHAP equality observed after rounding was not a
> copy-paste or tensor-broadcasting error. A baseline-sensitivity audit found
> non-identical maps across all folds, with high but imperfect map correlation.
> We therefore report EfficientNet as a gradient-fragile mid-tier architecture
> rather than excluding it.

If the audit shows baseline sensitivity changes the ranking:

> EfficientNet gradient-based scores are baseline-sensitive; we report the
> corrected baseline-controlled values and move the baseline sensitivity table
> to the supplement.

If the maps are truly identical:

> We treat this as a gradient pathology, include a gradient-norm diagnostic, and
> avoid using the affected gradient methods as evidence for EfficientNet
> consensus superiority.

## ViT Attention Rollout Language

After rerunning folds 0 and 1, remove all apologetic language about missing
rollout. The final paper should not contain an "old checkpoint" footnote.

If for some reason folds 0 and 1 still cannot be repaired, restrict the claim:

> ViT Attention Rollout is reported on the three folds for which the rollout
> hook succeeded; DeiT provides the full five-fold Transformer-native analysis.

But the preferred final state is five ViT folds with Attention Rollout.

## Clinical Framing

Do not spin expert IoU around 0.28 as good alignment. The mentors converged on
the opposite framing:

> Even the best post-hoc saliency maps overlap expert-coordinate ROIs less than
> 30 percent of the time, indicating that post-hoc XAI remains clinically weak
> despite strong classification performance.

This makes the CBM intervention result more natural:

> The CBM is not merely another high-IoU saliency method; it enables a different
> interaction mode in which clinicians can intervene on concepts.

## Discussion Section

Recommended themes:

- Architecture shapes explanation consistency, but the evidence is empirical,
  not causal proof.
- The three-tier hierarchy is more defensible than a strict architecture
  ranking.
- Transformers may disagree because patchification and self-attention create
  non-local gradient and attention pathways, so different XAI families probe
  different parts of the computation.
- DeiT's CNN-teacher distillation not eliminating low agreement suggests that
  the issue is not simply a training-data or teacher-supervision artifact.
- Low expert IoU cautions against uncritical clinical deployment of saliency.
- Consensus helps, but curated Top-k consensus is better than averaging every
  available method.

Important hedge:

> We do not claim that architecture alone causally determines explanations.
> Rather, across two domains and multiple folds, architecture class is a strong
> organizing factor for inter-method explanation agreement.

## Limitations

Mention these openly:

- Only two Transformer architectures; Swin/CaiT would strengthen family-level
  claims.
- CBM concepts are deterministic/metadata-derived, not fully learned visual
  concepts.
- Feature coherence is exploratory and does not explain Transformer behavior.
- RSNA images cannot be fully redistributed because of dataset restrictions.
- Expert alignment is coordinate-derived rather than dense radiologist masks.

## Venue Framing

For BMVC:

- Lead with the CUB + architecture-class finding.
- Treat RSNA as the safety-critical validation domain.
- Figure 1 must be visually strong.
- Emphasize the fundamental CV claim: architecture class shapes explanation
  consistency.

For MIDL:

- Lead with clinical trust, expert annotations, and concept intervention.
- Treat CUB as generalization evidence.
- Emphasize false confidence from post-hoc XAI and the need for ante-hoc or
  consensus-based alternatives.

For Pattern Recognition / journal route:

- Expand discussion and add optional mechanistic analyses such as k-sensitivity,
  accuracy-vs-disagreement, and patchification/attention hypotheses.

## Do Not Do

- Do not add major new datasets or many new architectures before drafting.
- Do not claim to be larger than LATEC.
- Do not claim strict architecture ranking.
- Do not oversell CBM as learned visual concept discovery.
- Do not leave EfficientNet or ViT rollout unresolved in the final text.
- Do not hide incomplete CUB consensus rows; either repair them or restrict the
  claim.

## Six-Day Draft Plan

| Day | Writing task | Parallel experiment task |
|---|---|---|
| 1 | Introduction + Related Work skeleton | EfficientNet audit and gallery generation |
| 2 | Methods | ViT fold 0/1 rollout repair |
| 3 | Results tables | Optional CUB consensus repair |
| 4 | Discussion + Limitations | k-sensitivity or gradient-norm diagnostic if needed |
| 5 | Abstract, captions, p-value checks | Final aggregation |
| 6 | Internal review and claim audit | Code/package freeze |

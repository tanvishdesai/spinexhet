# Venue Notes: BMVC 2026

Target venue: British Machine Vision Conference 2026.

Official requirements checked on 2026-05-17:
- Review submissions must use the official BMVC 2026 template.
- Review submissions are anonymous and use `\bmvcreviewcopy{??}` with the OpenReview paper number.
- Main paper length is 14 pages excluding references.
- Typesetting is expected through PDFLaTeX.

Style synthesis used for the draft:
- BMVC papers reward a compact problem statement, a visually strong first figure, and concrete quantitative claims.
- Tables should use booktabs-style formatting and report mean/std over folds where available.
- Contribution language should be evidence-bearing rather than promotional.
- Related work should position against large XAI benchmarks such as LATEC without claiming scale superiority.
- The paper should read as a computer-vision benchmark first, with clinical MRI as the safety-critical validation domain.

Draft implications:
- The title uses "Shapes" rather than "Governs" to match the three-tier evidence.
- Figure 1 is a disagreement gallery.
- Main tables focus on classification competence, agreement hierarchy, and significance-supported cross-domain reproduction.
- Limitations explicitly discuss two Transformer architectures, coordinate-derived expert ROIs, deterministic CBM concepts, and incomplete full-CUB top-k reruns.

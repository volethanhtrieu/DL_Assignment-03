# A3 M2 publication verification record

Status: **LOCAL PACKAGE VERIFIED — REMOTE PUBLICATION NOT CONFIRMED** (17 Nov 2026).

- M1 protocol retained by construction; this patch has no `A3-M1/` or `docs/m1/` changes.
- Source draft: `CO3133_Assignment 3_M2_Report.tex`; compiled PDF: matching `CO3133_Assignment 3_M2_Report.pdf`.
- Nine REAL M2 development runs: image, text, concat × seeds 42/43/44; 986 dev predictions each, no synthetic smoke results and no final test metrics.
- The `best.pt` weights are feature-head checkpoints; corresponding upstream frozen encoder versions are documented in `feature_metadata.json`.
- `python scripts/verify_m2_evidence.py` recomputes per-run accuracy, macro-F1 and confusion matrices, and validates SHA256 for published evidence.
- Raw images, tweet texts, model downloads and large feature NPZ files are deliberately excluded.
- Both A3 and landing-page publications require an **authorized manual push**. The current ChatGPT GitHub connection returned HTTP 403 (`Resource not accessible by integration`) while attempting `create_branch`; no remote file was changed by this publication task.
- After pushing, check actual GitHub file URLs, GitHub Pages PDF link, landing M2 status, M2 disclosure, and LMS submission. Do not mark these tasks PASS before verification.
- Dataset approval waiver is only group-reported; formal decision/waiver evidence and LMS submission must be checked by students.

M2 meets the handbook's *draft milestone model scope* (unimodal A/B and simple fusion with quantitative results). Improved fusion, ablations, final test evaluation, slides/video and full final report are later M3 deliverables.

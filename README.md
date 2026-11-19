# DL_Assignment-03

## Group Goat — M1 Dataset, M2 Experiments and M3 Evaluation

**CO3133 - Deep Learning and Its Applications - Semester 261**  
Ho Chi Minh City University of Technology - VNU-HCM, Faculty of Computer Science and Engineering.  
Instructor: **Lê Thành Sách**. M1 due **18 November 2026, 23:59 GMT+7**.

## Project Authorship & Individual Contribution

| Member Name | Student ID | Assignment 03 Contribution (%) | Role & Responsibilities |
|---|---|---:|---|
| Hoàng Trọng Huy Minh | 2452743 | **0%** | Assignment 1 lead. |
| Võ Lê Thành Triệu | 2453297 | **0%** | Group team leader and Assignment 2 lead. |
| **Nguyễn Huỳnh Nam Quốc** | **2453089** | **100%** | **Assignment 3 sole academic contributor:** dataset research, validation, remediation, proposal, reproducibility, documentation and verification. |
### Current scope

M1 dataset validation and remediation are preserved. M2 completed
nine development-only runs. M3 completed local evaluation of five
model configurations across three seeds, fusion ablation, inference
benchmarking, gate analysis and checkpoint-based Gradio verification.
M3 GitHub publication completed on 19 November 2026. Video and LMS submission remain separate actions.

- [Assignment page](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-03/index.html) · [Main landing page](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/). These are deployment destinations; actual live verification is recorded in the publication checklist.
- **M1 report:** [PDF](docs/m1/CO3133_Assignment%203_M1_Report.pdf) · [LaTeX source](docs/m1/CO3133_Assignment%203_M1_Report.tex) (academic LaTeX layout; M1 scope and dataset unchanged).
- **Original M1 proposal:** [PDF](docs/m1/CO3133_A3_M1_GroupGoat_Dataset_Proposal.pdf) · [Editable DOCX](docs/m1/CO3133_A3_M1_GroupGoat_Dataset_Proposal.docx) · [M1 document index](docs/m1/README.md).
- [Earlier pre-publication compliance PDF](docs/m1/CO3133_A3_M1_GroupGoat_Compliance_Checklist.pdf) · [Current publication checklist](docs/m1/PUBLICATION_CHECKLIST.md).
- [AI Usage Disclosure](AI_USAGE.md) · [Prompt summaries](docs/m1/PROMPT_LOG.md).
- [Dataset provenance, EDA and preprocessing/split rules](docs/m1/DATASET.md).

### Locked dataset

CrisisMMD v2.0 humanitarian image-text classification; five-class mirror-derived custom version `crisismmd5-mirror-remediation-20261104-v1`. Primary `filtered_original`: **7,938 pairs (6,001 / 986 / 951)**. The 141 exclusions are **7 annotation discrepancies + 9 placeholders + 125 duplicate/leakage-related rows**. The alternative repartition is backup only.

Original archive equivalence is unverified (prior HTTP 403); current primary annotation differences are quarantined without relabelling. Test is custom filtered, and the rare class has **9 dev / 9 test** records. These limitations remain explicit. Raw data is not bundled. Upstream [CC BY-NC-SA 4.0 license](A3-M1/provenance/licenses_by-nc-sa_4.0_legalcode.txt) and source/changes attribution apply to derived dataset materials.

### Reproduce primary membership without raw data

Requires Python 3.10+ standard library only. From the repository root:

```bash
python scripts/reproduce_primary.py --out /tmp/co3133-a3-m1-replay
```

This replays the **already audited group metadata**, verifies published SHA256 locks, inclusion/exclusion membership, all five classes and declared identity isolation. It does **not** independently re-audit raw images/text or prove archive equivalence. It never trains a model and writes outside the repo.

If the earlier complete source-audit manifest already exists locally, additionally rebuild all three complete CSVs and verify byte equality:

```bash
python scripts/reproduce_primary.py --audited-manifest /absolute/path/benchmark_manifest.csv --out /tmp/co3133-a3-m1-full-replay
```

The full manifest contains tweet text and is intentionally excluded from Git. Original source-audit/download scripts and actual audit dependencies are retained under `A3-M1/scripts` and `A3-M1/requirements.txt`. They require the pinned raw mirror/primary annotations locally and write local audit outputs. Do not run them merely to replay membership; publication reused completed audit evidence without redownloading data.

Dataset source paths, versions and payload checksums are in `A3-M1/provenance/metadata_downloads.json`, `mirror_downloads.json`, and `A3-M1/DATASET_LOCK.json`. The source scripts use these historical pins; if fetching sources for a fresh audit, use a separate working copy and retain raw files outside publication. Dataset preprocessing/split code is available now; model train/evaluate commands and checkpoint reconstruction are later M2/M3 work, not implemented M1 commands.

### M2 REAL development results (17 November 2026)

The M2-only development experiments are now complete under the unchanged primary split. **No held-out test evaluation was performed during M2.** Subsequent M3 test results are presented below.

| Model | Dev macro-F1 mean ± sample SD | Dev accuracy mean ± sample SD | Trainable head parameters |
|---|---:|---:|---:|
| Image-only ResNet-18 | 0.5746 ± 0.0068 | 0.6930 ± 0.0077 | 66,309 |
| Text-only DistilBERT | 0.6811 ± 0.0016 | 0.7762 ± 0.0064 | 99,077 |
| Concatenation fusion | **0.8116 ± 0.0041** | **0.8540 ± 0.0044** | 197,637 |

- [M2 scientific report PDF](docs/m2/CO3133_Assignment%203_M2_Report.pdf) · [LaTeX source](docs/m2/CO3133_Assignment%203_M2_Report.tex).
- [M2 dev summary](results/m2_dev/DEV_RESULTS.md) · [Nine model logs and trained head checkpoints](artifacts/m2/experiments/primary/) · [Run verification](docs/m2/M2_PUBLICATION_CHECKLIST.md).
- To recalculate all 9 dev result rows and validate published files: `python scripts/verify_m2_evidence.py`.
- M2-only reproducibility notebook: `m2m3/CO3133_A3_M2_GPU_Execution.ipynb`. The feature cache and raw CrisisMMD images/text are not published.

**Limitations:** This is not a held-out test result; the rarest class has only 9 development examples; original agreed-label archive equivalence remains unverified. M3 work is separate.

### M3 evaluation - Local results

M3 uses the custom `filtered_original` CrisisMMD dataset:
**6,001 train / 986 dev / 951 test pairs**.

Five model configurations were evaluated over seeds 42, 43 and 44
(15 runs in total).

| Model | Test macro-F1 (mean +/- SD) |
|---|---:|
| Image-only | 0.6127 +/- 0.0122 |
| Text-only | 0.6666 +/- 0.0076 |
| Concatenation fusion | 0.7965 +/- 0.0176 |
| Fixed gate | 0.7869 +/- 0.0145 |
| Learned gated fusion | **0.7986 +/- 0.0027** |

The learned gate exceeds concatenation by only 0.0021 mean macro-F1.
Paired group-bootstrap confidence intervals include zero, so
statistical superiority has not been established.

#### M3 deliverables

- [Public-review report](docs/m3/CO3133_A3_M3_Public_Review.pdf)
- [M3 documentation](docs/m3/README.md)
- [Experimental results](results/m3/README.md)
- [M3 source code](m2m3/m3/README.md)

- [M3 per-seed test metrics](results/m3/test/per_seed_metrics.csv)
- [M3 trained-head checkpoints](artifacts/m3/experiments/primary/)
- [M3 evidence verification](scripts/verify_m3_evidence.py)
- [M3 SHA256 manifest](docs/m3/SHA256SUMS.txt)
- [AI usage disclosure](AI_USAGE.md)

The Gradio demo has been validated locally with original checkpoints.
It has not been publicly deployed. Raw crisis media and internal case galleries remain excluded. The six M3 gated/fixed-gate head checkpoints, fifteen per-seed test metric files, run evidence and SHA256 ledger are now published. The nine original M2 checkpoints are reused without duplication. Verify with `python scripts/verify_m3_evidence.py`.

#### Remaining milestones

- M2 deadline: 25 November 2026.
- M3 deadline: 2 December 2026, 23:59 GMT+7.
- Published: Assignment 3 commit `26444f5` and landing-page commit `41d8479`. Both GitHub Pages locations passed live content verification on 19 November 2026.
- Pending: video, LMS submission, and rights/privacy clearance for unreleased presentation assets.

### Publication and exclusions

Only report artifacts, aggregate EDA, identifiers/hashes, selected provenance/evidence and scripts are tracked. Raw tweets/images, Parquet/archives, image-contact caches, environments, credentials and transient logs are excluded. Existing history and unrelated A1/A2 work are preserved. No force push is used. See [publication status](docs/m1/PUBLICATION_CHECKLIST.md) for actual commit/deployment verification and remaining student/LMS actions.

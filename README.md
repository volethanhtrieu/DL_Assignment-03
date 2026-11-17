# DL_Assignment-03

## Group Goat — M1 Proposal and M2 Experimental Draft

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

M1 proposal and data-integrity evidence remain locked. **M2 has nine real, development-only runs (image/text/concatenation × seeds 42/43/44).** M3 improved fusion, ablations, final test evaluation, slides/video and LMS submission are not claimed as complete. The group reports prior dataset approval was waived; this is not a formal approval decision.

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

The M2-only development experiments are now complete under the unchanged primary split. **No test evaluation has been performed.**

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

### M2 and M3 roadmap

M2 (25 Nov): image-only, text-only and concatenation quantitative **development** results are completed as above; publication and LMS submission remain student actions. M3 (2 Dec): gated fusion, controlled gate ablation, test evaluation, modality support/conflict review, final report/slides/video and end-to-end cost are future work only. The unchanged proposal contains the detailed experiment design and estimates; its disclosure tool-identification cell is corrected for ChatGPT Work/Codex.

### Publication and exclusions

Only report artifacts, aggregate EDA, identifiers/hashes, selected provenance/evidence and scripts are tracked. Raw tweets/images, Parquet/archives, image-contact caches, environments, credentials and transient logs are excluded. Existing history and unrelated A1/A2 work are preserved. No force push is used. See [publication status](docs/m1/PUBLICATION_CHECKLIST.md) for actual commit/deployment verification and remaining student/LMS actions.

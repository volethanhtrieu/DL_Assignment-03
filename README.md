# DL_Assignment-03

## Group Goat M1 Dataset Proposal

**CO3133 - Deep Learning and Its Applications - Semester 261**  
Ho Chi Minh City University of Technology - VNU-HCM, Faculty of Computer Science and Engineering.  
Instructor: **Lê Thành Sách**. M1 due **18 November 2026, 23:59 GMT+7**.

| Member | Student ID | Role |
|---|---|---|
| Hoàng Trọng Huy Minh | 2452743 | Member |
| Võ Lê Thành Triệu | 2453297 | Team leader |
| Nguyễn Huỳnh Nam Quốc | 2453089 | Assignment 3 lead |

### Current scope

M1 dataset/proposal and audit evidence are prepared. **No model training, M2/M3 results, checkpoints, slides/video, LMS submission or completed student final review are claimed.** The group reports prior dataset approval was waived; this is not a formal approval decision.

- [Assignment page](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/assignment-03/index.html) · [Main landing page](https://volethanhtrieu.github.io/Deep-Learning-project-main-page/). These are deployment destinations; actual live verification is recorded in the publication checklist.
- [Proposal PDF](docs/m1/CO3133_A3_M1_GroupGoat_Dataset_Proposal.pdf) · [Editable DOCX](docs/m1/CO3133_A3_M1_GroupGoat_Dataset_Proposal.docx).
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

### M2 and M3 roadmap

M2 (25 Nov): image-only, text-only and concatenation quantitative results. M3 (2 Dec): gated fusion, controlled gate ablation, modality support/conflict review, errors, cost, final report/slides/video and checkpoints/reproduction instructions. These are plans. The unchanged proposal contains the detailed experiment design and estimates; its disclosure tool-identification cell is corrected for ChatGPT Work/Codex.

### Publication and exclusions

Only report artifacts, aggregate EDA, identifiers/hashes, selected provenance/evidence and scripts are tracked. Raw tweets/images, Parquet/archives, image-contact caches, environments, credentials and transient logs are excluded. Existing history and unrelated A1/A2 work are preserved. No force push is used. See [publication status](docs/m1/PUBLICATION_CHECKLIST.md) for actual commit/deployment verification and remaining student/LMS actions.

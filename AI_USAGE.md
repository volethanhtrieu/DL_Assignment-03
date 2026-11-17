# Assignment 3 AI Usage Disclosure

Group Goat - CO3133, Semester 261. Recorded 4 November 2026. Scope: M1 only.

## Observed use

| Handbook Section 5.2 field | Evidence-based record |
|---|---|
| Tool name and version/model | **ChatGPT Work with the Codex assistant** in this conversation. No independently verified underlying model/build identifier is available; no specific GPT version or separate IDE/CLI/Copilot use is asserted. |
| Member who used it | **Nguyễn Huỳnh Nam Quốc (2453089)**, confirmed by the requester during proposal preparation. Other members' AI use is not attested here. |
| Time or stage | 4 November 2026: M1 dataset research, validation/remediation assistance, proposal preparation and publication maintenance. |
| Purpose | Source/handbook inspection, audit and split scripts, evidence interpretation, English drafting, document rendering, GitHub/Pages maintenance and checks. No model training or inference experiment. |
| Affected work | M1 proposal DOCX/PDF; dataset provenance, EDA and split records; validation/replay scripts; README; A3 page; landing-page A3 entries; this disclosure and publication checklist. |
| Representative prompt | See [representative prompt summaries](https://github.com/volethanhtrieu/DL_Assignment-03/blob/main/docs/m1/PROMPT_LOG.md). These summarize actual requests; they are not a fabricated verbatim transcript. |
| Editing and verification | The assistant executed the earlier data checks and document checks, reviewed source records, rendered proposal pages and checked counts/hashes. Publication-specific tests and remote outcomes are recorded in [PUBLICATION_CHECKLIST.md](https://github.com/volethanhtrieu/DL_Assignment-03/blob/main/docs/m1/PUBLICATION_CHECKLIST.md). A prepared file, commit or passing local check is not evidence of successful deployment. |
| Member responsible for final verification | **Nguyễn Huỳnh Nam Quốc**, Assignment 3 lead. Completed independent student review/understanding is **not attested** by the assistant. No student signature or approval is generated. |
| Verification sources | Handbook Part I, Sections 7.6 and 24-32; pinned QCRI and mirror sources; immutable source/manifest hashes; existing validation/remediation reports; original papers and official encoder documentation linked in the proposal and [DATASET.md](https://github.com/volethanhtrieu/DL_Assignment-03/blob/main/docs/m1/DATASET.md). |

## What the evidence establishes

The earlier audit decoded 8,079 source pairs; remediation retained 7,938 (6,001/986/951). It quarantined 7 annotation discrepancies and 9 placeholder records, and filtered 125 lower-priority duplicate/leakage-related records. These are dataset measurements, not model scores. The original agreed-label archive was unavailable (HTTP 403); equivalence is unproved. Human labels were not silently changed.

The proposal correction changes only its tool-identification disclosure cell to describe ChatGPT Work/Codex. All other paragraph/cell text, embedded EDA bytes and reference links are preserved. The PDF is regenerated from that corrected DOCX.

## Limits and responsibilities

M2/M3 encoder training, fusion experiments, ablations, model metrics, checkpoints, slides and video are **planned, not completed**. The prior repository disclosure was a planning template; its prospective file/model names and blanket student-verification statement are not treated as evidence of executed work.

The group reports that prior dataset approval was waived for this group. No formal approval status is claimed. Students retain responsibility for understanding, correcting and verifying submitted material. LMS submission and completed student review are not asserted by this log. No additional AI providers or model versions are inferred from templates.

---

## M2 supplement — executed 17 November 2026 (development only)

This section adds *executed* M2 work to the historical M1 disclosure above. It does not retroactively change M1 claims.

| Mandatory field (Handbook Section 5.2) | M2 record |
|---|---|
| AI tool and model/version | ChatGPT Work/Codex assistance during implementation planning and report preparation. The exact underlying model/build was not independently verified. No additional AI provider is inferred. |
| Member using the tool | Nguyễn Huỳnh Nam Quốc (2453089), the Assignment 3 academic contributor (100%; the two other A3 members 0%). |
| Date and stage | 17 November 2026, A3 M2 dataset feature extraction, model training supervision, result checks and scientific report drafting/publication preparation. |
| Purpose | Assist with code generation/review, methodological consistency checks, result aggregation, plotting and LaTeX scientific writing. |
| Sections/files affected | `m2m3/` experimental source, M2 Colab notebook, `docs/m2/CO3133_Assignment 3_M2_Report.tex`/`.pdf`, summaries, publication records and A3 page text. |
| Representative prompts | Set up the locked dataset and GPU run, check nine M2 development runs, evaluate the results evidence, write a scientific M2 report in LaTeX, display colored graphs/matrix while preserving a conventional paper layout, and publish M2 without altering M1. Summaries, not fabricated verbatim transcripts. |
| Editing and verification performed | Student ran the real frozen encoders and 9 training runs on a Colab Tesla T4. Saved predictions, checkpoint/run JSON data, locked metadata and aggregate tables were inspected and recalculated. Reproduction from raw media is possible with published scripts but is not independently claimed for the current publication. |
| Final responsible student/verifier | Nguyễn Huỳnh Nam Quốc. Independent final student comprehension, LMS submission and instructor approval are student responsibilities; they are not certified by the assistant. |
| Verification sources | M1 `DATASET_LOCK.json`/dataset documentation; pinned QCRI and third-party mirror references; `artifacts/m2/runs/primary/` (nine run JSONs, predictions, trained-head checkpoints), `results/m2_dev/`, `artifacts/m2/feature_metadata.json`, and the CO3133 handbook. |

**Observed M2 outcome (development only):** image-only macro-F1 0.5746 ± 0.0068; text-only 0.6811 ± 0.0016; concatenation 0.8116 ± 0.0041, all means ± sample SD over seeds 42/43/44. No M3 improved model, ablation, or held-out test results are claimed. Original CrisisMMD agreed-label archive equivalence remains unverified. The report uses student-run measurements, not synthetic smoke metrics.

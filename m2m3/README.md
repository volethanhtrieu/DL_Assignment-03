# Assignment 3 — Milestone 2: empirical draft (Group Goat)

This directory contains the **actual M2 implementation and 9 development runs** performed on 17 November 2026. It supersedes the *pre-execution* README in the original code bundle. The main M1 dataset design, manifests and proposal remain unchanged. **M3 was completed subsequently.** This README records the M2-stage experiments only. See [M3 documentation](m3/README.md) and [M3 results](../results/m3/README.md).

## Locked protocol

- CrisisMMD v2.0 five-class *mirror-derived custom protocol* `crisismmd5-mirror-remediation-20261104-v1`, `filtered_original` 6,001 train / 986 dev / 951 held-out test pairs.
- Frozen ResNet-18 `IMAGENET1K_V1` (512 image features), DistilBERT base uncased (768 token-mask mean-pooled text features), trainable projection and classification heads.
- M2: image-only, text-only and concatenation fusion, each seed 42, 43, 44. Head-only Adam 1e-3, weight decay 1e-4, weighted cross entropy based on training labels, dropout 0.2, batch 64, max 30 epochs, dev macro-F1 early stopping patience 5.
- **Observed dev mean macro-F1 ± sample SD:** image 0.5746 ± 0.0068; text 0.6811 ± 0.0016; concat 0.8116 ± 0.0041.
- This M2 results section does not include held-out test scores or M3 gated/fixed-gate results; those are documented separately.

## Published M2 evidence

- LaTeX scientific report: `docs/m2/CO3133_Assignment 3_M2_Report.tex` and matching PDF.
- Verifiable run artefacts: `artifacts/m2/experiments/primary/{image,text,concat}/seed_{42,43,44}/{run.json,dev_predictions.csv,best.pt}` (27 files).
- Derived tables: `results/m2_dev/{individual_dev.csv,summary_dev.csv,DEV_RESULTS.md}`.
- Frozen encoder extraction metadata (not a feature cache): `artifacts/m2/feature_metadata.json`.
- Published SHA256 checksum ledger: `docs/m2/SHA256SUMS.txt`.
- Independent evidence verifier (Python standard library): `python scripts/verify_m2_evidence.py`.
- M2-only Google Colab notebook: `m2m3/CO3133_A3_M2_GPU_Execution.ipynb`. Original execution used a combined M2/M3 notebook but stopped after M2; the published notebook removes the M3 cells for safe reproduction.

Raw image bytes, tweet text, Parquet shards and full NPZ feature caches are *not* published. They must be obtained under the source license and SHA256-pinned M1 protocol for a fresh execution. The 9 `best.pt` files contain **trained head weights, not a full ResNet/DistilBERT encoder package**. To run raw-pair inference, regenerate the frozen features with the pinned encoders first; head-only inference timing is not end-to-end latency.

## Reproduce M2 from the dataset

In an environment with enough disk space (~3–5 GB), Python 3.10+ and optional NVIDIA GPU:

```bash
python -m pip install -r m2m3/requirements.txt
python -m pytest -q m2m3/tests
python m2m3/experiment.py check --repo .
python m2m3/download_mirror.py --repo . --out data/mirror
python m2m3/experiment.py extract --repo . --mirror data/mirror --out features/primary --batch-size 16 --device auto
python m2m3/run_all.py --repo . --features features/primary --runs runs/primary --milestone m2
python m2m3/summarize_dev.py --runs runs/primary --out results/m2_dev_new --milestone m2
```

`run_all.py` intentionally refuses to overwrite existing outputs. It saves only dev metrics and does not access test labels for selection. Source files are derived from the pinned QCRI/mirror revision; the archive-equivalence issue remains unresolved. Exact image-text alignment and leakage protection follow `A3-M1/DATASET_LOCK.json` and `docs/m1/DATASET.md`.

## Verification and limitations

Run `python scripts/verify_m2_evidence.py` to recompute dev metrics from all saved predictions and validate the checksum ledger. The original GPU execution was on a Tesla T4, torch 2.14.1+cu130, torchvision 0.29.1+cu130, transformers 4.57.6; frozen encoder extraction took 144.66 s. Full training/inference costs are described in the report; the per-run train timings are *head training + dev only*. Rare class `affected_individuals` has 9 development records; even a large F1 improvement has high variance. Unseen-disaster generalization is not established. M2 is a graded draft, not final M3.

Assignment 3 contribution: Nguyễn Huỳnh Nam Quốc 100%; Hoàng Trọng Huy Minh 0%; Võ Lê Thành Triệu 0%. The group remains responsible for verifying and submitting the correct files on LMS.

# Assignment 3 — M3 Experimental Results

This directory contains selected aggregate results from
the frozen five-class CrisisMMD experiment.

- development/: Development-set model summary.
- test/: Held-out test summary and paired bootstrap.
- ablation/: Fusion ablation results.
- efficiency/: End-to-end inference latency summary.

Protocol: filtered_original
Train: 6,001 pairs
Development: 986 pairs
Test: 951 pairs

These files were copied from the local experiment outputs
without changing their numerical contents.

Raw CrisisMMD media and individual case galleries are excluded. Trained-head checkpoints are available under `artifacts/m2/` and `artifacts/m3/`. This directory additionally includes 15 per-seed test metric files and `per_seed_metrics.csv`. Use `python scripts/verify_m3_evidence.py` to validate the 63 published checksum entries and aggregate results.

The dataset is a custom filtered protocol, not an
unchanged official benchmark.

Statistical significance should not be inferred from
small differences in mean macro-F1.

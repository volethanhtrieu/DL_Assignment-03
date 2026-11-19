#!/usr/bin/env python3
"""Verify published M3 evidence without raw media or GPU."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("image", "text", "concat", "fixed_gate", "gated")
SEEDS = (42, 43, 44)

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ledger = ROOT / "docs/m3/SHA256SUMS.txt"
    lines = [s for s in ledger.read_text(encoding="utf-8").splitlines() if s.strip()]
    require(len(lines) == 63, f"Expected 63 checksums, got {len(lines)}")
    seen = set()
    for line in lines:
        expected_hash, relative = line.split("  ", 1)
        require(relative not in seen, f"Duplicate ledger entry: {relative}")
        seen.add(relative)
        path = ROOT / relative
        require(path.is_file(), f"Missing: {relative}")
        require(digest(path) == expected_hash, f"SHA256 mismatch: {relative}")
    print("PASS: 63 evidence SHA256 checks")

    metric_path = ROOT / "results/m3/test/per_seed_metrics.csv"
    with metric_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected_pairs = {(m, s) for m in MODELS for s in SEEDS}
    pairs = [(r["model"], int(r["seed"])) for r in rows]
    require(len(pairs) == 15 and set(pairs) == expected_pairs,
            "Missing or duplicate model/seed metrics")

    for model, seed in sorted(expected_pairs):
        milestone = "m2" if model in ("image", "text", "concat") else "m3"
        folder = ROOT / f"artifacts/{milestone}/experiments/primary/{model}/seed_{seed}"
        run = read_json(folder / "run.json")
        require(run["model"] == model and int(run["seed"]) == seed,
                f"Run metadata mismatch: {model}/{seed}")
        require(run["smoke_only"] is False, f"Smoke run: {model}/{seed}")
        require((folder / "best.pt").is_file(), f"Checkpoint missing: {model}/{seed}")

        with (folder / "dev_predictions.csv").open(encoding="utf-8", newline="") as f:
            dev_count = sum(1 for _ in csv.DictReader(f))
        require(dev_count == 986, f"Dev prediction count: {model}/{seed}")

        test_metrics = read_json(
            ROOT / f"results/m3/test/{model}/seed_{seed}/test_metrics.json"
        )
        record = next(r for r in rows
                      if r["model"] == model and int(r["seed"]) == seed)
        for metric in ("macro_f1", "accuracy"):
            require(abs(float(record[metric]) - test_metrics[metric]) < 1e-9,
                    f"Test metric mismatch: {model}/{seed}/{metric}")

    summary = read_json(ROOT / "results/m3/test/summary.json")
    require(len(summary["model_metrics"]) == 5, "Expected five models")
    require(len({r["model"] for r in summary["model_metrics"]}) == 5,
            "Duplicate summary model")
    for item in summary["model_metrics"]:
        subset = [r for r in rows if r["model"] == item["model"]]
        require(len(subset) == 3, "Expected three seeds")
        for metric in ("macro_f1", "accuracy"):
            values = [float(r[metric]) for r in subset]
            require(abs(statistics.mean(values) - item[f"{metric}_mean"]) < 1e-9,
                    f"Summary mean mismatch: {item['model']}/{metric}")
            require(abs(statistics.stdev(values) - item[f"{metric}_sd"]) < 1e-9,
                    f"Summary SD mismatch: {item['model']}/{metric}")

    print("PASS: 15 run logs, checkpoints and dev prediction counts")
    print("PASS: 15 per-seed test metric comparisons")
    print("PASS: five model means and sample SDs")
    print("M3 PUBLISHED EVIDENCE VERIFICATION PASSED")

if __name__ == "__main__":
    main()

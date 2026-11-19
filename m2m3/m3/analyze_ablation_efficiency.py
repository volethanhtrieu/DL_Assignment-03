
import csv
import json
import statistics
from pathlib import Path
from _m3_paths import DATA_ROOT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = (DATA_ROOT / "Results")
final = base / "m3_test_final"
runs = base / "runs/primary"
out = base / "ablation"

MODELS = ("image", "text", "concat", "fixed_gate", "gated")
SEEDS = (42, 43, 44)

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

assert final.is_dir() and runs.is_dir()
out.mkdir(parents=True)

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def mean(values):
    return statistics.mean(values)

def sd(values):
    return statistics.stdev(values)

original = read_csv(final / "per_seed_metrics.csv")
assert len(original) == 15
assert {
    (r["model"], int(r["seed"])) for r in original
} == {(m, s) for m in MODELS for s in SEEDS}

rows = []
class_scores = {}

for item in original:
    model = item["model"]
    seed = int(item["seed"])

    run = json.loads(
        (runs / model / f"seed_{seed}" / "run.json").read_text()
    )
    test = json.loads(
        (final / model / f"seed_{seed}" /
         "test_metrics.json").read_text()
    )

    assert run["model"] == model
    assert run["seed"] == seed
    assert not run["smoke_only"]
    assert abs(
        test["macro_f1"] - float(item["macro_f1"])
    ) < 1e-9
    assert abs(
        test["accuracy"] - float(item["accuracy"])
    ) < 1e-9

    assert run["best_epoch"] == int(item["best_epoch"])
    assert run["trainable_parameters"] == int(
        item["trainable_parameters"]
    )

    rows.append({
        "model": model,
        "seed": seed,
        "dev_macro_f1": run["dev"]["macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_accuracy": test["accuracy"],
        "best_epoch": run["best_epoch"],
        "epochs_run": run["epochs_run"],
        "trainable_parameters": run["trainable_parameters"],
        "train_and_dev_seconds": run["train_and_dev_seconds"],
        "test_head_inference_seconds": float(
            item["test_head_inference_seconds"]
        ),
        "device": run["device"],
    })

    for cls, metrics in test["per_class"].items():
        class_scores[(model, seed, cls)] = metrics["f1"]

save_csv(out / "01_ablation_per_seed.csv", rows)

summary = []

for model in MODELS:
    subset = [r for r in rows if r["model"] == model]
    assert len(subset) == 3
    params = {r["trainable_parameters"] for r in subset}
    assert len(params) == 1

    summary.append({
        "model": model,
        "dev_f1_mean": mean([r["dev_macro_f1"] for r in subset]),
        "dev_f1_sd": sd([r["dev_macro_f1"] for r in subset]),
        "test_f1_mean": mean([r["test_macro_f1"] for r in subset]),
        "test_f1_sd": sd([r["test_macro_f1"] for r in subset]),
        "test_accuracy_mean": mean(
            [r["test_accuracy"] for r in subset]
        ),
        "trainable_parameters": params.pop(),
        "train_and_dev_seconds_mean": mean(
            [r["train_and_dev_seconds"] for r in subset]
        ),
        "test_head_inference_seconds_mean": mean(
            [r["test_head_inference_seconds"] for r in subset]
        ),
    })

save_csv(out / "02_ablation_summary.csv", summary)

by_model = {r["model"]: r for r in summary}
by_run = {(r["model"], r["seed"]): r for r in rows}

# Paired seed comparisons: descriptive, not significance tests.
comparisons = []

for baseline in ("fixed_gate", "concat"):
    for seed in SEEDS:
        a = by_run[("gated", seed)]
        b = by_run[(baseline, seed)]

        comparisons.append({
            "comparison": f"gated_minus_{baseline}",
            "seed": seed,
            "delta_dev_macro_f1": (
                a["dev_macro_f1"] - b["dev_macro_f1"]
            ),
            "delta_test_macro_f1": (
                a["test_macro_f1"] - b["test_macro_f1"]
            ),
        })

save_csv(out / "03_paired_seed_differences.csv", comparisons)

# Figure 1: Main fusion ablation
fusion = ["concat", "fixed_gate", "gated"]
labels = ["Concat", "Fixed Gate", "Learned Gate"]

means = [by_model[m]["test_f1_mean"] for m in fusion]
stds = [by_model[m]["test_f1_sd"] for m in fusion]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(3)

ax.bar(x, means, yerr=stds, capsize=5)
ax.set_xticks(x, labels)
ax.set_ylabel("Test macro-F1")
ax.set_title("Fusion Ablation — 3 Seeds")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.2)

for i, value in enumerate(means):
    ax.text(
        i, value + stds[i] + 0.02,
        f"{value:.4f}", ha="center"
    )

fig.tight_layout()
fig.savefig(out / "04_fusion_ablation.png", dpi=200)
plt.close(fig)

# Figure 2: Training and dev time; head training only
fig, ax = plt.subplots(figsize=(9, 5))

models = list(MODELS)
times = [
    by_model[m]["train_and_dev_seconds_mean"]
    for m in models
]

ax.bar(models, times)
ax.set_ylabel("Mean elapsed seconds")
ax.set_title("Head Training + Development Evaluation")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(out / "05_training_cost.png", dpi=200)
plt.close(fig)

# Figure 3: Parameters
fig, ax = plt.subplots(figsize=(9, 5))

params = [
    by_model[m]["trainable_parameters"]
    for m in models
]

ax.bar(models, params)
ax.set_ylabel("Trainable head parameters")
ax.set_title("Trainable Parameter Comparison")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(out / "06_parameter_comparison.png", dpi=200)
plt.close(fig)

# Figure 4: Class-wise F1 across fusion architectures
classes = list(
    json.loads(
        (final / "gated/seed_42/test_metrics.json").read_text()
    )["per_class"]
)

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(classes))
width = 0.25

for index, model in enumerate(fusion):
    scores = [
        mean([
            class_scores[(model, seed, cls)]
            for seed in SEEDS
        ])
        for cls in classes
    ]

    ax.bar(
        x + (index - 1) * width,
        scores,
        width,
        label=labels[index]
    )

ax.set_xticks(
    x,
    ["Affected", "Infrastructure", "Rescue",
     "Other relevant", "Not humanitarian"]
)
ax.set_ylabel("Mean class F1 across seeds")
ax.set_ylim(0, 1)
ax.set_title("Fusion Ablation — Class-wise F1")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(out / "07_classwise_ablation.png", dpi=200)
plt.close(fig)

print("\nABLATION & EFFICIENCY ANALYSIS PASS")

for r in summary:
    print(
        f"{r['model']:12s} "
        f"F1={r['test_f1_mean']:.4f} +/- "
        f"{r['test_f1_sd']:.4f} | "
        f"params={r['trainable_parameters']:,} | "
        f"train+dev={r['train_and_dev_seconds_mean']:.2f}s"
    )

for baseline in ("fixed_gate", "concat"):
    delta = (
        by_model["gated"]["test_f1_mean"]
        - by_model[baseline]["test_f1_mean"]
    )
    print(f"Gated minus {baseline}: {delta:+.4f}")

print("\nSaved files:")
for path in sorted(out.iterdir()):
    print(" -", path.name)

print("\nNOTE: Historical timing covers head-only operations.")
print("Full image/text encoder latency requires separate benchmarking.")
print("No test-time model selection or retraining performed.")

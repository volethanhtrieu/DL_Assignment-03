
import csv
import json
import statistics
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo = Path("DL_Assignment-03").resolve()
sys.path.insert(0, str(repo / "m2m3"))
import experiment as exp

base = (DATA_ROOT / "Results")
runs = base / "runs/primary/gated"
final = base / "m3_test_final/gated/seed_42"
gallery = base / "qualitative_analysis/test_cases"
out = base / "interpretability/gate_weights"

SEEDS = (42, 43, 44)
CLASSES = list(exp.CLASSES)

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

lock, manifests = exp.validate_locked_repository(repo)

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows):
    if not rows:
        raise ValueError("No rows to save")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

dev_reference = {
    r["row_id"]: r for r in manifests["dev"]
}

all_rows = []
class_seed_rows = []

for seed in SEEDS:
    directory = runs / f"seed_{seed}"
    log = json.loads(
        (directory / "run.json").read_text(encoding="utf-8")
    )

    assert log["model"] == "gated"
    assert log["seed"] == seed
    assert not log["smoke_only"]

    predictions = read_csv(
        directory / "dev_predictions.csv"
    )

    assert len(predictions) == 986
    assert len({r["row_id"] for r in predictions}) == 986

    for row in predictions:
        reference = dev_reference[row["row_id"]]

        assert row["group_id"] == reference["group_id"]
        assert int(row["y_true"]) == int(reference["label_id"])

        g = float(row["gate_image_weight"])
        assert np.isfinite(g) and 0 <= g <= 1

        all_rows.append({
            "seed": seed,
            "row_id": row["row_id"],
            "true_class": int(row["y_true"]),
            "pred_class": int(row["y_pred"]),
            "correct": int(row["y_true"] == row["y_pred"]),
            "gate_image": g,
            "gate_text": 1 - g,
        })

    # Each class is summarized separately for each seed.
    current = [r for r in all_rows if r["seed"] == seed]

    for cls in range(5):
        subset = [
            r for r in current if r["true_class"] == cls
        ]

        values = np.array(
            [r["gate_image"] for r in subset]
        )

        class_seed_rows.append({
            "seed": seed,
            "class": CLASSES[cls],
            "support": len(subset),
            "gate_mean": float(values.mean()),
            "gate_median": float(np.median(values)),
            "gate_q25": float(np.percentile(values, 25)),
            "gate_q75": float(np.percentile(values, 75)),
            "fraction_gate_above_0_5": float(
                np.mean(values > 0.5)
            ),
        })

assert len(all_rows) == 986 * 3

out.mkdir(parents=True)

save_csv(out / "01_gate_per_sample_dev.csv", all_rows)
save_csv(out / "02_gate_by_class_seed.csv", class_seed_rows)

summary = []
for cls in CLASSES:
    subset = [
        r for r in class_seed_rows if r["class"] == cls
    ]
    means = [r["gate_mean"] for r in subset]

    summary.append({
        "class": cls,
        "support_per_seed": subset[0]["support"],
        "gate_mean_across_seeds": statistics.mean(means),
        "gate_sd_across_seeds": statistics.stdev(means),
    })

save_csv(out / "03_gate_class_summary.csv", summary)

# Figure 1: Class-wise learned image gate.
fig, ax = plt.subplots(figsize=(11, 5))

x = np.arange(5)
means = [r["gate_mean_across_seeds"] for r in summary]
errors = [r["gate_sd_across_seeds"] for r in summary]

ax.bar(x, means, yerr=errors, capsize=5)
ax.axhline(
    0.5, linestyle="--", color="gray",
    label="Equal scalar weighting (g=0.5)"
)

ax.set_xticks(
    x,
    ["Affected", "Infrastructure", "Rescue",
     "Other relevant", "Not humanitarian"]
)
ax.set_ylim(0, 1)
ax.set_ylabel("Mean learned image gate")
ax.set_title("Gated Fusion — Gate Values by True Class")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(out / "04_gate_by_class.png", dpi=220)
plt.close(fig)

# Figure 2: Gate distribution for correct vs incorrect.
correct = [
    r["gate_image"] for r in all_rows if r["correct"] == 1
]
incorrect = [
    r["gate_image"] for r in all_rows if r["correct"] == 0
]

fig, ax = plt.subplots(figsize=(7, 5))

ax.boxplot(
    [correct, incorrect],
    tick_labels=["Correct", "Incorrect"],
    showfliers=False
)
ax.axhline(0.5, linestyle="--", color="gray")
ax.set_ylabel("Learned image gate")
ax.set_ylim(0, 1)
ax.set_title("Gate Distribution by Prediction Correctness")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(
    out / "05_gate_correct_vs_incorrect.png",
    dpi=220
)
plt.close(fig)

# Link the 15 previously reviewed test examples.
# Descriptive post-hoc analysis only.
selected = read_csv(gallery / "selected_cases.csv")
test_predictions = {
    r["row_id"]: r
    for r in read_csv(final / "test_predictions.csv")
}

assert len(selected) == 15
assert len(test_predictions) == 951

illustrative = []
for case in selected:
    pred = test_predictions[case["row_id"]]

    assert int(case["seed"]) == 42
    assert int(pred["y_true"]) == int(case["true_label"])
    assert int(pred["y_pred"]) == int(case["gated_pred"])

    g = float(pred["gate_image_weight"])
    assert 0 <= g <= 1

    illustrative.append({
        "row_id": case["row_id"],
        "category": case["category"],
        "true_class": CLASSES[int(case["true_label"])],
        "gated_prediction": CLASSES[
            int(case["gated_pred"])
        ],
        "gate_image": g,
        "gate_text": 1 - g,
        "correct": int(
            case["true_label"] == case["gated_pred"]
        ),
        "review_note": "Manual semantic interpretation required",
    })

save_csv(
    out / "06_gallery_gate_values_posthoc.csv",
    illustrative
)

print("GATE EXPLAINABILITY ANALYSIS PASS")
print("Development predictions:", len(all_rows))
print("Illustrative test cases:", len(illustrative))

print("\nCLASS-WISE IMAGE GATE")
for r in summary:
    print(
        f"{r['class']:38s} "
        f"g={r['gate_mean_across_seeds']:.4f} "
        f"+/- {r['gate_sd_across_seeds']:.4f}"
    )

print("\nCORRECT VS INCORRECT")
print(f"Correct median g:   {np.median(correct):.4f}")
print(f"Incorrect median g: {np.median(incorrect):.4f}")

print("\nSaved files:")
for path in sorted(out.iterdir()):
    print(" -", path.name)

print("\nNo retraining or test-time tuning performed.")
print("Gate values are descriptive, not causal attribution.")

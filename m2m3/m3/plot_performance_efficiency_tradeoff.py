
import csv
from pathlib import Path
from _m3_paths import DATA_ROOT

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = (DATA_ROOT / "Results")
out = root / "efficiency/tradeoff"

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

quality = {
    r["model"]: r
    for r in read_csv(
        root / "ablation/02_ablation_summary.csv"
    )
}
latency = {
    r["model"]: r
    for r in read_csv(
        root / "efficiency/inference_benchmark/latency_summary.csv"
    )
}

models = ["image", "text", "concat", "fixed_gate", "gated"]
assert set(models) == set(quality) == set(latency)
assert all(
    int(latency[m]["seed"]) == 42
    and int(latency[m]["samples"]) == 48
    and int(latency[m]["repeats"]) == 2
    and float(latency[m]["dev_prediction_agreement"]) == 1.0
    for m in models
)

out.mkdir(parents=True)

rows = []
for model in models:
    q, l = quality[model], latency[model]
    rows.append({
        "model": model,
        "test_f1_mean_3seeds": q["test_f1_mean"],
        "test_f1_sd_3seeds": q["test_f1_sd"],
        "parameters": q["trainable_parameters"],
        "p50_ms_seed42": l["median_ms"],
        "p90_ms_seed42": l["p90_ms"],
        "p95_ms_seed42": l["p95_ms"],
    })

with (out / "tradeoff_summary.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Figure 1: Predictive performance versus latency.
fig, ax = plt.subplots(figsize=(9, 6))

for row in rows:
    name = row["model"]
    x = float(row["p50_ms_seed42"])
    y = float(row["test_f1_mean_3seeds"])
    err = float(row["test_f1_sd_3seeds"])

    ax.errorbar(
        x, y, yerr=err,
        fmt="o", markersize=8,
        capsize=4
    )
    ax.annotate(
        name.replace("_", " ").title(),
        (x, y),
        xytext=(7, 6),
        textcoords="offset points"
    )

ax.set_xlabel("Median end-to-end latency (ms), seed 42")
ax.set_ylabel("Test macro-F1 (mean ± SD, 3 seeds)")
ax.set_title("Accuracy–Latency Trade-off")
ax.set_ylim(0.55, 0.87)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(out / "01_accuracy_latency_tradeoff.png", dpi=220)
plt.close(fig)

# Figure 2: P50 and P90 latency.
x = np.arange(len(models))
width = 0.35

p50 = [float(latency[m]["median_ms"]) for m in models]
p90 = [float(latency[m]["p90_ms"]) for m in models]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, p50, width, label="P50")
ax.bar(x + width/2, p90, width, label="P90")

ax.set_xticks(x, [m.replace("_", " ").title() for m in models])
ax.set_ylabel("End-to-end latency (ms)")
ax.set_title("Batch-1 Inference Latency — RTX 4060")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(out / "02_latency_comparison.png", dpi=220)
plt.close(fig)

print("EFFICIENCY–PERFORMANCE TRADE-OFF PASS")
for path in sorted(out.iterdir()):
    print(" -", path.name)
print("Historical test metrics; dev-only latency; no retraining.")

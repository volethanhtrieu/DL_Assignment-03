
import csv
import hashlib
import io
import json
import statistics
import sys
import time
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import models
from transformers import AutoModel, AutoTokenizer

repo = REPO_ROOT
sys.path.insert(0, str(repo / "m2m3"))
import experiment as exp

base = DATA_ROOT
mirror = base / "Raw/mirror/humanitarian"
features = base / "Features/features/primary"
runs = base / "Results/runs/primary"
out = base / "Results/efficiency/inference_benchmark"

MODELS = ("image", "text", "concat", "fixed_gate", "gated")
SEED = 42
N = 48
REPEATS = 2
WARMUP = 5

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

assert torch.cuda.is_available(), "CUDA GPU required"
device = torch.device("cuda")

lock, manifests = exp.validate_locked_repository(repo)
dev = exp.load_features(
    features / "dev.npz", expected_split="dev"
)
reference = {r["row_id"]: r for r in manifests["dev"]}

# Select reproducibly, avoiding duplicate connected groups.
rng = np.random.default_rng(20261117)
selected = {}
for i in rng.permutation(len(dev["y"])):
    group = str(dev["group_ids"][i])
    if group not in selected:
        selected[group] = str(dev["row_ids"][i])
    if len(selected) == N:
        break

wanted = set(selected.values())
assert len(wanted) == N

# Verify pinned validation source.
source_lock = json.loads(
    (repo / "A3-M1/provenance/mirror_downloads.json").read_text()
)
source = next(
    x for x in source_lock["files"]
    if x["path"].endswith(
        "validation-00000-of-00001.parquet"
    )
)

source_path = mirror / Path(source["path"]).name
assert source_path.stat().st_size == source["bytes"]
assert exp.sha256(source_path) == source["sha256"]

records = {}

for batch in pq.ParquetFile(source_path).iter_batches(
    batch_size=64
):
    for item in batch.to_pylist():
        row_id = "dev:" + str(item["image_id"])
        if row_id not in wanted:
            continue

        m = reference[row_id]
        blob = item["image"]["bytes"]
        raw_text = item["text"]

        assert hashlib.sha256(blob).hexdigest() == (
            m["image_sha256"]
        )
        assert hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest() == m["raw_text_sha256"]
        assert str(item["tweet_id"]) == m["tweet_id"]
        assert int(item["label"]) == int(m["label_id"])

        records[row_id] = {
            "row_id": row_id,
            "image_bytes": blob,
            "text": raw_text,
        }

assert len(records) == N
samples = [records[r] for r in selected.values()]
print("Verified dev samples:", N, flush=True)

# Original M1/M2 encoders and preprocessing.
weights = models.ResNet18_Weights.IMAGENET1K_V1
image_encoder = models.resnet18(weights=weights)
image_encoder.fc = torch.nn.Identity()
image_encoder.eval().to(device)
transform = weights.transforms()

tokenizer = AutoTokenizer.from_pretrained(
    exp.TEXT_MODEL,
    revision=exp.TEXT_REVISION,
)
text_encoder = AutoModel.from_pretrained(
    exp.TEXT_MODEL,
    revision=exp.TEXT_REVISION,
).eval().to(device)

heads = {}
saved_preds = {}

for name in MODELS:
    path = runs / name / f"seed_{SEED}"
    ck = torch.load(
        path / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    run = json.loads((path / "run.json").read_text())

    assert ck["model"] == name
    assert ck["seed"] == SEED
    assert ck["best_epoch"] == run["best_epoch"]

    head = exp.Head(name).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    heads[name] = head

    with (path / "dev_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        saved_preds[name] = {
            r["row_id"]: int(r["y_pred"])
            for r in csv.DictReader(f)
        }

def predict_one(sample, name):
    img_vec = None
    txt_vec = None

    if name != "text":
        with Image.open(
            io.BytesIO(sample["image_bytes"])
        ) as im:
            tensor = transform(
                im.convert("RGB")
            ).unsqueeze(0).to(device)

    if name != "image":
        normalized = exp.normalise_text(sample["text"])
        tokens = tokenizer(
            normalized,
            max_length=exp.MAX_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = {
            k: v.to(device) for k, v in tokens.items()
        }

    with torch.inference_mode():
        if name != "text":
            img_vec = image_encoder(tensor)

        if name != "image":
            hidden = text_encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            ).last_hidden_state

            mask = tokens["attention_mask"].unsqueeze(-1)
            mask = mask.to(hidden.dtype)

            txt_vec = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            )

        logits = heads[name](img_vec, txt_vec)
        return int(logits.argmax(dim=-1).item())

def sync():
    torch.cuda.synchronize()

results = []
summaries = []

for name in MODELS:
    print("Benchmarking:", name, flush=True)

    # GPU warm-up is excluded from measurement.
    for i in range(WARMUP):
        predict_one(samples[i], name)
    sync()

    durations = []
    matches = 0
    checked = 0

    for repeat in range(REPEATS):
        for sample in samples:
            sync()
            started = time.perf_counter()

            prediction = predict_one(sample, name)

            sync()
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            durations.append(elapsed_ms)

            # Agreement with original dev predictions.
            if repeat == 0:
                checked += 1
                matches += int(
                    prediction
                    == saved_preds[name][sample["row_id"]]
                )

            results.append({
                "model": name,
                "seed": SEED,
                "row_id": sample["row_id"],
                "repeat": repeat + 1,
                "latency_ms": elapsed_ms,
                "predicted_label": prediction,
            })

    summary = {
        "model": name,
        "seed": SEED,
        "samples": N,
        "repeats": REPEATS,
        "measurements": len(durations),
        "mean_ms": statistics.mean(durations),
        "median_ms": float(np.median(durations)),
        "p90_ms": float(np.percentile(durations, 90)),
        "p95_ms": float(np.percentile(durations, 95)),
        "dev_prediction_agreement": matches / checked,
    }
    summaries.append(summary)

    print(
        f"  p50={summary['median_ms']:.2f} ms | "
        f"p90={summary['p90_ms']:.2f} ms | "
        f"agreement={matches}/{checked}",
        flush=True,
    )

out.mkdir(parents=True)

def write_csv(path, rows):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

write_csv(out / "latency_per_request.csv", results)
write_csv(out / "latency_summary.csv", summaries)

metadata = {
    "dataset": lock["dataset_version"],
    "split": "dev",
    "samples": N,
    "repeats": REPEATS,
    "warmup_per_model": WARMUP,
    "batch_size": 1,
    "seed": SEED,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "encoder": "ResNet18 + DistilBERT (M1 pinned)",
    "included": (
        "RGB decoding, image transform, text normalization, "
        "tokenization, applicable encoder forward passes, "
        "classification head, output transfer, CUDA sync"
    ),
    "excluded": (
        "Model loading, Parquet disk I/O, training, "
        "network transfers"
    ),
    "test_used": False,
}
(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nEND-TO-END BENCHMARK PASS")
for r in summaries:
    print(
        f"{r['model']:12s} "
        f"p50={r['median_ms']:.2f} ms | "
        f"p90={r['p90_ms']:.2f} ms | "
        f"p95={r['p95_ms']:.2f} ms"
    )
print("Output:", out)
print("TEST SET NOT USED")

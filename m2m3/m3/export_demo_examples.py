
import csv
import hashlib
import io
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

import pyarrow.parquet as pq
from PIL import Image

base = DATA_ROOT
repo = REPO_ROOT
sys.path.insert(0, str(repo / "m2m3"))
import experiment as exp

manifest_dir = repo / "A3-M1/remediation/manifests/filtered_original"
pred_path = base / "Results/runs/primary/gated/seed_42/dev_predictions.csv"
parquet_path = base / "Raw/mirror/humanitarian/validation-00000-of-00001.parquet"
out = base / "Results/demo/examples"

assert not out.exists(), "Output exists: refusing overwrite"

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

manifest = {
    r["row_id"]: r
    for r in read_csv(manifest_dir / "dev.csv")
}
predictions = read_csv(pred_path)

# One correctly predicted DEV example per class.
classes = (1, 2, 3)
chosen = {}

for cls in classes:
    candidates = sorted(
        (
            r for r in predictions
            if int(r["y_true"]) == cls
            and int(r["y_pred"]) == cls
        ),
        key=lambda r: r["row_id"],
    )
    assert candidates, f"No candidate for class {cls}"
    chosen[candidates[0]["row_id"]] = cls

# Verify source provenance.
shards = exp.validate_source_shards(
    repo, base / "Raw/mirror"
)
assert parquet_path in shards.values()

out.mkdir(parents=True)
found = set()

for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
    for item in batch.to_pylist():
        row_id = "dev:" + str(item["image_id"])
        if row_id not in chosen:
            continue

        reference = manifest[row_id]
        blob = item["image"]["bytes"]
        tweet = item["text"]

        assert hashlib.sha256(blob).hexdigest() == reference["image_sha256"]
        assert hashlib.sha256(tweet.encode()).hexdigest() == reference["raw_text_sha256"]
        assert int(item["label"]) == int(reference["label_id"])

        cls = chosen[row_id]
        stem = f"class_{cls}"

        with Image.open(io.BytesIO(blob)) as im:
            im.convert("RGB").save(out / f"{stem}.png")

        (out / f"{stem}_tweet.txt").write_text(
            tweet, encoding="utf-8"
        )
        found.add(row_id)

assert found == set(chosen)

print("DEMO EXAMPLES PASS")
for row_id, cls in chosen.items():
    print(f"{exp.CLASSES[cls]} | {row_id}")
print("Output:", out)
print("Dev-only illustrative examples; no test selection.")

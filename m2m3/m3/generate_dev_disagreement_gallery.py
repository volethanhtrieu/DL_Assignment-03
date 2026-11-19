
import csv
import hashlib
import html
import io
import json
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT

import pyarrow.parquet as pq
from PIL import Image

base = DATA_ROOT
repo = Path("DL_Assignment-03")
sys.path.insert(0, str(repo / "m2m3"))
import experiment as exp

runs = base / "Results/runs/primary"
out = base / "Results/qualitative_analysis/dev_disagreements"

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Output already exists: {out}")

def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

manifest = {
    r["row_id"]: r
    for r in read_csv(
        repo / "A3-M1/remediation/manifests/filtered_original/dev.csv"
    )
}

models = ("image", "text", "concat", "gated")
preds = {
    model: {
        r["row_id"]: r
        for r in read_csv(
            runs / model / "seed_42/dev_predictions.csv"
        )
    }
    for model in models
}

assert len(manifest) == 986
assert all(set(p) == set(manifest) for p in preds.values())

candidates = []

for row_id in sorted(manifest):
    gt = int(manifest[row_id]["label_id"])
    g = int(preds["gated"][row_id]["y_pred"])
    c = int(preds["concat"][row_id]["y_pred"])

    if g == gt and c != gt:
        category = "Gated wins"
    elif g != gt and c == gt:
        category = "Concat wins"
    else:
        continue

    for model in models:
        p = preds[model][row_id]
        assert int(p["y_true"]) == gt
        assert p["group_id"] == manifest[row_id]["group_id"]

    candidates.append({
        "row_id": row_id,
        "category": category,
        "event": manifest[row_id]["event_name"],
        "ground_truth": exp.CLASSES[gt],
        "image_pred": exp.CLASSES[int(preds["image"][row_id]["y_pred"])],
        "text_pred": exp.CLASSES[int(preds["text"][row_id]["y_pred"])],
        "concat_pred": exp.CLASSES[c],
        "gated_pred": exp.CLASSES[g],
        "gate_image": float(
            preds["gated"][row_id]["gate_image_weight"]
        ),
    })

wins = sum(r["category"] == "Gated wins" for r in candidates)
losses = sum(r["category"] == "Concat wins" for r in candidates)

assert (wins, losses) == (16, 25), (wins, losses)

# Verify the original source and extract actual image–text pairs.
shards = exp.validate_source_shards(
    repo, base / "Raw/mirror"
)
validation_shards = [
    p for name, p in shards.items()
    if name.startswith("validation-")
]

wanted = {r["row_id"]: r for r in candidates}

out.mkdir(parents=True)
found = set()

for shard in validation_shards:
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
        for item in batch.to_pylist():
            row_id = "dev:" + str(item["image_id"])
            if row_id not in wanted:
                continue

            m = manifest[row_id]
            blob = item["image"]["bytes"]
            tweet = item["text"]

            assert hashlib.sha256(blob).hexdigest() == m["image_sha256"]
            assert hashlib.sha256(
                tweet.encode("utf-8")
            ).hexdigest() == m["raw_text_sha256"]
            assert int(item["label"]) == int(m["label_id"])
            assert str(item["tweet_id"]) == m["tweet_id"]
            assert item["event_name"] == m["event_name"]

            case = wanted[row_id]
            number = candidates.index(case) + 1
            stem = f"case_{number:02d}"

            with Image.open(io.BytesIO(blob)) as img:
                img.convert("RGB").save(out / f"{stem}.png")

            (out / f"{stem}_tweet.txt").write_text(
                tweet, encoding="utf-8"
            )

            case["image_file"] = f"{stem}.png"
            case["tweet_file"] = f"{stem}_tweet.txt"
            case["tweet"] = tweet
            found.add(row_id)

assert found == set(wanted)

with (out / "review_candidates.csv").open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(candidates[0])
    )
    writer.writeheader()
    writer.writerows(candidates)

cards = []

for r in candidates:
    def safe(value):
        return html.escape(str(value))

    details = "".join(
        f"<tr><th>{safe(label)}</th><td>{safe(value)}</td></tr>"
        for label, value in [
            ("Event", r["event"]),
            ("Ground truth", r["ground_truth"]),
            ("Image", r["image_pred"]),
            ("Text", r["text_pred"]),
            ("Concat", r["concat_pred"]),
            ("Gated", r["gated_pred"]),
            ("Image gate", f"{r['gate_image']:.3f}"),
        ]
    )

    cards.append(f"""
    <article>
      <h3>{safe(r["category"])}</h3>
      <small>{safe(r["row_id"])}</small>
      <img src="{safe(r["image_file"])}" alt="Dataset image">
      <p>{safe(r["tweet"])}</p>
      <table>{details}</table>
      <p><b>Manual review:</b> Not reviewed</p>
    </article>
    """)

page = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Group Goat — Dev Disagreement Review</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #f4f6f9;
  color: #202633;
  padding: 24px;
}
main { max-width: 1450px; margin: auto; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px,1fr));
  gap: 18px;
}
article {
  background: white;
  padding: 18px;
  border: 1px solid #ddd;
  border-radius: 10px;
}
img {
  width: 100%;
  height: 260px;
  object-fit: contain;
}
p { overflow-wrap: anywhere; line-height: 1.5; }
table { width: 100%; border-collapse: collapse; }
td, th {
  border-bottom: 1px solid #eee;
  padding: 7px;
  text-align: left;
  font-size: 13px;
}
small { color: #667085; }
</style>
</head>
<body>
<main>
<h1>Gated Fusion vs Concatenation</h1>
<p>Development set | Seed 42 | 16 Gated wins | 25 Concat wins</p>
<p>Local qualitative review only. Prediction differences
do not prove semantic image–text conflict.</p>
<div class="grid">
""" + "\n".join(cards) + """
</div></main></body></html>
"""

(out / "case_gallery.html").write_text(
    page, encoding="utf-8"
)

print("DEV DISAGREEMENT GALLERY PASS")
print("Gated wins:", wins)
print("Concat wins:", losses)
print("Verified image–text pairs:", len(found))
print("Output:", out)
print("No retraining or test evaluation.")

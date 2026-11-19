
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

import pyarrow.parquet as pq
from PIL import Image

base = DATA_ROOT
repo = REPO_ROOT

cases_path = base / "Results/m3_test_final/modality_cases.csv"
manifest_path = (
    repo / "A3-M1/remediation/manifests/filtered_original/test.csv"
)
parquet_path = (
    base / "Raw/mirror/humanitarian/test-00000-of-00001.parquet"
)
out = base / "Results/qualitative_analysis/test_cases"

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

manifest = {r["row_id"]: r for r in read_csv(manifest_path)}
cases = [r for r in read_csv(cases_path) if int(r["seed"]) == 42]

assert len(manifest) == 951
assert len(cases) == 951
assert len({r["row_id"] for r in cases}) == 951

# Verify predictions against the locked test manifest.
for c in cases:
    m = manifest[c["row_id"]]
    assert c["group_id"] == m["group_id"]
    assert int(c["true_label"]) == int(m["label_id"])

def category(c):
    y = int(c["true_label"])
    i = int(c["image_pred"])
    t = int(c["text_pred"])
    g = int(c["gated_pred"])

    if i == t == g == y:
        return "Agreement: all correct"
    if i != t and g == y and (i == y or t == y):
        return "Recovery: one modality correct"
    if i != y and t != y and g == y:
        return "Recovery: both unimodal wrong"
    if g != y and (i == y or t == y):
        return "Fusion harm"
    if i != t and g != y:
        return "Disagreement: unresolved"
    return "Other errors or agreements"

categories = [
    "Agreement: all correct",
    "Recovery: one modality correct",
    "Recovery: both unimodal wrong",
    "Fusion harm",
    "Disagreement: unresolved",
]

pools = {name: [] for name in categories}
for c in cases:
    name = category(c)
    if name in pools:
        pools[name].append(c)

# Deterministic, label/event-diverse illustrative selection.
selected = []
for name in categories:
    remaining = pools[name].copy()
    used_labels = set()
    used_events = set()
    used_groups = set()

    for _ in range(min(3, len(remaining))):
        def priority(c):
            m = manifest[c["row_id"]]
            digest = hashlib.sha256(
                c["row_id"].encode()
            ).hexdigest()
            return (
                int(c["true_label"] in used_labels),
                int(m["event_name"] in used_events),
                int(c["group_id"] in used_groups),
                digest,
            )

        chosen = min(remaining, key=priority)
        remaining.remove(chosen)
        chosen = dict(chosen)
        chosen["category"] = name
        selected.append(chosen)

        used_labels.add(chosen["true_label"])
        used_events.add(manifest[chosen["row_id"]]["event_name"])
        used_groups.add(chosen["group_id"])

wanted = {r["row_id"] for r in selected}
records = {}

for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64):
    for record in batch.to_pylist():
        row_id = "test:" + str(record["image_id"])
        if row_id not in wanted:
            continue

        m = manifest[row_id]
        blob = record["image"]["bytes"]
        text = record["text"]

        assert hashlib.sha256(blob).hexdigest() == m["image_sha256"]
        assert hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest() == m["raw_text_sha256"]
        assert str(record["tweet_id"]) == m["tweet_id"]
        assert record["event_name"] == m["event_name"]
        assert int(record["label"]) == int(m["label_id"])

        with Image.open(io.BytesIO(blob)) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((440, 330))
            buffer = io.BytesIO()
            thumb.save(buffer, format="JPEG", quality=85)

        records[row_id] = {
            "text": text,
            "image": base64.b64encode(
                buffer.getvalue()
            ).decode("ascii"),
            "event": m["event_name"],
            "label": m["label"],
        }

assert len(records) == len(selected)

out.mkdir(parents=True)

fields = [
    "category", "row_id", "group_id", "seed", "event_name",
    "true_label", "true_class", "image_pred", "text_pred",
    "concat_pred", "gated_pred", "tweet_text"
]

with (out / "selected_cases.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for c in selected:
        record = records[c["row_id"]]
        writer.writerow({
            **{k: c.get(k, "") for k in fields},
            "event_name": record["event"],
            "true_class": record["label"],
            "tweet_text": record["text"],
        })

counts = {name: len(pools[name]) for name in categories}
(out / "category_counts.json").write_text(
    json.dumps({
        "seed": 42,
        "scope": "Post-hoc qualitative review of finalized test",
        "counts": counts,
        "selected": len(selected),
        "sampling": "Deterministic illustrative selection",
    }, indent=2),
    encoding="utf-8",
)

cards = []

for c in selected:
    r = records[c["row_id"]]
    esc = lambda v: html.escape(str(v), quote=True)

    cards.append(f"""
    <article>
      <h3>{esc(c["category"])}</h3>
      <img src="data:image/jpeg;base64,{r["image"]}"
           alt="CrisisMMD sample">
      <p class="tweet">{esc(r["text"])}</p>
      <table>
        <tr><td>Event</td><td>{esc(r["event"])}</td></tr>
        <tr><td>Ground truth</td><td>{esc(r["label"])}</td></tr>
        <tr><td>Image prediction</td>
            <td>{esc(c["image_pred"])}</td></tr>
        <tr><td>Text prediction</td>
            <td>{esc(c["text_pred"])}</td></tr>
        <tr><td>Concat prediction</td>
            <td>{esc(c["concat_pred"])}</td></tr>
        <tr><td>Gated prediction</td>
            <td>{esc(c["gated_pred"])}</td></tr>
      </table>
      <small>{esc(c["row_id"])} | Seed 42</small>
      <p class="note">Human review: ____________________</p>
    </article>
    """)

page = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CrisisMMD — Qualitative Case Review</title>
<style>
body {
  font-family: Arial, sans-serif;
  max-width: 1400px;
  margin: 35px auto;
  padding: 0 20px;
  color: #202633;
  background: #f4f6f9;
}
h1 { margin-bottom: 6px; }
.subtitle { color: #667085; margin-bottom: 30px; }
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 22px;
}
article {
  background: white;
  padding: 22px;
  border: 1px solid #dce2eb;
  border-radius: 12px;
}
article img {
  display: block;
  max-width: 100%;
  max-height: 330px;
  object-fit: contain;
  margin: 15px auto;
}
.tweet {
  line-height: 1.55;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
table { width: 100%; border-collapse: collapse; }
td { padding: 7px; border-bottom: 1px solid #edf0f4; }
td:first-child { font-weight: bold; width: 42%; }
small { color: #667085; }
.note { margin-top: 20px; color: #667085; }
</style>
</head>
<body>
<h1>Humanitarian Multimodal Classification</h1>
<p class="subtitle">
Qualitative Case Review — Group Goat — Seed 42
<br>Local research preview. Not a public deployment.
<br>Prediction disagreement is not automatically
semantic image–text conflict.
</p>
<div class="gallery">
""" + "\n".join(cards) + """
</div>
</body>
</html>
"""

(out / "case_gallery.html").write_text(
    page, encoding="utf-8"
)

print("QUALITATIVE GALLERY PASS")
print("Selected cases:", len(selected))
for name, count in counts.items():
    print(f"{name}: {count} candidates")
print("Output:", out)
print("Manual semantic review still required.")

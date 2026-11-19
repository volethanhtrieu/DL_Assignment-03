
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

repo = Path("DL_Assignment-03").resolve()
sys.path.insert(0, str(repo / "m2m3"))

from experiment import validate_locked_repository, normalise_text

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
raw = (DATA_ROOT / "Raw/mirror/humanitarian")
out = (DATA_ROOT / "Features/clip_vit_b32_v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

if out.exists() and any(out.iterdir()):
    raise RuntimeError(f"Refusing to overwrite: {out}")

lock, manifests = validate_locked_repository(repo)
provenance = json.loads(
    (repo / "A3-M1/provenance/mirror_downloads.json").read_text()
)
sources = {Path(x["path"]).name: x for x in provenance["files"]}

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("Loading pinned CLIP on", device, flush=True)

processor = CLIPProcessor.from_pretrained(
    MODEL, revision=REVISION, local_files_only=True, use_fast=False
)
model = CLIPModel.from_pretrained(
    MODEL, revision=REVISION, local_files_only=True
).eval().to(device)

assert getattr(model.config, "_commit_hash", None) == REVISION

def encode(pending):
    images = []
    texts = []

    for manifest, record in pending:
        blob = record["image"]["bytes"]
        text = record["text"]

        assert blob and text
        assert sha_bytes(blob) == manifest["image_sha256"]
        assert sha_bytes(text.encode("utf-8")) == manifest["raw_text_sha256"]
        assert sha_bytes(
            normalise_text(text).encode("utf-8")
        ) == manifest["normalized_text_sha256"]

        assert str(record["tweet_id"]) == manifest["tweet_id"]
        assert record["event_name"] == manifest["event_name"]
        assert int(record["label"]) == int(manifest["label_id"])

        with Image.open(io.BytesIO(blob)) as image:
            images.append(image.convert("RGB"))

        # Preserve original tweet content for CLIP.
        texts.append(text)

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model(**inputs)

    image_features = output.image_embeds.float().cpu().numpy()
    text_features = output.text_embeds.float().cpu().numpy()

    assert image_features.shape == (len(pending), 512)
    assert text_features.shape == (len(pending), 512)

    return image_features, text_features

metadata = {
    "model": MODEL,
    "revision": REVISION,
    "protocol": lock["primary_protocol"],
    "dataset_version": lock["dataset_version"],
    "processor_fast": False,
    "text_input": "raw_tweet",
    "max_tokens": 77,
    "batch_size": 16,
    "device": device,
    "torch": torch.__version__,
    "splits": {},
    "npz_sha256": {},
}

for split, pattern, expected_source in (
    ("train", "train-*.parquet", 6126),
    ("dev", "validation-*.parquet", 998),
):
    wanted = {r["row_id"]: r for r in manifests[split]}
    collected = []
    scanned = 0

    files = sorted(raw.glob(pattern))
    assert len(files) == (2 if split == "train" else 1)

    print(f"Processing {split}...", flush=True)

    for path in files:
        source = sources[path.name]
        assert path.stat().st_size == source["bytes"]
        assert sha_file(path) == source["sha256"]

        for batch in pq.ParquetFile(path).iter_batches(batch_size=24):
            pending = []

            for record in batch.to_pylist():
                scanned += 1
                row_id = f"{split}:{record['image_id']}"
                manifest = wanted.pop(row_id, None)

                if manifest is not None:
                    pending.append((manifest, record))

            for start in range(0, len(pending), 16):
                chunk = pending[start:start + 16]
                image_features, text_features = encode(chunk)

                for i, (manifest, _) in enumerate(chunk):
                    collected.append((
                        manifest["row_id"],
                        manifest["group_id"],
                        int(manifest["label_id"]),
                        image_features[i],
                        text_features[i],
                    ))

            if len(collected) and len(collected) % 500 < 24:
                print(f"  {split}: {len(collected)} pairs", flush=True)

    assert scanned == expected_source
    assert not wanted, f"Missing locked pairs: {len(wanted)}"
    assert len(collected) == lock["split_counts"][split]

    collected.sort(key=lambda x: x[0])

    row_ids = np.array([x[0] for x in collected])
    group_ids = np.array([x[1] for x in collected])
    labels = np.array([x[2] for x in collected], dtype=np.int64)
    image = np.stack([x[3] for x in collected]).astype(np.float32)
    text = np.stack([x[4] for x in collected]).astype(np.float32)

    assert np.isfinite(image).all()
    assert np.isfinite(text).all()
    assert len(set(row_ids.tolist())) == len(row_ids)

    actual = {
        str(r): (str(g), int(y))
        for r, g, y in zip(row_ids, group_ids, labels)
    }
    expected = {
        r["row_id"]: (r["group_id"], int(r["label_id"]))
        for r in manifests[split]
    }
    assert actual == expected

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.npz"

    np.savez_compressed(
        path,
        row_ids=row_ids,
        group_ids=group_ids,
        y=labels,
        image=image,
        text=text,
    )

    metadata["splits"][split] = len(labels)
    metadata["npz_sha256"][split] = sha_file(path)

    print(f"{split.upper()} PASS: {len(labels)} aligned pairs", flush=True)

(out / "metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)

assert not (out / "test.npz").exists()
print("CLIP FEATURE EXTRACTION PASS")
print("No test features were extracted.")

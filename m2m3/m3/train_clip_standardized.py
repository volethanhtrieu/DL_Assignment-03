
import json
import sys
import statistics
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

import numpy as np

repo = REPO_ROOT
sys.path.insert(0, str(repo / "m2m3"))

import experiment as exp

exp.TEXT_DIM = 512

source = (DATA_ROOT / "Features/clip_vit_b32_v1")
features = (DATA_ROOT / "Features/clip_zscore_v1")
runs = (DATA_ROOT / "Results/runs/clip_zscore_v1")

for path in (features, runs):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite: {path}")

lock, manifests = exp.validate_locked_repository(repo)

metadata = json.loads(
    (source / "metadata.json").read_text(encoding="utf-8")
)

assert metadata["dataset_version"] == lock["dataset_version"]
assert metadata["protocol"] == "filtered_original"
assert metadata["revision"] == (
    "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
assert not (source / "test.npz").exists()

data = {}
for split in ("train", "dev"):
    path = source / f"{split}.npz"
    assert exp.sha256(path) == metadata["npz_sha256"][split]

    d = exp.load_features(path, expected_split=split)

    actual = {
        str(r): (str(g), int(y))
        for r, g, y in zip(d["row_ids"], d["group_ids"], d["y"])
    }
    expected = {
        r["row_id"]: (r["group_id"], int(r["label_id"]))
        for r in manifests[split]
    }
    assert actual == expected
    data[split] = d

features.mkdir(parents=True)

# Fit standardization using TRAIN only.
normalizers = {}

for modality in ("image", "text"):
    x = data["train"][modality]
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-4)

    normalizers[f"{modality}_mean"] = mean
    normalizers[f"{modality}_std"] = std

    for split in ("train", "dev"):
        d = data[split]
        d[modality] = (
            (d[modality] - mean) / std
        ).astype(np.float32)

np.savez_compressed(features / "normalizer.npz", **normalizers)

hashes = {}
for split, d in data.items():
    path = features / f"{split}.npz"
    np.savez_compressed(path, **d)
    hashes[split] = exp.sha256(path)

    exp.load_features(path, expected_split=split)
    print(f"{split.upper()} VERIFIED: {len(d['y'])}", flush=True)

assert not (features / "test.npz").exists()

provenance = {
    "experiment": "clip_zscore_v1",
    "source_model": metadata["model"],
    "source_revision": metadata["revision"],
    "source_sha256": metadata["npz_sha256"],
    "transformation": "per-dimension zscore; train-fitted",
    "normalizer_sha256": exp.sha256(features / "normalizer.npz"),
    "npz_sha256": hashes,
    "test_used": False,
}

(features / "metadata.json").write_text(
    json.dumps(provenance, indent=2),
    encoding="utf-8",
)

scores = []

for seed in (42, 43, 44):
    result = exp.train_one(
        model_name="gated",
        seed=seed,
        feat=features,
        runs=runs,
        epochs=30,
        batch_size=64,
        patience=5,
        device_name="cuda",
    )

    f1 = result["dev"]["macro_f1"]
    scores.append(f1)

    print(
        f"Seed {seed}: F1={f1:.4f}, "
        f"epoch={result['best_epoch']}",
        flush=True,
    )

print(
    "CLIP ZSCORE GATED DEV:",
    f"{statistics.mean(scores):.4f} "
    f"+/- {statistics.stdev(scores):.4f}",
)
print("Previous CLIP Gated: 0.8391 +/- 0.0151")
print("Target: >= 0.8500")


import json
import statistics
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

repo = REPO_ROOT
sys.path.insert(0, str(repo / "m2m3"))

import experiment as exp

features = (DATA_ROOT / "Features/clip_vit_b32_v1")
runs = (DATA_ROOT / "Results/runs/clip_v1")

# CLIP produces 512-dimensional embeddings for BOTH modalities.
# Reuse the original M3 training implementation without modifying it.
exp.TEXT_DIM = 512

if runs.exists() and any(runs.iterdir()):
    raise RuntimeError(f"Refusing to overwrite existing results: {runs}")

lock, manifests = exp.validate_locked_repository(repo)
metadata = json.loads(
    (features / "metadata.json").read_text(encoding="utf-8")
)

assert metadata["model"] == "openai/clip-vit-base-patch32"
assert metadata["revision"] == (
    "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)
assert metadata["protocol"] == "filtered_original"
assert metadata["dataset_version"] == lock["dataset_version"]
assert metadata["splits"] == {"train": 6001, "dev": 986}
assert metadata["max_tokens"] == 77
assert metadata["processor_fast"] is False
assert metadata["text_input"] == "raw_tweet"

# Independently verify both feature files against locked manifests.
for split in ("train", "dev"):
    path = features / f"{split}.npz"

    assert exp.sha256(path) == metadata["npz_sha256"][split]

    data = exp.load_features(path, expected_split=split)

    actual = {
        str(row_id): (str(group_id), int(label))
        for row_id, group_id, label in zip(
            data["row_ids"], data["group_ids"], data["y"]
        )
    }

    expected = {
        row["row_id"]: (
            row["group_id"], int(row["label_id"])
        )
        for row in manifests[split]
    }

    assert actual == expected
    print(f"{split.upper()} FEATURE VERIFICATION PASS", flush=True)

# No test.npz is allowed in this experimental feature directory.
assert not (features / "test.npz").exists()

runs.mkdir(parents=True)

experiment_metadata = {
    "experiment": "clip_vit_b32_v1",
    "encoder": metadata["model"],
    "encoder_revision": metadata["revision"],
    "dataset_version": lock["dataset_version"],
    "data_protocol": "filtered_original",
    "training_implementation": "m2m3/experiment.py",
    "architecture": "experiment.Head with TEXT_DIM=512",
    "models": ["concat", "gated"],
    "seeds": [42, 43, 44],
    "selection_metric": "dev_macro_f1",
    "test_used": False,
    "feature_sha256": metadata["npz_sha256"],
}

(runs / "experiment_metadata.json").write_text(
    json.dumps(experiment_metadata, indent=2),
    encoding="utf-8"
)

for model in ("concat", "gated"):
    scores = []

    for seed in (42, 43, 44):
        result = exp.train_one(
            model_name=model,
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
            f"{model} | seed={seed} | "
            f"dev_macro_f1={f1:.4f} | "
            f"best_epoch={result['best_epoch']}",
            flush=True,
        )

    mean = statistics.mean(scores)
    sd = statistics.stdev(scores)

    print(
        f"\nCLIP {model.upper()} "
        f"DEV MACRO-F1: {mean:.4f} +/- {sd:.4f}\n",
        flush=True,
    )

print("Original M3 Gated: 0.8131 +/- 0.0027")
print("Target dev macro-F1: >= 0.8500")
print("CLIP DEVELOPMENT EXPERIMENT COMPLETE")

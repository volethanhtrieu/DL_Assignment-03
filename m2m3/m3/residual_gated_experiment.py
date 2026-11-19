
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from _m3_paths import DATA_ROOT

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

repo = Path("DL_Assignment-03").resolve()
sys.path.insert(0, str(repo / "m2m3"))

from experiment import (
    check_feature_provenance, load_features, score, set_seed
)

features = (DATA_ROOT / "Features/features/primary")
output = (DATA_ROOT / "Results/runs/residual_gated_v1")

# Never overwrite earlier experiment evidence.
if output.exists() and any(output.iterdir()):
    raise RuntimeError(f"Output already exists: {output}")

check_feature_provenance(features, repo)

train = load_features(features / "train.npz", expected_split="train")
dev = load_features(features / "dev.npz", expected_split="dev")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Dataset:", len(train["y"]), "train;", len(dev["y"]), "dev")

class ResidualGated(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_proj = nn.Linear(512, 128)
        self.text_proj = nn.Linear(768, 128)
        self.gate = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    def forward(self, image, text):
        a = torch.relu(self.image_proj(image))
        b = torch.relu(self.text_proj(text))
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=1)))
        fused = g * a + (1 - g) * b
        combined = torch.cat([a, b, fused], dim=1)
        return self.classifier(self.dropout(combined))


def predict_dev(model):
    model.eval()
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(dev["y"]), 256):
            end = start + 256
            image = torch.from_numpy(dev["image"][start:end]).float().to(device)
            text = torch.from_numpy(dev["text"][start:end]).float().to(device)
            predictions.append(model(image, text).argmax(1).cpu().numpy())

    return np.concatenate(predictions)


counts = np.bincount(train["y"], minlength=5)
weights = counts.sum() / (5.0 * counts.astype(float))
weights /= weights.mean()

dataset = TensorDataset(
    torch.from_numpy(train["image"]).float(),
    torch.from_numpy(train["text"]).float(),
    torch.from_numpy(train["y"]).long(),
)

output.mkdir(parents=True)
results = []

for seed in (42, 43, 44):
    set_seed(seed)
    model = ResidualGated().to(device)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001
    )

    folder = output / f"seed_{seed}"
    folder.mkdir()

    best_f1 = -1.0
    best_epoch = 0
    patience_count = 0
    started = time.perf_counter()
    history = []

    for epoch in range(1, 31):
        model.train()
        losses = []

        for image, text, labels in loader:
            image = image.to(device)
            text = text.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image, text), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        predictions = predict_dev(model)
        metrics = score(dev["y"], predictions)
        f1 = metrics["macro_f1"]

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_macro_f1": f1,
        })

        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {"model": "residual_gated_v1",
                 "seed": seed,
                 "best_epoch": epoch,
                 "state_dict": model.state_dict()},
                folder / "best.pt",
            )
        else:
            patience_count += 1

        if patience_count >= 5:
            break

    checkpoint = torch.load(
        folder / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict_dev(model)
    metrics = score(dev["y"], predictions)

    record = {
        "model": "residual_gated_v1",
        "seed": seed,
        "best_epoch": best_epoch,
        "dev": metrics,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "train_and_dev_seconds": time.perf_counter() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "hyperparameters": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "dropout": 0.2,
            "epochs": 30,
            "patience": 5,
            "class_weights": weights.tolist(),
        },
        "history": history,
    }

    (folder / "run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    with (folder / "dev_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "group_id", "y_true", "y_pred"])
        for i, prediction in enumerate(predictions):
            writer.writerow([
                str(dev["row_ids"][i]),
                str(dev["group_ids"][i]),
                int(dev["y"][i]),
                int(prediction),
            ])

    results.append(metrics["macro_f1"])
    print(f"Seed {seed}: macro-F1={metrics['macro_f1']:.4f}, epoch={best_epoch}")

print(
    f"RESIDUAL GATED DEV: "
    f"{statistics.mean(results):.4f} "
    f"+/- {statistics.stdev(results):.4f}"
)
print("Original Gated dev: 0.8131 +/- 0.0027")
print("Original Concat dev: 0.8116 +/- 0.0041")

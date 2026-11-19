
import csv
import json
import sys
from pathlib import Path
from _m3_paths import DATA_ROOT, REPO_ROOT

import gradio as gr
import torch
from torchvision import models
from transformers import AutoModel, AutoTokenizer

HERE = DATA_ROOT
ROOT = HERE.parent
REPO = REPO_ROOT

sys.path.insert(0, str(REPO / "m2m3"))
import experiment as exp

RUNS = HERE / "Results/runs/primary"
FEATURES = HERE / "Features/features/primary"

SEED = 42
MODELS = ("image", "text", "concat", "fixed_gate", "gated")
DISPLAY = {
    "image": "Image-only",
    "text": "Text-only",
    "concat": "Concatenation",
    "fixed_gate": "Fixed Gate",
    "gated": "Learned Gated Fusion",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def class_name(index):
    return exp.CLASSES[index].replace("_", " ").title()


print("Loading original M3 encoders...", flush=True)

weights = models.ResNet18_Weights.IMAGENET1K_V1
image_encoder = models.resnet18(weights=weights)
image_encoder.fc = torch.nn.Identity()
image_encoder.eval().to(DEVICE)

image_transform = weights.transforms()

tokenizer = AutoTokenizer.from_pretrained(
    exp.TEXT_MODEL,
    revision=exp.TEXT_REVISION,
)
text_encoder = AutoModel.from_pretrained(
    exp.TEXT_MODEL,
    revision=exp.TEXT_REVISION,
).eval().to(DEVICE)

heads = {}

for name in MODELS:
    path = RUNS / name / f"seed_{SEED}"
    run = json.loads(
        (path / "run.json").read_text(encoding="utf-8")
    )

    assert run["model"] == name
    assert run["seed"] == SEED
    assert run["smoke_only"] is False

    checkpoint = torch.load(
        path / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["model"] == name
    assert checkpoint["seed"] == SEED
    assert checkpoint["best_epoch"] == run["best_epoch"]

    head = exp.Head(name)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval().to(DEVICE)
    heads[name] = head

print("Loaded 5 original M3 checkpoints.", flush=True)


def verify_saved_predictions():
    """Use saved dev features only; no test evaluation."""
    data = exp.load_features(
        FEATURES / "dev.npz",
        expected_split="dev",
    )

    indices = list(range(16))
    image_vectors = torch.as_tensor(
        data["image"][indices], dtype=torch.float32,
        device=DEVICE,
    )
    text_vectors = torch.as_tensor(
        data["text"][indices], dtype=torch.float32,
        device=DEVICE,
    )

    with torch.inference_mode():
        for name in MODELS:
            path = (
                RUNS / name / f"seed_{SEED}"
                / "dev_predictions.csv"
            )

            with path.open(
                encoding="utf-8", newline=""
            ) as f:
                expected = {
                    row["row_id"]: int(row["y_pred"])
                    for row in csv.DictReader(f)
                }

            predictions = heads[name](
                image_vectors, text_vectors
            ).argmax(dim=-1).cpu().tolist()

            for i, predicted in zip(indices, predictions):
                row_id = str(data["row_ids"][i])
                if predicted != expected[row_id]:
                    raise RuntimeError(
                        f"Checkpoint mismatch: {name}, {row_id}"
                    )

    print(
        "DEMO CHECKPOINT VALIDATION PASS: "
        "16 dev samples x 5 models",
        flush=True,
    )


verify_saved_predictions()


@torch.inference_mode()
def predict(image, tweet):
    if image is None:
        raise gr.Error("Please upload an image.")

    if not tweet or not tweet.strip():
        raise gr.Error("Please enter tweet text.")

    normalized = exp.normalise_text(tweet)
    if not normalized:
        raise gr.Error(
            "Tweet contains no usable text after preprocessing."
        )

    # Same image preprocessing as M1/M2.
    image_tensor = image_transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    image_features = image_encoder(image_tensor)

    # Same text normalization, tokenization and pooling.
    tokens = tokenizer(
        normalized,
        max_length=exp.MAX_TOKENS,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    ids = tokens["input_ids"].to(DEVICE)
    mask = tokens["attention_mask"].to(DEVICE)

    hidden = text_encoder(
        input_ids=ids,
        attention_mask=mask,
    ).last_hidden_state

    expanded_mask = mask.unsqueeze(-1).to(hidden.dtype)

    text_features = (
        (hidden * expanded_mask).sum(dim=1)
        / expanded_mask.sum(dim=1).clamp(min=1)
    )

    comparison = []
    gated_scores = None
    learned_gate = None

    for name in MODELS:
        logits, gate = heads[name](
            image_features,
            text_features,
            return_gate=True,
        )

        probabilities = torch.softmax(
            logits, dim=-1
        ).cpu().numpy()[0]

        prediction = int(probabilities.argmax())
        confidence = float(probabilities[prediction])

        if name == "gated":
            learned_gate = float(gate.item())
            gated_scores = {
                class_name(i): float(probabilities[i])
                for i in range(exp.NUM_CLASSES)
            }

        gate_value = {
            "image": "N/A",
            "text": "N/A",
            "concat": "N/A",
            "fixed_gate": "0.500 (fixed)",
        }.get(name)

        if name == "gated":
            gate_value = f"{learned_gate:.3f} (learned)"

        comparison.append([
            DISPLAY[name],
            class_name(prediction),
            f"{confidence:.3f}",
            gate_value,
        ])

    g = learned_gate

    gate_html = f"""
    <div style="padding:18px;border:1px solid #ddd;
                border-radius:12px;font-family:Arial">
      <h3>Learned Gate — Seed 42</h3>
      <div style="display:flex;height:28px;overflow:hidden;
                  border-radius:6px">
        <div style="width:{100*g:.2f}%;
                    background:#4779C0"></div>
        <div style="width:{100*(1-g):.2f}%;
                    background:#D4A65E"></div>
      </div>
      <p><b>Image coefficient:</b> {g:.3f}</p>
      <p><b>Text coefficient:</b> {1-g:.3f}</p>
      <p style="color:#666;font-size:13px">
        These are fusion coefficients, not causal
        attribution or calibrated importance.
      </p>
    </div>
    """

    return comparison, gated_scores, gate_html


with gr.Blocks(
    title="Group Goat | CrisisMMD Multimodal Demo",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # Humanitarian Multimodal Classification
        **CO3133 Deep Learning — Group Goat**

        Compare five frozen-encoder classification models
        using an image and accompanying tweet.

        **Architecture:** ResNet-18 + DistilBERT<br>
        **Protocol:** Custom filtered CrisisMMD, seed 42<br>
        **Purpose:** Educational research demonstration
        """
    )

    with gr.Row():
        image_input = gr.Image(
            type="pil",
            image_mode="RGB",
            label="Upload an image",
        )

        tweet_input = gr.Textbox(
            label="Tweet text",
            lines=7,
            placeholder=(
                "Enter text describing the image "
                "or a humanitarian event..."
            ),
        )

    run_button = gr.Button(
        "Classify Image + Tweet",
        variant="primary",
    )

    comparison_output = gr.Dataframe(
        headers=[
            "Model",
            "Predicted class",
            "Top softmax score",
            "Image gate",
        ],
        datatype=["str", "str", "str", "str"],
        interactive=False,
        label="Model Comparison — Seed 42",
    )

    with gr.Row():
        probability_output = gr.Label(
            label="Learned Gate — Class Scores",
            num_top_classes=5,
        )

        gate_output = gr.HTML(
            label="Gate Visualization",
        )

    gr.Markdown(
        """
        **Interpretation notes**

        - Softmax scores are not calibrated probabilities.
        - Gate coefficients describe fusion behavior,
          not causal feature importance.
        - Outputs are predictions, not verified disaster reports.
        - The demo runs locally and does not retrain models.
        """
    )

    run_button.click(
        fn=predict,
        inputs=[image_input, tweet_input],
        outputs=[
            comparison_output,
            probability_output,
            gate_output,
        ],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    print("Launching Group Goat Demo...", flush=True)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )

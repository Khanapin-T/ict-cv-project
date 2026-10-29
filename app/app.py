"""SmartWaste AI demo: upload a photo of a waste item and get its TrashNet class."""

import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pth"
CLASSES_PATH = ROOT / "models" / "classes.json"
TOP_K = 3

# Must match the eval transform in src/dataset.py (see AGENTS.md, "Interface contract").
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model() -> tuple[nn.Module, list[str]]:
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return model.eval(), classes


def predict(model: nn.Module, image: Image.Image) -> torch.Tensor:
    x = EVAL_TRANSFORM(image).unsqueeze(0)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=1)[0]


st.set_page_config(page_title="SmartWaste AI", page_icon="♻️")
st.title("SmartWaste AI")
st.caption("Upload a photo of a waste item. A MobileNetV2 fine-tuned on TrashNet predicts its class.")

if not (MODEL_PATH.exists() and CLASSES_PATH.exists()):
    st.info(
        "**Model not trained yet.** `models/best.pth` and `models/classes.json` were not found. "
        "Train a model with `src/train.py`, then reload this page."
    )
    st.stop()

try:
    model, classes = load_model()
except Exception as exc:  # weights that don't match the contract in AGENTS.md
    st.error(f"Could not load `models/best.pth`: {exc}")
    st.stop()

uploaded = st.file_uploader("Waste photo", type=["jpg", "jpeg", "png", "webp"])
if uploaded is None:
    st.stop()

try:
    image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
except UnidentifiedImageError:
    st.error("This file is not a readable image.")
    st.stop()

probs = predict(model, image)
top = torch.topk(probs, k=min(TOP_K, len(classes)))
best_prob, best_idx = top.values[0].item(), top.indices[0].item()

left, right = st.columns(2)
left.image(image, width="stretch")
with right:
    st.metric("Predicted class", classes[best_idx])
    st.metric("Confidence", f"{best_prob:.1%}")
    st.subheader(f"Top-{len(top.indices)}")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        st.progress(p, text=f"{classes[i]}: {p:.1%}")

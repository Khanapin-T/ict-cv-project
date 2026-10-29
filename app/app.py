"""SmartWaste AI demo: upload a photo of a waste item and get its TrashNet class."""

import json
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # streamlit puts app/ on sys.path, not the repo root

from src.dataset import EVAL_TRANSFORM  # the same transform as val/test (AGENTS.md, "Interface contract")
from src.models import get_model

MODEL_PATH = ROOT / "models" / "best.pth"
CONFIG_PATH = ROOT / "models" / "config.json"
TOP_K = 3


@st.cache_resource
def load_model(version: tuple[int, int]) -> tuple[nn.Module, dict]:
    """`version` (the files' mtimes) is the cache key, so a replaced model loads on page reload."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model = get_model(config["arch"], num_classes=len(config["classes"]), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    return model.eval(), config


def predict(model: nn.Module, image: Image.Image) -> torch.Tensor:
    x = EVAL_TRANSFORM(image).unsqueeze(0)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=1)[0]


st.set_page_config(page_title="SmartWaste AI", page_icon="♻️")
st.title("SmartWaste AI")
st.caption("Upload a photo of a waste item. A model trained on TrashNet predicts its class.")

if not (MODEL_PATH.exists() and CONFIG_PATH.exists()):
    st.info(
        "**Model not trained yet.** `models/best.pth` and `models/config.json` were not found. "
        "Train a model with `src/train.py`, then reload this page."
    )
    st.stop()

try:
    model, config = load_model((MODEL_PATH.stat().st_mtime_ns, CONFIG_PATH.stat().st_mtime_ns))
except Exception as exc:  # weights or config that don't match the contract in AGENTS.md
    st.error(f"Could not load the model from `models/`: {exc}")
    st.stop()

classes = config["classes"]
st.caption(f"Model: `{config['arch']}`, run `{config.get('run_name', '?')}`")

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

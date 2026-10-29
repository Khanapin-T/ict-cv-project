"""Single-image inference, shared by the Streamlit app and src/demo_images.py (contract: AGENTS.md)."""

import torch
from PIL import Image, ImageOps
from torch import nn

from src.dataset import EVAL_TRANSFORM  # the same transform as val/test


def load_image(source) -> Image.Image:
    """Open a path or file-like object as RGB, applying the EXIF rotation (like ImageListDataset)."""
    with Image.open(source) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def predict_image(model: nn.Module, image: Image.Image) -> torch.Tensor:
    """Class probabilities (softmax) for one image."""
    x = EVAL_TRANSFORM(image).unsqueeze(0)
    with torch.inference_mode():
        return torch.softmax(model(x), dim=1)[0]

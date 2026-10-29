"""Streamlit app tests (AppTest). The app reads a temp models dir (SMARTWASTE_MODELS_DIR), never the real models/."""

import json
from pathlib import Path

import pytest
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest

from src.dataset import CLASSES
from src.models import get_model

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")  # AppTest resolves relative paths against the caller, so pass an absolute one
SAMPLE = ROOT / "data" / "raw" / "glass" / "glass1.jpg"


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    """An empty models dir for the app, so the committed models/best.pth is never read or touched."""
    monkeypatch.setenv("SMARTWASTE_MODELS_DIR", str(tmp_path))
    st.cache_resource.clear()
    yield tmp_path
    st.cache_resource.clear()


@pytest.fixture
def install_model(models_dir):
    """Returns install(arch, weights_arch) that writes a dummy model into the temp models dir."""

    def install(arch: str, weights_arch: str | None = None) -> None:
        torch.save(get_model(weights_arch or arch, dropout=0.3).state_dict(), models_dir / "best.pth")
        config = {"arch": arch, "classes": CLASSES, "run_name": "test"}
        (models_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    return install


def run_app(upload: tuple[str, bytes, str] | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    if upload is not None:
        at.file_uploader[0].set_value(upload).run()
    assert not at.exception, at.exception
    return at


def test_app_uses_dataset_eval_transform():
    source = Path(APP).read_text(encoding="utf-8")
    assert "from src.inference import load_image, predict_image" in source
    assert "transforms.Compose" not in source
    inference = (ROOT / "src" / "inference.py").read_text(encoding="utf-8")
    assert "from src.dataset import EVAL_TRANSFORM" in inference and "transforms.Compose" not in inference


def test_no_model_shows_notice():
    at = run_app()
    assert any("Model not trained yet" in info.value for info in at.info)


@pytest.mark.skipif(not SAMPLE.exists(), reason="data/raw/ is missing; run python src/prepare_split.py")
def test_prediction_with_dummy_model(install_model):
    install_model("baseline_cnn")
    at = run_app(("glass1.jpg", SAMPLE.read_bytes(), "image/jpeg"))
    predicted = [m.value for m in at.metric if m.label == "Predicted class"]
    assert len(predicted) == 1 and predicted[0] in CLASSES
    assert len(at.get("progress")) == 3


def test_unreadable_upload_shows_error(install_model):
    install_model("baseline_cnn")
    at = run_app(("broken.jpg", b"not an image", "image/jpeg"))
    assert any("not a readable image" in e.value for e in at.error)


def test_mismatched_checkpoint_shows_error(install_model):
    install_model("modified_cnn", weights_arch="baseline_cnn")
    at = run_app()
    assert any("Could not load the model" in e.value for e in at.error)

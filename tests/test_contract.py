"""Contract tests for AGENTS.md "Interface contract". CPU only, tiny subsets (max_batches)."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src import dataset

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

needs_data = pytest.mark.skipif(
    not (ROOT / "data" / "raw" / ".complete").exists(),
    reason="data/raw/ is missing; run python src/prepare_split.py",
)


# ---------------------------------------------------------------- dataset


def test_class_order():
    assert dataset.CLASSES == CLASSES
    assert dataset.CLASSES == sorted(dataset.CLASSES)


def test_split_disjoint_and_sizes():
    import csv

    with open(dataset.SPLITS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = [r["path"] for r in rows]
    assert len(paths) == len(set(paths))
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    assert counts == {"train": 1744, "val": 374, "test": 374}


def test_eval_transform_shape():
    x = dataset.EVAL_TRANSFORM(Image.new("RGB", (512, 384), "white"))
    assert x.shape == (3, 224, 224)


@needs_data
def test_dataloaders_batches():
    loaders = dataset.get_dataloaders(4, aug=False, synthetic=False)
    for loader, n in zip(loaders, (1744, 374, 374)):
        assert len(loader.dataset) == n
        x, y = next(iter(loader))
        assert x.shape == (4, 3, 224, 224)
        assert y.dtype == torch.int64 and 0 <= y.min() and y.max() <= 5


@needs_data
def test_aug_train_only():
    train, val, test = dataset.get_dataloaders(4, aug=True, synthetic=False)
    assert train.dataset.transform is dataset.TRAIN_AUG_TRANSFORM
    assert val.dataset.transform is dataset.EVAL_TRANSFORM
    assert test.dataset.transform is dataset.EVAL_TRANSFORM


@needs_data
def test_synthetic_train_only(tmp_path, monkeypatch):
    for cls in CLASSES:
        (tmp_path / cls).mkdir()
        Image.new("RGBA", (300, 200), (10, 20, 30, 128)).save(tmp_path / cls / "a.png")
        Image.new("L", (64, 80), 128).save(tmp_path / cls / "b.jpg")
    monkeypatch.setattr(dataset, "SYNTHETIC_DIR", tmp_path)

    train, val, test = dataset.get_dataloaders(4, aug=False, synthetic=True)
    assert len(train.dataset) == 1744 + 12
    assert train.dataset.n_synthetic == 12
    assert (len(val.dataset), len(test.dataset)) == (374, 374)
    ds = train.dataset
    for i in range(len(ds) - 12, len(ds)):
        x, y = ds[i]
        assert x.shape == (3, 224, 224)
        assert CLASSES[y] == ds.items[i][0].parent.name


@needs_data
def test_synthetic_empty_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "SYNTHETIC_DIR", tmp_path)
    with pytest.warns(UserWarning, match="synthetic"):
        train, _, _ = dataset.get_dataloaders(4, aug=False, synthetic=True)
    assert train.dataset.n_synthetic == 0
    assert len(train.dataset) == 1744


def test_missing_raw_data_message(tmp_path, monkeypatch):
    csv_path = tmp_path / "splits.csv"
    csv_path.write_text("path,label,split\ndata/raw/cardboard/__missing__.jpg,cardboard,train\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "SPLITS_CSV", csv_path)
    with pytest.raises(FileNotFoundError, match="prepare_split"):
        dataset.get_dataloaders(4, aug=False, synthetic=False)

    monkeypatch.setattr(dataset, "SPLITS_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="prepare_split"):
        dataset.read_split("train")


# ---------------------------------------------------------------- models

MODEL_NAMES = ["baseline_cnn", "modified_cnn", "mobilenet_v2"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_forward(name):
    from src.models import get_model

    model = get_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 6)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_dropout_keeps_state_dict_keys(name):
    from src.models import get_model

    a = get_model(name, dropout=0.0, pretrained=False).state_dict()
    b = get_model(name, dropout=0.3, pretrained=False).state_dict()
    assert a.keys() == b.keys()
    assert all(a[k].shape == b[k].shape for k in a)


def test_modified_has_one_more_block():
    from torch import nn

    from src.models import get_model

    def n_convs(name):
        return sum(isinstance(m, nn.Conv2d) for m in get_model(name).modules())

    assert (n_convs("baseline_cnn"), n_convs("modified_cnn")) == (3, 4)


def test_unknown_model_raises():
    from src.models import get_model

    with pytest.raises(ValueError):
        get_model("resnet50")


# ---------------------------------------------------------------- evaluate

EXP_KEYS = {"run_name", "accuracy", "precision_macro", "recall_macro", "f1_macro", "per_class", "confusion_matrix"}


def test_evaluate_outputs(tmp_path):
    import json

    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.evaluate import evaluate

    torch.manual_seed(0)
    loader = DataLoader(TensorDataset(torch.randn(12, 3, 8, 8), torch.arange(12) % 6), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 6))
    result = evaluate(model, loader, CLASSES, "dummy", reports_dir=tmp_path)

    assert set(result) == EXP_KEYS
    assert list(result["per_class"]) == CLASSES
    assert set(result["per_class"]["glass"]) == {"precision", "recall", "f1", "support"}
    assert len(result["confusion_matrix"]) == 6 and all(len(r) == 6 for r in result["confusion_matrix"])
    assert sum(map(sum, result["confusion_matrix"])) == 12
    assert json.loads((tmp_path / "exp_dummy.json").read_text(encoding="utf-8")) == result
    assert (tmp_path / "cm_dummy.png").stat().st_size > 0

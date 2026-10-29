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


# ---------------------------------------------------------------- train

HISTORY_KEYS = {"run_name", "model_name", "num_params", "train_time_sec", "train_loss", "val_loss", "val_f1_macro",
                "best_epoch", "best_val_f1_macro", "hparams"}
HPARAM_KEYS = {"epochs", "lr", "optimizer", "dropout", "aug", "synthetic", "n_synthetic", "batch_size", "seed",
               "max_batches"}


def _leading_params(fn, n):
    import inspect

    return list(inspect.signature(fn).parameters)[:n]


def test_signatures():
    import inspect

    from src.dataset import get_dataloaders
    from src.evaluate import evaluate
    from src.models import get_model
    from src.train import train

    assert _leading_params(get_dataloaders, 3) == ["batch_size", "aug", "synthetic"]
    params = inspect.signature(get_model).parameters
    assert list(params)[:3] == ["name", "num_classes", "dropout"]
    assert (params["num_classes"].default, params["dropout"].default) == (6, 0.0)
    assert params["pretrained"].kind is inspect.Parameter.KEYWORD_ONLY and params["pretrained"].default is True
    assert _leading_params(train, 8) == ["model_name", "epochs", "lr", "optimizer", "dropout", "aug", "synthetic",
                                         "run_name"]
    assert _leading_params(evaluate, 4) == ["model", "loader", "class_names", "run_name"]


@needs_data
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_smoke_every_model(name, tmp_path):
    import json

    from src.models import get_model
    from src.train import train

    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    result = train(name, epochs=1, lr=1e-3, optimizer="adam", dropout=0.2, aug=True, synthetic=False,
                   run_name=f"smoke_{name}", batch_size=4, max_batches=1, pretrained=False,
                   models_dir=models_dir, reports_dir=reports_dir)

    assert set(result) == EXP_KEYS
    state = torch.load(models_dir / f"smoke_{name}.pth", map_location="cpu", weights_only=True)
    get_model(name, pretrained=False).load_state_dict(state)
    history = json.loads((reports_dir / f"history_smoke_{name}.json").read_text(encoding="utf-8"))
    assert set(history) == HISTORY_KEYS
    assert set(history["hparams"]) == HPARAM_KEYS
    assert all(len(history[k]) == 1 for k in ("train_loss", "val_loss", "val_f1_macro"))
    assert history["best_epoch"] == 1 and history["model_name"] == name
    assert (reports_dir / f"exp_smoke_{name}.json").exists()
    assert (reports_dir / f"cm_smoke_{name}.png").exists()


@pytest.mark.parametrize("run_name", ["best", "a/b", r"a\b"])
def test_train_rejects_bad_run_name(run_name, tmp_path):
    from src.train import train

    with pytest.raises(ValueError):
        train("baseline_cnn", 1, 1e-3, "adam", 0.0, False, False, run_name, models_dir=tmp_path, reports_dir=tmp_path)


def test_train_rejects_unknown_optimizer(tmp_path):
    from src.train import train

    with pytest.raises(ValueError):
        train("baseline_cnn", 1, 1e-3, "rmsprop", 0.0, False, False, "x", models_dir=tmp_path, reports_dir=tmp_path)


@needs_data
def test_train_cli_smoke(tmp_path):
    import subprocess
    import sys

    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", "baseline_cnn", "--epochs", "1",
           "--max-batches", "1", "--batch-size", "4", "--optimizer", "sgd", "--lr", "0.01", "--aug",
           "--run-name", "cli", "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r")]
    subprocess.run(cmd, check=True, cwd=tmp_path, capture_output=True, timeout=300)
    assert (tmp_path / "m" / "cli.pth").exists()
    assert (tmp_path / "r" / "history_cli.json").exists()

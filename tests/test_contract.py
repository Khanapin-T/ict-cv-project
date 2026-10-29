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
    loaders = dataset.get_dataloaders(4)
    for loader, n in zip(loaders, (1744, 374, 374)):
        assert len(loader.dataset) == n
        x, y = next(iter(loader))
        assert x.shape == (4, 3, 224, 224)
        assert y.dtype == torch.int64 and 0 <= y.min() and y.max() <= 5


@needs_data
def test_every_split_uses_eval_transform():
    train, val, test = dataset.get_dataloaders(4)
    assert all(loader.dataset.transform is dataset.EVAL_TRANSFORM for loader in (train, val, test))


@needs_data
def test_train_shuffle_is_seeded():
    def first_labels(seed):
        return next(iter(dataset.get_dataloaders(16, seed=seed)[0]))[1].tolist()

    assert first_labels(42) == first_labels(42)
    assert first_labels(42) != first_labels(7)


@needs_data
def test_missing_raw_data_message(tmp_path, monkeypatch):
    csv_path = tmp_path / "splits.csv"
    csv_path.write_text("path,label,split\ndata/raw/cardboard/__missing__.jpg,cardboard,train\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "SPLITS_CSV", csv_path)
    with pytest.raises(FileNotFoundError, match="prepare_split"):
        dataset.get_dataloaders(4)

    monkeypatch.setattr(dataset, "SPLITS_CSV", tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError, match="prepare_split"):
        dataset.read_split("train")


# ---------------------------------------------------------------- models

MODEL_NAMES = ["baseline_cnn", "modified_cnn"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models_forward(name):
    from src.models import get_model

    model = get_model(name).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 6)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_dropout_keeps_state_dict_keys(name):
    from src.models import get_model

    a = get_model(name, dropout=0.0).state_dict()
    b = get_model(name, dropout=0.3).state_dict()
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
    with pytest.raises(ValueError):
        get_model("mobilenet_v2")  # removed: option 1 compares CNNs trained from scratch


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
HPARAM_KEYS = {"epochs", "lr", "optimizer", "dropout", "batch_size", "seed", "max_batches"}


def _leading_params(fn, n):
    import inspect

    return list(inspect.signature(fn).parameters)[:n]


def test_signatures():
    import inspect

    from src.dataset import get_dataloaders
    from src.evaluate import evaluate
    from src.models import get_model
    from src.train import train

    params = inspect.signature(get_dataloaders).parameters
    assert list(params)[0] == "batch_size"
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(params.values())[1:])
    params = inspect.signature(get_model).parameters
    assert list(params) == ["name", "num_classes", "dropout"]
    assert (params["num_classes"].default, params["dropout"].default) == (6, 0.0)
    assert _leading_params(train, 6) == ["model_name", "epochs", "lr", "optimizer", "dropout", "run_name"]
    assert _leading_params(evaluate, 4) == ["model", "loader", "class_names", "run_name"]


@needs_data
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_smoke_every_model(name, tmp_path):
    import json

    from src.models import get_model
    from src.train import train

    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    result = train(name, epochs=1, lr=1e-3, optimizer="adam", dropout=0.2, run_name=f"smoke_{name}", batch_size=4,
                   max_batches=1, models_dir=models_dir, reports_dir=reports_dir)

    assert set(result) == EXP_KEYS
    state = torch.load(models_dir / f"smoke_{name}.pth", map_location="cpu", weights_only=True)
    get_model(name).load_state_dict(state)
    history = json.loads((reports_dir / f"history_smoke_{name}.json").read_text(encoding="utf-8"))
    assert set(history) == HISTORY_KEYS
    assert set(history["hparams"]) == HPARAM_KEYS
    assert all(len(history[k]) == 1 for k in ("train_loss", "val_loss", "val_f1_macro"))
    assert history["best_epoch"] == 1 and history["model_name"] == name
    assert (reports_dir / f"exp_smoke_{name}.json").exists()
    assert (reports_dir / f"cm_smoke_{name}.png").exists()


@needs_data
def test_train_same_seed_same_run(tmp_path):
    """Same seed -> same weight init and batch order, so the 3 runs differ only in what is compared."""
    import json

    from src.train import train

    losses = []
    for run in ("seed_a", "seed_b"):
        train("baseline_cnn", epochs=1, lr=1e-3, optimizer="adam", dropout=0.3, run_name=run, batch_size=4,
              max_batches=2, models_dir=tmp_path, reports_dir=tmp_path)
        losses.append(json.loads((tmp_path / f"history_{run}.json").read_text(encoding="utf-8"))["train_loss"])
    assert losses[0] == losses[1]


@pytest.mark.parametrize("run_name", ["best", "Best", "BEST", "a/b", r"a\b"])
def test_train_rejects_bad_run_name(run_name, tmp_path):
    from src.train import train

    with pytest.raises(ValueError):
        train("baseline_cnn", 1, 1e-3, "adam", 0.0, run_name, models_dir=tmp_path, reports_dir=tmp_path)


def test_train_rejects_unknown_optimizer(tmp_path):
    from src.train import train

    with pytest.raises(ValueError):
        train("baseline_cnn", 1, 1e-3, "rmsprop", 0.0, "x", models_dir=tmp_path, reports_dir=tmp_path)


@needs_data
def test_train_cli_smoke(tmp_path):
    import subprocess
    import sys

    cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--model", "baseline_cnn", "--epochs", "1",
           "--max-batches", "1", "--batch-size", "4", "--optimizer", "sgd", "--lr", "0.01",
           "--run-name", "cli", "--models-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r")]
    subprocess.run(cmd, check=True, cwd=tmp_path, capture_output=True, timeout=300)
    assert (tmp_path / "m" / "cli.pth").exists()
    assert (tmp_path / "r" / "history_cli.json").exists()


# ---------------------------------------------------------------- report + optimization

SUMMARY_COLUMNS = ["run_name", "model_name", "optimizer", "lr", "dropout", "epochs", "best_epoch", "num_params", "train_time_sec", "best_val_f1_macro", "accuracy", "precision_macro",
                   "recall_macro", "f1_macro", "trash_f1"]


def _fake_run(reports_dir, models_dir, run, model_name, val_f1, test_f1, optimizer="adam", lr=1e-3):
    """Write history/exp JSON (and a checkpoint) like train() does, without training."""
    import json

    from src.models import get_model

    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    hparams = {"epochs": 2, "lr": lr, "optimizer": optimizer, "dropout": 0.3, "batch_size": 32, "seed": 42,
               "max_batches": None}
    history = {"run_name": run, "model_name": model_name, "num_params": 1000, "train_time_sec": 60.0,
               "train_loss": [1.5, 1.0], "val_loss": [1.4, 1.1], "val_f1_macro": [val_f1 - 0.1, val_f1],
               "best_epoch": 2, "best_val_f1_macro": val_f1, "hparams": hparams}
    per_class = {c: {"precision": 0.5, "recall": 0.5, "f1": 0.25 if c == "trash" else test_f1, "support": 10}
                 for c in CLASSES}
    exp = {"run_name": run, "accuracy": test_f1 + 0.05, "precision_macro": test_f1, "recall_macro": test_f1,
           "f1_macro": test_f1, "per_class": per_class, "confusion_matrix": [[0] * 6] * 6}
    (reports_dir / f"history_{run}.json").write_text(json.dumps(history), encoding="utf-8")
    (reports_dir / f"exp_{run}.json").write_text(json.dumps(exp), encoding="utf-8")
    torch.save(get_model(model_name).state_dict(), models_dir / f"{run}.pth")


def test_build_summary_columns(tmp_path):
    import csv

    from src.report import build_summary

    _fake_run(tmp_path, tmp_path, "baseline_cnn", "baseline_cnn", 0.6, 0.55)
    _fake_run(tmp_path, tmp_path, "modified_cnn", "modified_cnn", 0.7, 0.65)
    rows = build_summary(reports_dir=tmp_path)
    with open(tmp_path / "summary.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == SUMMARY_COLUMNS
        assert [r["run_name"] for r in reader] == ["baseline_cnn", "modified_cnn"]
    assert rows[0]["trash_f1"] == 0.25 and rows[1]["f1_macro"] == 0.65


def test_export_best_picks_val_not_test(tmp_path):
    import json

    from src.report import export_best

    _fake_run(tmp_path, tmp_path, "high_val", "baseline_cnn", val_f1=0.8, test_f1=0.5)
    _fake_run(tmp_path, tmp_path, "high_test", "modified_cnn", val_f1=0.6, test_f1=0.9)
    assert export_best(["high_val", "high_test"], reports_dir=tmp_path, models_dir=tmp_path) == "high_val"
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config == {"arch": "baseline_cnn", "classes": CLASSES, "run_name": "high_val"}
    assert (tmp_path / "best.pth").read_bytes() == (tmp_path / "high_val.pth").read_bytes()


def test_export_best_keeps_best_from_earlier_session(tmp_path):
    """Runs from an earlier Colab session have history/exp JSON but no <run>.pth (not committed)."""
    import json

    from src.report import export_best

    _fake_run(tmp_path, tmp_path, "modified_cnn", "modified_cnn", val_f1=0.9, test_f1=0.8)
    _fake_run(tmp_path, tmp_path, "baseline_cnn_sgd", "baseline_cnn", val_f1=0.7, test_f1=0.7, optimizer="sgd")
    (tmp_path / "modified_cnn.pth").rename(tmp_path / "best.pth")
    config = {"arch": "modified_cnn", "classes": CLASSES, "run_name": "modified_cnn"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    before = (tmp_path / "best.pth").read_bytes()

    assert export_best(["modified_cnn", "baseline_cnn_sgd"], reports_dir=tmp_path, models_dir=tmp_path) == "modified_cnn"
    assert (tmp_path / "best.pth").read_bytes() == before
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == config


def test_export_best_missing_checkpoint_raises(tmp_path):
    import json

    from src.report import export_best

    _fake_run(tmp_path, tmp_path, "modified_cnn", "modified_cnn", val_f1=0.9, test_f1=0.8)
    _fake_run(tmp_path, tmp_path, "baseline_cnn", "baseline_cnn", val_f1=0.5, test_f1=0.5)
    (tmp_path / "modified_cnn.pth").unlink()
    (tmp_path / "baseline_cnn.pth").rename(tmp_path / "best.pth")
    (tmp_path / "config.json").write_text(json.dumps({"arch": "baseline_cnn", "classes": CLASSES,
                                                      "run_name": "baseline_cnn"}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="modified_cnn"):
        export_best(["modified_cnn", "baseline_cnn"], reports_dir=tmp_path, models_dir=tmp_path)


def test_results_markdown_has_both_tables(tmp_path):
    from src.report import build_summary, results_markdown

    _fake_run(tmp_path, tmp_path, "baseline_cnn", "baseline_cnn", 0.7, 0.6)
    _fake_run(tmp_path, tmp_path, "baseline_cnn_sgd", "baseline_cnn", 0.65, 0.55, optimizer="sgd", lr=1e-2)
    md = results_markdown(build_summary(reports_dir=tmp_path), ["baseline_cnn", "modified_cnn"],
                          ["baseline_cnn", "baseline_cnn_sgd"])
    assert "| Model | Run | Macro F1 | Accuracy | Params | Train time |" in md
    assert "| Optimizer | Run | lr | Macro F1 | Accuracy | Best epoch | Train time |" in md
    assert "| adam | baseline_cnn | 0.001 | 0.600 | 0.650 | 2 | 1.0 min |" in md
    assert "| sgd | baseline_cnn_sgd | 0.01 | 0.550 | 0.600 | 2 | 1.0 min |" in md
    assert "| modified_cnn | modified_cnn | — |" in md  # runs that haven't happened yet still get a row


def test_smoke_runs_are_flagged(tmp_path):
    import json

    from src.report import smoke_runs

    _fake_run(tmp_path, tmp_path, "baseline_cnn", "baseline_cnn", 0.7, 0.6)
    _fake_run(tmp_path, tmp_path, "modified_cnn", "modified_cnn", 0.7, 0.6)
    path = tmp_path / "history_modified_cnn.json"
    history = json.loads(path.read_text(encoding="utf-8"))
    history["hparams"]["max_batches"] = 2
    path.write_text(json.dumps(history), encoding="utf-8")
    assert smoke_runs(tmp_path) == ["modified_cnn"]


def test_update_readme_replaces_between_markers(tmp_path):
    from src.report import README_END, README_START, update_readme

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold table\n{README_END}\noutro\n", encoding="utf-8")
    update_readme("| new | table |\n", readme)
    assert readme.read_text(encoding="utf-8") == f"intro\n{README_START}\n| new | table |\n{README_END}\noutro\n"

    readme.write_text("no markers\n", encoding="utf-8")
    with pytest.raises(ValueError):
        update_readme("x", readme)


def test_readme_has_result_markers():
    from src.report import README_END, README_START

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.count(README_START) == 1 and text.count(README_END) == 1


def test_plot_loss_curves(tmp_path):
    from src.report import plot_loss_curves

    _fake_run(tmp_path, tmp_path, "baseline_cnn", "baseline_cnn", 0.6, 0.55)
    plot_loss_curves(["baseline_cnn"], tmp_path / "loss.png", reports_dir=tmp_path)
    assert (tmp_path / "loss.png").stat().st_size > 0


def test_compare_optimizers_requires_adam_run(tmp_path):
    from src.optimization import compare_optimizers

    with pytest.raises(FileNotFoundError, match="baseline_cnn"):
        compare_optimizers(reports_dir=tmp_path, models_dir=tmp_path)


@needs_data
def test_compare_optimizers_reuses_adam_run(tmp_path):
    import csv
    import json

    from src.optimization import compare_optimizers
    from src.train import train

    models_dir, reports_dir = tmp_path / "models", tmp_path / "reports"
    train("baseline_cnn", epochs=1, lr=1e-3, optimizer="adam", dropout=0.3, run_name="baseline_cnn", batch_size=4,
          max_batches=1, models_dir=models_dir, reports_dir=reports_dir)
    adam_history = reports_dir / "history_baseline_cnn.json"
    mtime = adam_history.stat().st_mtime_ns

    rows = compare_optimizers(reports_dir=reports_dir, models_dir=models_dir)

    assert adam_history.stat().st_mtime_ns == mtime
    sgd = json.loads((reports_dir / "history_baseline_cnn_sgd.json").read_text(encoding="utf-8"))["hparams"]
    assert (sgd["optimizer"], sgd["lr"], sgd["epochs"], sgd["dropout"], sgd["max_batches"]) == ("sgd", 0.01, 1, 0.3, 1)
    assert sgd["seed"] == json.loads(adam_history.read_text(encoding="utf-8"))["hparams"]["seed"]
    with open(reports_dir / "optimization.csv", newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert [(r["run_name"], r["optimizer"], float(r["lr"])) for r in csv_rows] == [
        ("baseline_cnn", "adam", 0.001), ("baseline_cnn_sgd", "sgd", 0.01)]
    assert len(rows) == 2
    assert (reports_dir / "optimization_curves.png").stat().st_size > 0

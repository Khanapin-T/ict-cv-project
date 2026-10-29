"""Checks on the logic inside notebooks/colab_train.ipynb that can run without Colab."""

from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]


def colab_cell(marker: str) -> str:
    nb = nbformat.read(ROOT / "notebooks" / "colab_train.ipynb", as_version=4)
    (source,) = [c.source for c in nb.cells if c.cell_type == "code" and marker in c.source]
    return source


def test_colab_refuses_quick_push():
    guard = colab_cell("QUICK and PUSH_RESULTS")
    with pytest.raises(ValueError):
        exec(guard, {"QUICK": True, "PUSH_RESULTS": True, "RUN_PARTS": {"A"}})
    exec(guard, {"QUICK": True, "PUSH_RESULTS": False, "RUN_PARTS": {"A"}})
    exec(guard, {"QUICK": False, "PUSH_RESULTS": True, "RUN_PARTS": {"A"}})


@pytest.mark.parametrize("parts", [set(), {"A", "B"}, {"A", "D"}, "AC"])
def test_colab_rejects_bad_run_parts(parts):
    with pytest.raises(ValueError, match="RUN_PARTS"):
        exec(colab_cell("QUICK and PUSH_RESULTS"), {"QUICK": False, "PUSH_RESULTS": False, "RUN_PARTS": parts})


@pytest.mark.parametrize("parts, runs, optimizer_calls", [
    ({"A", "C"}, ["baseline_cnn", "modified_cnn"], 1),
    ({"A"}, ["baseline_cnn", "modified_cnn"], 0),
    ({"C"}, [], 1),
])
def test_colab_runs_only_selected_parts(parts, runs, optimizer_calls):
    trained, compared = [], []
    ns = {
        "RUN_PARTS": parts, "MAX_BATCHES": None, "CNN_HP": {"seed": 42}, "ARCH_RUNS": ["baseline_cnn", "modified_cnn"],
        "train": lambda name, **kw: trained.append(kw["run_name"]),
        "compare_optimizers": lambda: compared.append(1) or [],
        "display": lambda *a: None, "pd": SimpleNamespace(DataFrame=lambda rows: rows), "Image": lambda path: path,
    }
    for marker in ['if "A" in RUN_PARTS', 'if "C" in RUN_PARTS']:
        exec(colab_cell(marker), ns)
    assert trained == runs
    assert len(compared) == optimizer_calls


def test_colab_runs_are_option_1_only():
    import re

    setup = colab_cell("CNN_HP =")
    ns = {}
    exec("\n".join(line for line in setup.splitlines() if re.match(r"^(MAX_BATCHES|SEED|CNN_HP|[A-Z_]+_RUNS) =", line)),
         {"QUICK": False}, ns)
    assert ns["ALL_RUNS"] == ["baseline_cnn", "modified_cnn", "baseline_cnn_sgd"]
    assert ns["CNN_HP"] == {"epochs": 30, "lr": 1e-3, "optimizer": "adam", "dropout": 0.3, "seed": 42}
    nb = nbformat.read(ROOT / "notebooks" / "colab_train.ipynb", as_version=4)
    source = "\n".join(c.source for c in nb.cells).lower()
    assert not any(word in source for word in ("mobilenet", "synthetic", "aug="))


def test_colab_unmet_pins():
    ns = {"REPO": "owner/repo"}
    exec(colab_cell("def unmet_pins"), ns)
    import importlib.metadata

    pytest_version = importlib.metadata.version("pytest")
    assert ns["unmet_pins"]([f"pytest=={pytest_version}", "pytest==0.0.1", "not-a-real-package-xyz==1.0", "# c", ""]) == [
        "pytest==0.0.1", "not-a-real-package-xyz==1.0"]

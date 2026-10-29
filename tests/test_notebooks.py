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


@pytest.mark.parametrize("parts", [set(), {"A", "D"}, "AC"])
def test_colab_rejects_bad_run_parts(parts):
    with pytest.raises(ValueError, match="RUN_PARTS"):
        exec(colab_cell("QUICK and PUSH_RESULTS"), {"QUICK": False, "PUSH_RESULTS": False, "RUN_PARTS": parts})


@pytest.mark.parametrize("parts, runs, optimizer_calls", [
    ({"A", "B", "C"}, ["baseline_cnn", "modified_cnn", "mobilenet_v2", "mobilenet_v2_aug", "mobilenet_v2_syn",
                       "mobilenet_v2_aug_syn"], 1),
    ({"A", "C"}, ["baseline_cnn", "modified_cnn", "mobilenet_v2"], 1),
    ({"B"}, ["mobilenet_v2_aug", "mobilenet_v2_syn", "mobilenet_v2_aug_syn"], 0),
])
def test_colab_runs_only_selected_parts(parts, runs, optimizer_calls):
    trained, compared = [], []
    ns = {
        "RUN_PARTS": parts, "MAX_BATCHES": None, "CNN_HP": {}, "MNV2_HP": {},
        "train": lambda name, **kw: trained.append(kw["run_name"]),
        "compare_optimizers": lambda: compared.append(1) or [],
        "display": lambda *a: None, "pd": SimpleNamespace(DataFrame=lambda rows: rows), "Image": lambda path: path,
    }
    for marker in ['if "A" in RUN_PARTS', 'if "B" in RUN_PARTS', 'if "C" in RUN_PARTS']:
        exec(colab_cell(marker), ns)
    assert trained == runs
    assert len(compared) == optimizer_calls


def test_colab_unmet_pins():
    ns = {"REPO": "owner/repo"}
    exec(colab_cell("def unmet_pins"), ns)
    import importlib.metadata

    pytest_version = importlib.metadata.version("pytest")
    assert ns["unmet_pins"]([f"pytest=={pytest_version}", "pytest==0.0.1", "not-a-real-package-xyz==1.0", "# c", ""]) == [
        "pytest==0.0.1", "not-a-real-package-xyz==1.0"]

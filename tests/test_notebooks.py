"""Checks on the logic inside notebooks/colab_train.ipynb that can run without Colab."""

from pathlib import Path

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
        exec(guard, {"QUICK": True, "PUSH_RESULTS": True})
    exec(guard, {"QUICK": True, "PUSH_RESULTS": False})
    exec(guard, {"QUICK": False, "PUSH_RESULTS": True})


def test_colab_unmet_pins():
    ns = {"REPO": "owner/repo"}
    exec(colab_cell("def unmet_pins"), ns)
    import importlib.metadata

    pytest_version = importlib.metadata.version("pytest")
    assert ns["unmet_pins"]([f"pytest=={pytest_version}", "pytest==0.0.1", "not-a-real-package-xyz==1.0", "# c", ""]) == [
        "pytest==0.0.1", "not-a-real-package-xyz==1.0"]

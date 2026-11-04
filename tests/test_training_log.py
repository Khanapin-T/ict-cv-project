"""The committed run reports are internally consistent (src/training_log.py checks)."""

from src import training_log


def test_training_log_checks_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(training_log, "OUT_PATH", tmp_path / "training_log.md")
    assert training_log.main() == 0
    text = (tmp_path / "training_log.md").read_text(encoding="utf-8")
    assert "❌" not in text and text.count("← best") == len(training_log.RUNS)

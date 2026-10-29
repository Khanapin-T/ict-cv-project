"""Demo photos for the app: 2 TEST-split images per class (never seen in training), copied from data/raw.

Then checks that the app's prediction code (src.inference) gives the same class as evaluate.py's batch
prediction for every demo image, using models/best.pth + models/config.json.

Example:
    python src/demo_images.py              # copy to demo_images/ and print predicted vs true
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `python src/demo_images.py` puts src/ on sys.path, not the repo root

import torch
from torch.utils.data import DataLoader

from src.dataset import CLASSES, EVAL_TRANSFORM, SPLITS_CSV, ImageListDataset
from src.evaluate import predict
from src.inference import load_image, predict_image
from src.models import get_model

DEMO_DIR = ROOT / "demo_images"
PER_CLASS = 2


def pick_test_images(per_class: int = PER_CLASS) -> list[tuple[Path, int]]:
    """The first `per_class` test rows of each class in splits.csv (deterministic, test split only)."""
    picked = {c: [] for c in CLASSES}
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "test" and len(picked[r["label"]]) < per_class:
                picked[r["label"]].append(ROOT / r["path"])
    return [(p, CLASSES.index(c)) for c in CLASSES for p in picked[c]]


def copy_demo_images(items: list[tuple[Path, int]], out_dir: Path = DEMO_DIR) -> list[tuple[Path, int]]:
    """Copy the images to out_dir/ (file names already start with the class, e.g. glass1.jpg)."""
    missing = [p for p, _ in items if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{missing[0]} is missing; run python src/prepare_split.py first")
    out_dir.mkdir(exist_ok=True)
    copied = []
    for path, label in items:
        shutil.copyfile(path, out_dir / path.name)
        copied.append((out_dir / path.name, label))
    return copied


def check_predictions(items: list[tuple[Path, int]], models_dir: Path = ROOT / "models") -> list[dict]:
    """Per image: true class, the app's prediction (+ confidence) and evaluate.py's prediction."""
    config = json.loads((models_dir / "config.json").read_text(encoding="utf-8"))
    model = get_model(config["arch"], num_classes=len(config["classes"]))
    model.load_state_dict(torch.load(models_dir / "best.pth", map_location="cpu", weights_only=True))
    model.eval()

    loader = DataLoader(ImageListDataset(items, EVAL_TRANSFORM), batch_size=len(items), shuffle=False)
    _, eval_pred, _ = predict(model, loader, torch.device("cpu"))
    rows = []
    for (path, label), batch_pred in zip(items, eval_pred):
        probs = predict_image(model, load_image(path))
        rows.append({"file": path.name, "true": CLASSES[label], "app": CLASSES[int(probs.argmax())],
                     "confidence": float(probs.max()), "evaluate": CLASSES[int(batch_pred)]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEMO_DIR)
    a = parser.parse_args()
    items = copy_demo_images(pick_test_images(), a.out_dir)
    rows = check_predictions(items)
    print(f"{'file':<18}{'true':<11}{'app':<11}{'conf':>6}  evaluate")
    for r in rows:
        mark = "" if r["app"] == r["true"] else "  <- wrong"
        print(f"{r['file']:<18}{r['true']:<11}{r['app']:<11}{r['confidence']:>6.1%}  {r['evaluate']}{mark}")
    correct = sum(r["app"] == r["true"] for r in rows)
    mismatches = [r["file"] for r in rows if r["app"] != r["evaluate"]]
    print(f"\n{correct}/{len(rows)} correct; app vs evaluate.py: "
          f"{'identical' if not mismatches else f'DIFFERENT for {mismatches}'}")
    if mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Data loaders over the fixed split in splits/splits.csv (contract: AGENTS.md).

Every split uses EVAL_TRANSFORM: the data stays fixed and only the model and its training settings change.
"""

import csv
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SPLITS_CSV = ROOT / "splits" / "splits.csv"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_MISSING_DATA_HINT = "run python src/prepare_split.py to download data/raw/ (it never re-splits)"


def read_split(split: str) -> list[tuple[Path, int]]:
    """(absolute image path, label index) for every row of `split` in splits.csv."""
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(f"{SPLITS_CSV} not found; {_MISSING_DATA_HINT}")
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        items = [(ROOT / r["path"], CLASSES.index(r["label"])) for r in csv.DictReader(f) if r["split"] == split]
    missing = [p for p, _ in items if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} images are missing (e.g. {missing[0]}); {_MISSING_DATA_HINT}")
    return items


class ImageListDataset(Dataset):
    """Images from a list of (path, label); any mode (RGBA, grayscale, ...) is converted to RGB."""

    def __init__(self, items: list[tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[i]
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        return self.transform(image), label


def get_dataloaders(batch_size: int, *, num_workers: int = 0, seed: int = 42
                    ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """(train, val, test) loaders. Only train is shuffled, with a generator seeded by `seed`."""
    train_ds = ImageListDataset(read_split("train"), EVAL_TRANSFORM)
    val_ds = ImageListDataset(read_split("val"), EVAL_TRANSFORM)
    test_ds = ImageListDataset(read_split("test"), EVAL_TRANSFORM)

    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available(),
              "persistent_workers": num_workers > 0}
    train = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(seed), **common)
    return train, DataLoader(val_ds, shuffle=False, **common), DataLoader(test_ds, shuffle=False, **common)

"""Download TrashNet, drop duplicate images and write the fixed train/val/test split.

Usage:
    python src/prepare_split.py          # download data/raw if missing; split only if splits.csv is missing
    python src/prepare_split.py --force  # rebuild splits.csv (don't: it is the single source of truth)

Outputs:
    data/raw/<class>/*.jpg       TrashNet images (not committed)
    splits/splits.csv            path,label,split (committed)
    reports/class_counts.csv     per-class raw / duplicates / kept / train / val / test counts
    reports/duplicates.csv       every duplicate group and which image was kept
"""

import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SPLITS_CSV = ROOT / "splits" / "splits.csv"
COUNTS_CSV = ROOT / "reports" / "class_counts.csv"
DUPLICATES_CSV = ROOT / "reports" / "duplicates.csv"

HF_REPO = "garythung/trashnet"
HF_REVISION = "94cd17ebfd6702bf62281d3c89f22ff649815b46"  # pinned so the data never changes under us
HF_FILE = "dataset-resized.zip"  # 512x384 images; dataset-original.zip holds the same photos at full size

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

HASH_SIZE = 16
HASH_THRESHOLD = 48
SEED = 42
SPLITS = ("train", "val", "test")


def download() -> None:
    """Extract dataset-resized.zip into data/raw/<class>/, keeping the original file names."""
    marker = RAW_DIR / ".complete"
    if marker.exists():
        return
    print(f"Downloading {HF_FILE} from huggingface.co/datasets/{HF_REPO} ...")
    zip_path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", revision=HF_REVISION)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = info.filename.split("/")
            # Entries look like dataset-resized/<class>/<file>; skip __MACOSX and dotfiles.
            if info.is_dir() or "__MACOSX" in parts or len(parts) < 2:
                continue
            label, name = parts[-2], parts[-1]
            if label not in CLASSES or name.startswith(".") or Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            dest = RAW_DIR / label / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    marker.touch()


def list_images() -> list[tuple[str, str]]:
    """Return sorted (repo-relative posix path, label) pairs for every raw image."""
    items = [
        (p.relative_to(ROOT).as_posix(), label)
        for label in CLASSES
        for p in (RAW_DIR / label).iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(items)


def find_duplicate_groups(paths: list[str], threshold: int) -> list[list[int]]:
    """Group images whose perceptual hashes differ by <= threshold bits (union-find)."""
    bits = []
    for path in paths:
        with Image.open(ROOT / path) as im:
            bits.append(imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten())
    hashes = np.packbits(np.array(bits), axis=1)

    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(paths) - 1):
        distances = np.bitwise_count(hashes[i] ^ hashes[i + 1 :]).sum(axis=1)
        for j in np.nonzero(distances <= threshold)[0] + i + 1:
            parent[find(int(j))] = find(i)

    groups: dict[int, list[int]] = {}
    for i in range(len(paths)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def dedupe(items: list[tuple[str, str]], threshold: int) -> tuple[list[tuple[str, str]], list[dict]]:
    """Keep the first image of each same-label duplicate group; drop whole groups with mixed labels."""
    groups = find_duplicate_groups([path for path, _ in items], threshold)
    dropped: set[int] = set()
    rows = []
    for group_id, group in enumerate(groups):
        mixed = len({items[i][1] for i in group}) > 1
        for rank, i in enumerate(sorted(group)):
            drop = mixed or rank > 0
            if drop:
                dropped.add(i)
            action = "dropped_mixed_labels" if mixed else ("dropped" if drop else "kept")
            rows.append({"group": group_id, "path": items[i][0], "label": items[i][1], "action": action})
    kept = [item for i, item in enumerate(items) if i not in dropped]
    return kept, rows


def split(items: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Stratified 70/15/15 split with a fixed seed."""
    labels = [label for _, label in items]
    train, rest = train_test_split(items, test_size=0.30, stratify=labels, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=[label for _, label in rest], random_state=SEED)
    rows = [(path, label, name) for name, part in zip(SPLITS, (train, val, test)) for path, label in part]
    return sorted(rows, key=lambda r: (SPLITS.index(r[2]), r[1], r[0]))


def write_csv(path: Path, header: list[str], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def verify_existing_split() -> None:
    with open(SPLITS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["path"] for r in rows if not (ROOT / r["path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} files listed in splits.csv are missing, e.g. {missing[0]}")
    counts = Counter(r["split"] for r in rows)
    print(f"{SPLITS_CSV.relative_to(ROOT)} exists and all {len(rows)} files are present "
          f"({', '.join(f'{s}={counts[s]}' for s in SPLITS)}). Not re-splitting.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing splits/splits.csv")
    parser.add_argument("--hash-threshold", type=int, default=HASH_THRESHOLD,
                        help=f"max phash Hamming distance (of {HASH_SIZE ** 2} bits) to treat two images as duplicates")
    args = parser.parse_args()

    download()
    if SPLITS_CSV.exists() and not args.force:
        verify_existing_split()
        return

    items = list_images()
    kept, duplicate_rows = dedupe(items, args.hash_threshold)
    rows = split(kept)

    write_csv(SPLITS_CSV, ["path", "label", "split"], rows)
    write_csv(DUPLICATES_CSV, ["group", "path", "label", "action"],
              [[r["group"], r["path"], r["label"], r["action"]] for r in duplicate_rows])

    raw = Counter(label for _, label in items)
    kept_counts = Counter(label for _, label in kept)
    per_split = Counter((label, name) for _, label, name in rows)
    counts = [
        [c, raw[c], raw[c] - kept_counts[c], kept_counts[c], *(per_split[c, s] for s in SPLITS)]
        for c in CLASSES
    ]
    header = ["class", "raw", "duplicates_removed", "kept", *SPLITS]
    write_csv(COUNTS_CSV, header, counts)

    for row in [header, *counts, ["total", *(sum(col) for col in list(zip(*counts))[1:])]]:
        print("".join(f"{v:>20}" for v in row))
    print(f"\nWrote {SPLITS_CSV.relative_to(ROOT)}, {COUNTS_CSV.relative_to(ROOT)}, "
          f"{DUPLICATES_CSV.relative_to(ROOT)} ({len({r['group'] for r in duplicate_rows})} duplicate groups)")


if __name__ == "__main__":
    main()

# SmartWaste AI

SmartWaste AI classifies photos of household waste into the 6 [TrashNet](https://huggingface.co/datasets/garythung/trashnet) classes
(cardboard, glass, metal, paper, plastic, trash). It compares CNNs trained from scratch with transfer learning from
**MobileNetV2**. It's a small step toward automated waste sorting.

## Goal

1. Train a **baseline CNN** and a **modified CNN** (the baseline plus one extra conv block), and fine-tune an
   ImageNet-pretrained **MobileNetV2**.
2. Measure the effect of **data augmentation**, **synthetic images** and training settings (optimizer, learning rate,
   dropout).
3. Compare all runs on the same fixed, deduplicated test split. **Macro F1 is the main metric.** Accuracy, macro
   precision/recall, per-class metrics and confusion matrices are reported too.
4. Serve the best model in a Streamlit demo, packaged with Docker.

## Team

| Member | Role | Owns |
|---|---|---|
| Participant 1 | Data & evaluation | `src/dataset.py`, `src/evaluate.py`, `notebooks/data_analysis.ipynb`, `data/extra/synthetic/` |
| Participant 2 | Models & training | `src/models.py`, `src/train.py`, `notebooks/colab_train.ipynb`, results table, final model |
| Tair | App & delivery | `src/optimization.py`, `app/`, `Dockerfile`, the rest of this README |

Shared rules (the fixed split, the train-only augmentation and synthetic data, the interface contract, and the
branch/PR workflow) are in [AGENTS.md](AGENTS.md). Read it before writing code.

## Dataset

TrashNet `dataset-resized` (512×384 photos on a white background) is downloaded from Hugging Face.
`src/prepare_split.py` drops perceptual-hash duplicates and makes a stratified 70/15/15 split with `seed=42`.
The result is committed in [splits/splits.csv](splits/splits.csv) and is never re-split.

| Class | Raw | Duplicates removed | Kept | Train | Val | Test |
|---|---:|---:|---:|---:|---:|---:|
| cardboard | 403 | 2 | 401 | 281 | 60 | 60 |
| glass | 501 | 4 | 497 | 348 | 74 | 75 |
| metal | 410 | 4 | 406 | 284 | 61 | 61 |
| paper | 594 | 23 | 571 | 399 | 86 | 86 |
| plastic | 482 | 2 | 480 | 336 | 72 | 72 |
| trash | 137 | 0 | 137 | 96 | 21 | 20 |
| **total** | **2527** | **35** | **2492** | **1744** | **374** | **374** |

The dataset is imbalanced: `trash` has about 4× fewer images than `paper`, so look at macro F1 and per-class recall,
not just accuracy.

Deduplication uses a 256-bit perceptual hash (`imagehash.phash`, `hash_size=16`) with a threshold of 48 bits, chosen
by inspecting candidate pairs by eye. It found 32 duplicate pairs, mostly re-saved or slightly shifted copies of the
same photo. In 29 pairs both copies had the same label, so one was kept. In 3 pairs the **same photo appears under
two classes** (glass/metal, glass/plastic ×2), so both copies were dropped as label noise. Per-class counts are in
[reports/class_counts.csv](reports/class_counts.csv), and every duplicate pair is in
[reports/duplicates.csv](reports/duplicates.csv).

Synthetic training images go in `data/extra/synthetic/<class>/` and are used **only for training**.

## Experiments

All runs use the same split and eval protocol. Metrics are on **test**, precision/recall/F1 are macro averages, and
**macro F1 is the main metric**. Each run writes `reports/history_<run>.json` (training curves, time, parameter count),
`reports/exp_<run>.json` (test metrics) and `reports/cm_<run>.png` (confusion matrix). Participant 2 maintains the
table below.

| Run | `run_name` | Augmentation | Synthetic | Accuracy | Precision | Recall | F1 |
|---|---|:---:|:---:|---:|---:|---:|---:|
| Original | `original` | – | – | | | | |
| + Augmentation | `aug` | ✓ | – | | | | |
| + Synthetic | `synthetic` | – | ✓ | | | | |
| + Both | `aug_synthetic` | ✓ | ✓ | | | | |

## Structure

```
AGENTS.md              shared rules for the team and coding agents (CLAUDE.md imports it)
src/prepare_split.py   download TrashNet, dedup, fixed split
src/dataset.py         get_dataloaders(batch_size, aug, synthetic)                   (Participant 1)
src/evaluate.py        evaluate() -> reports/exp_<run>.json, reports/cm_<run>.png    (Participant 1)
src/models.py          get_model(name): baseline_cnn, modified_cnn, mobilenet_v2     (Participant 2)
src/train.py           train() + CLI -> models/<run>.pth, reports/history_<run>.json (Participant 2)
src/optimization.py    (Tair)
notebooks/             data_analysis.ipynb (Participant 1), colab_train.ipynb (Participant 2)
splits/splits.csv      fixed train/val/test split (committed)
data/raw/              TrashNet images (not committed)
data/extra/synthetic/  synthetic train-only images per class (committed)
models/                best.pth + config.json: the final model used by the app (committed)
reports/               class counts, duplicates, training histories, test metrics
app/app.py             Streamlit demo
```

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data split

```bash
python src/prepare_split.py
```

On a fresh clone this downloads TrashNet into `data/raw/` and checks that every file listed in `splits/splits.csv`
exists. It **does not** re-split. `--force` rebuilds the split, but don't use it: the committed CSV is the source of
truth.

## Demo app

```bash
streamlit run app/app.py
```

Open http://localhost:8501 and upload a photo. The app shows the predicted class, its confidence and the top-3 classes.
The app builds the architecture named in `models/config.json` with `src.models.get_model` and loads `models/best.pth`.
If either file doesn't exist yet, it shows a "model not trained yet" notice instead. A replaced model is picked up on
page reload.

## Docker

```bash
docker build -t smartwaste-ai .
docker run --rm -p 8501:8501 smartwaste-ai
```

The image uses CPU-only PyTorch. Anything in `models/` at build time is baked into the image. To use a model trained
after the build without rebuilding, mount the folder instead:

```bash
docker run --rm -p 8501:8501 -v "$(pwd)/models:/app/models" smartwaste-ai
```

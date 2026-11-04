# SmartWaste AI

SmartWaste AI classifies photos of household waste into the 6 [TrashNet](https://huggingface.co/datasets/garythung/trashnet) classes
(cardboard, glass, metal, paper, plastic, trash) with CNNs trained from scratch. It's a small step toward automated
waste sorting.

**Live demo:** https://smartwaste-aiaiai.streamlit.app

**For the defence:** [for_defence/](for_defence/) has all results and training evidence with self-explaining file names.

## Goal

The project keeps the open dataset fixed and changes the ML algorithm:

1. **Layers:** a **baseline CNN** vs a **modified CNN** (the baseline plus one extra conv block), both with Adam.
2. **Optimizer:** the baseline CNN with **Adam** vs **SGD** (momentum 0.9).
3. Compare the 3 runs on the same fixed, deduplicated test split. **Macro F1 is the main metric.** Accuracy, macro
   precision/recall, per-class metrics and confusion matrices are reported too.
4. Serve the best model in a Streamlit demo, packaged with Docker.

## Team

| Member | Defence topic |
|---|---|
| Participant 1 | Data: dataset & splits, data analysis, evaluation |
| Participant 2 | Models & training: baseline/modified CNN, training loop, results |
| Tair | Optimization (Adam vs SGD), app, Docker |

Shared rules (the fixed split and the interface contract) are in
[AGENTS.md](AGENTS.md). Read it before changing code.

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

## Experiments

All runs use the same split and eval protocol. The best epoch of each run is picked on **val** macro F1; the metrics
below are on **test**. **Macro F1 is the main metric.** Each run writes `reports/history_<run>.json` (training curves,
time, parameter count, hyperparameters), `reports/exp_<run>.json` (test metrics) and `reports/cm_<run>.png` (confusion
matrix). All runs together are in `reports/summary.csv` and `reports/loss_curves.png`. The tables below are written by
`notebooks/colab_train.ipynb` (`update_readme`); "—" means the run hasn't happened yet.

**Training log (evidence).** [reports/training_log.md](reports/training_log.md) lists every epoch of the 3 runs (train
loss, val loss, val macro F1, best epoch, training time, parameter count) with consistency checks that tie the history,
the confusion matrix and the test metrics together. It is generated from the JSON files the Colab run saved
(`python src/training_log.py`, checked by `tests/test_training_log.py`). The run's outputs were committed unchanged in
`76872cc` on 2026-10-29.

Three runs, all on the same data (no augmentation, no extra images) with the same seed (42: weight init and batch
order), 30 epochs, batch size 32 and dropout 0.3:

| Run | Model | Optimizer | Part |
|---|---|---|---|
| `baseline_cnn` | baseline CNN (3 conv blocks) | Adam, lr 1e-3 | A and C |
| `modified_cnn` | baseline + 1 conv block (4) | Adam, lr 1e-3 | A |
| `baseline_cnn_sgd` | baseline CNN (3 conv blocks) | SGD momentum 0.9, lr 1e-2 | C |

Part A compares architectures (layers), part C compares optimizers.

<!-- results:start (written by src/report.py update_readme) -->
**A) Architectures** (Adam, test set; best epoch picked on val macro F1)

| Model | Run | Macro F1 | Accuracy | Params | Train time |
|---|---|---|---|---|---|
| baseline_cnn | baseline_cnn | 0.768 | 0.781 | 0.62M | 4.0 min |
| modified_cnn | modified_cnn | 0.800 | 0.813 | 1.44M | 4.1 min |

**C) Optimizers** (baseline_cnn, test set)

| Optimizer | Run | lr | Macro F1 | Accuracy | Best epoch | Train time |
|---|---|---|---|---|---|---|
| adam | baseline_cnn | 0.001 | 0.768 | 0.781 | 22 | 4.0 min |
| sgd | baseline_cnn_sgd | 0.01 | 0.792 | 0.810 | 25 | 4.0 min |
<!-- results:end -->

The run with the best **val** macro F1 of the three is exported as `models/best.pth` + `models/config.json`.

### Optimization

Adam vs SGD (momentum 0.9) on `baseline_cnn` (`src/optimization.py`). The Adam side is part A's `baseline_cnn` run,
reused and not retrained; the SGD side is `baseline_cnn_sgd`, trained with the same epochs, dropout, seed and data.
Adam uses lr 1e-3 and SGD (momentum 0.9) uses lr 1e-2. Each optimizer uses its own standard learning rate, because SGD needs a larger step than Adam. So the comparison is between the two optimizers as typically configured, not at an identical lr.

Results: [reports/optimization.csv](reports/optimization.csv) and `reports/optimization_curves.png` (train loss and
val macro F1 per epoch).

## How to train

**Colab (GPU), all runs:**

1. Create a fine-grained GitHub token for this repo with *Contents: Read* (*Read and write* to push results).
2. In Colab, open `notebooks/colab_train.ipynb`, add the token as the secret `GH_TOKEN` (key icon) and turn on
   *Notebook access*.
3. *Runtime → Change runtime type → GPU*, then *Run all*. Set `QUICK = True` first for a few-minute pipeline check.
4. The first run installs the pinned packages and restarts the runtime once; then *Run all* again.
5. The last cells write the tables into this README and download `results.zip` (`reports/`, `README.md`,
   `models/best.pth`, `models/config.json`). With `PUSH_RESULTS = True` (not allowed with `QUICK`) they are also
   committed and pushed to `main`.

**Running the parts separately** (e.g. A now, C later): set `RUN_PARTS = {"A"}`, later `RUN_PARTS = {"C"}`. The results cells always merge every run in `reports/`, so the summary, README tables and
best model cover all runs so far. A new Colab session is a fresh clone, so carry the earlier results over with
`PUSH_RESULTS = True`, or by uploading the earlier `results.zip` to `/content/` before *Run all*. Per-run checkpoints
are not kept between sessions; if the best run is an earlier one, its exported `models/best.pth` stays. Results of
`QUICK` runs are never pushed: the push cell refuses while `reports/` holds any run trained with `max_batches`.

**CLI, one run:**

```bash
python src/train.py --model baseline_cnn --epochs 30 --lr 1e-3 --dropout 0.3 --run-name baseline_cnn
python src/train.py --model modified_cnn --epochs 30 --lr 1e-3 --dropout 0.3 --run-name modified_cnn
python src/optimization.py                  # after the baseline_cnn run
python src/train.py --model baseline_cnn --epochs 1 --max-batches 2 --run-name smoke   # quick CPU check
```

**Tests** (CPU, a few batches; tests that need `data/raw/` are skipped without it):

```bash
pytest -q
```

## Structure

```
AGENTS.md              shared rules for the team and coding agents (CLAUDE.md imports it)
src/prepare_split.py   download TrashNet, dedup, fixed split
src/dataset.py         get_dataloaders(batch_size)
src/models.py          get_model(name): baseline_cnn, modified_cnn
src/train.py           train() + CLI -> models/<run>.pth, reports/history_<run>.json
src/evaluate.py        evaluate() -> reports/exp_<run>.json, reports/cm_<run>.png
src/optimization.py    Adam vs SGD on baseline_cnn -> reports/optimization.csv
src/report.py          summary.csv, loss curves, best-model export, README tables
notebooks/             colab_train.ipynb (all training runs), data_analysis.ipynb
tests/                 pytest contract and app tests
splits/splits.csv      fixed train/val/test split (committed)
data/raw/              TrashNet images (not committed)
models/                best.pth + config.json: the final model used by the app (committed)
reports/               class counts, duplicates, data figures, training histories, test metrics
app/app.py             Streamlit demo (app/requirements.txt: Streamlit Cloud deps)
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

`demo_images/` has 2 **test-split** photos per class (never seen in training) to upload in the demo.
`python src/demo_images.py` recreates them from `data/raw/` and prints the app's prediction next to the true class
and to `evaluate.py`'s batch prediction for the same file.

### Online (Streamlit Community Cloud)

Live at https://smartwaste-aiaiai.streamlit.app, deployed from this repo with entrypoint `app/app.py`, branch `main`
(the Cloud picked Python 3.14). Community Cloud installs
`app/requirements.txt` (it is next to the entrypoint, so it wins over the root file): only the app's dependencies, with
CPU-only torch/torchvision. Keep its pins in sync with the root `requirements.txt`. Every push to `main` redeploys.

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

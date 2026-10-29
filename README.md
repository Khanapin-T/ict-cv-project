# ICT CV Project

Image classification project comparing a **baseline model** against a **modified model and/or augmentation strategy**.

## Goal

1. Train a baseline image classifier and record its metrics.
2. Apply a modification (architecture change and/or data augmentation).
3. Compare both under the same data split and evaluation protocol (accuracy, precision/recall/F1, confusion matrix).
4. Serve the trained model through a small Streamlit demo app.

## Structure

```
data/        datasets (not tracked by git)
notebooks/   exploration and experiment notebooks
src/         training, evaluation and model code
app/         Streamlit demo app
Dockerfile   container setup (placeholder)
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the demo app

```bash
streamlit run app/app.py
```

# Project 2: Data Classification Using AI 📊

**DecodeLabs Industrial Training Kit — Batch 2026**

## Goal

Build a supervised learning classifier from scratch — no external ML libraries.
You'll load data, split it into training/testing sets, train a k-Nearest Neighbors
model, and evaluate its performance.

## How to Run

```bash
# Run the full pipeline (exploration → train → test → interactive mode)
python main.py
```

## What It Does

| Phase | Description |
|-------|-------------|
| **1. Dataset Exploration** | Summarises the Iris flower dataset — samples, features, class balance, value ranges |
| **2. Training & Evaluation** | Splits data 80/20, trains a k-NN (k=3), prints accuracy + confusion matrix + per-class metrics |
| **3. Hyperparameter Comparison** | Tries k=1,3,5,7,9,11,15 and shows which performs best |
| **4. Interactive Prediction** | Type in flower measurements — the classifier tells you the species with a confidence score |

## Files

| File | Purpose |
|------|---------|
| `main.py` | Full pipeline + the Iris dataset |
| `classifier.py` | k-NN implementation from scratch + evaluation helpers |
| `README.md` | This file |

## What I Built From Scratch

- **`KNNClassifier`** — k-Nearest Neighbors with configurable `k`
- **`euclidean_distance()`** — standard L2 distance between feature vectors
- **`train_test_split()`** — shuffles and splits data deterministically
- **`accuracy()`** — fraction of correct predictions
- **`confusion_matrix()`** — N×N matrix showing actual vs predicted
- **`predict_with_confidence()`** — returns confidence score based on neighbor vote strength

## Key Skills Practiced

| Skill | Where |
|-------|-------|
| **Supervised learning pipeline** | `main.py` — `run_pipeline()` |
| **k-NN algorithm** | `classifier.py` — distance computation → k nearest → majority vote |
| **Train/test split** | `classifier.py` — `train_test_split()` |
| **Model evaluation** | accuracy, confusion matrix, precision, recall, F1 |
| **Hyperparameter tuning** | `main.py` — `compare_k_values()` |
| **Feature vectors** | 4-dimensional real-valued features |
| **Decision boundaries** | Interactive mode shows low-confidence warnings |

---

*Complete this project to unlock Project 3!*

"""
Project 2: Data Classification Using AI 📊🤖
DecodeLabs Industrial Training Kit — Batch 2026

A complete supervised learning pipeline built from scratch:
  • k-Nearest Neighbors classifier (no external ML libs)
  • Train / test split
  • Accuracy evaluation + confusion matrix
  • Hyperparameter comparison (different k values)
  • Interactive mode — classify your own measurements

Key Skills: Data handling, supervised learning, model training & testing
"""

from classifier import (
    KNNClassifier,
    accuracy,
    confusion_matrix,
    print_confusion_matrix,
    train_test_split,
)

# ═══════════════════════════════════════════════════════════════
#  DATASET — Iris flower classification (simplified)
# ═══════════════════════════════════════════════════════════════
#  4 features: [sepal_length, sepal_width, petal_length, petal_width]
#  3 classes:   0 = Setosa, 1 = Versicolor, 2 = Virginica
#
#  Each class has 50 samples → 150 total samples.
#  This is a real subset of the classic Fisher / Anderson Iris dataset.

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]

IRIS_DATA: list[tuple[list[float], int]] = [
    # ── Setosa (0) ────────────────────────────────────────
    ([5.1, 3.5, 1.4, 0.2], 0),
    ([4.9, 3.0, 1.4, 0.2], 0),
    ([4.7, 3.2, 1.3, 0.2], 0),
    ([4.6, 3.1, 1.5, 0.2], 0),
    ([5.0, 3.6, 1.4, 0.2], 0),
    ([5.4, 3.9, 1.7, 0.4], 0),
    ([4.6, 3.4, 1.4, 0.3], 0),
    ([5.0, 3.4, 1.5, 0.2], 0),
    ([4.4, 2.9, 1.4, 0.2], 0),
    ([4.9, 3.1, 1.5, 0.1], 0),
    ([5.4, 3.7, 1.5, 0.2], 0),
    ([4.8, 3.4, 1.6, 0.2], 0),
    ([4.8, 3.0, 1.4, 0.1], 0),
    ([4.3, 3.0, 1.1, 0.1], 0),
    ([5.8, 4.0, 1.2, 0.2], 0),
    ([5.7, 4.4, 1.5, 0.4], 0),
    ([5.4, 3.9, 1.3, 0.4], 0),
    ([5.1, 3.5, 1.4, 0.3], 0),
    ([5.7, 3.8, 1.7, 0.3], 0),
    ([5.1, 3.8, 1.5, 0.3], 0),
    ([5.4, 3.4, 1.7, 0.2], 0),
    ([5.1, 3.7, 1.5, 0.4], 0),
    ([4.6, 3.6, 1.0, 0.2], 0),
    ([5.1, 3.3, 1.7, 0.5], 0),
    ([4.8, 3.4, 1.9, 0.2], 0),
    ([5.0, 3.0, 1.6, 0.2], 0),
    ([5.0, 3.4, 1.6, 0.4], 0),
    ([5.2, 3.5, 1.5, 0.2], 0),
    ([5.2, 3.4, 1.4, 0.2], 0),
    ([4.7, 3.2, 1.6, 0.2], 0),
    ([4.8, 3.1, 1.6, 0.2], 0),
    ([5.4, 3.4, 1.5, 0.4], 0),
    ([5.2, 4.1, 1.5, 0.1], 0),
    ([5.5, 4.2, 1.4, 0.2], 0),
    ([4.9, 3.1, 1.5, 0.2], 0),
    ([5.0, 3.2, 1.2, 0.2], 0),
    ([5.5, 3.5, 1.3, 0.2], 0),
    ([4.9, 3.6, 1.4, 0.1], 0),
    ([4.4, 3.0, 1.3, 0.2], 0),
    ([5.1, 3.4, 1.5, 0.2], 0),
    ([5.0, 3.5, 1.3, 0.3], 0),
    ([4.5, 2.3, 1.3, 0.3], 0),
    ([4.4, 3.2, 1.3, 0.2], 0),
    ([5.0, 3.5, 1.6, 0.6], 0),
    ([5.1, 3.8, 1.9, 0.4], 0),
    ([4.8, 3.0, 1.4, 0.3], 0),
    ([5.1, 3.8, 1.6, 0.2], 0),
    ([4.6, 3.2, 1.4, 0.2], 0),
    ([5.3, 3.7, 1.5, 0.2], 0),
    ([5.0, 3.3, 1.4, 0.2], 0),
    # ── Versicolor (1) ────────────────────────────────────
    ([7.0, 3.2, 4.7, 1.4], 1),
    ([6.4, 3.2, 4.5, 1.5], 1),
    ([6.9, 3.1, 4.9, 1.5], 1),
    ([5.5, 2.3, 4.0, 1.3], 1),
    ([6.5, 2.8, 4.6, 1.5], 1),
    ([5.7, 2.8, 4.5, 1.3], 1),
    ([6.3, 3.3, 4.7, 1.6], 1),
    ([4.9, 2.4, 3.3, 1.0], 1),
    ([6.6, 2.9, 4.6, 1.3], 1),
    ([5.2, 2.7, 3.9, 1.4], 1),
    ([5.0, 2.0, 3.5, 1.0], 1),
    ([5.9, 3.0, 4.2, 1.5], 1),
    ([6.0, 2.2, 4.0, 1.0], 1),
    ([6.1, 2.9, 4.7, 1.4], 1),
    ([5.6, 2.9, 3.6, 1.3], 1),
    ([6.7, 3.1, 4.4, 1.4], 1),
    ([5.6, 3.0, 4.5, 1.5], 1),
    ([5.8, 2.7, 4.1, 1.0], 1),
    ([6.2, 2.2, 4.5, 1.5], 1),
    ([5.6, 2.5, 3.9, 1.1], 1),
    ([5.9, 3.2, 4.8, 1.8], 1),
    ([6.1, 2.8, 4.0, 1.3], 1),
    ([6.3, 2.5, 4.9, 1.5], 1),
    ([6.1, 2.8, 4.7, 1.2], 1),
    ([6.4, 2.9, 4.3, 1.3], 1),
    ([6.6, 3.0, 4.4, 1.4], 1),
    ([6.8, 2.8, 4.8, 1.4], 1),
    ([6.7, 3.0, 5.0, 1.7], 1),
    ([6.0, 2.9, 4.5, 1.5], 1),
    ([5.7, 2.6, 3.5, 1.0], 1),
    ([5.5, 2.4, 3.8, 1.1], 1),
    ([5.5, 2.4, 3.7, 1.0], 1),
    ([5.8, 2.7, 3.9, 1.2], 1),
    ([6.0, 2.7, 5.1, 1.6], 1),
    ([5.4, 3.0, 4.5, 1.5], 1),
    ([6.0, 3.4, 4.5, 1.6], 1),
    ([6.7, 3.1, 4.7, 1.5], 1),
    ([6.3, 2.3, 4.4, 1.3], 1),
    ([5.6, 3.0, 4.1, 1.3], 1),
    ([5.5, 2.5, 4.0, 1.3], 1),
    ([5.5, 2.6, 4.4, 1.2], 1),
    ([6.1, 3.0, 4.6, 1.4], 1),
    ([5.8, 2.6, 4.0, 1.2], 1),
    ([5.0, 2.3, 3.3, 1.0], 1),
    ([5.6, 2.7, 4.2, 1.3], 1),
    ([5.7, 3.0, 4.2, 1.2], 1),
    ([5.7, 2.9, 4.2, 1.3], 1),
    ([6.2, 2.9, 4.3, 1.3], 1),
    ([5.1, 2.5, 3.0, 1.1], 1),
    ([5.7, 2.8, 4.1, 1.3], 1),
    # ── Virginica (2) ─────────────────────────────────────
    ([6.3, 3.3, 6.0, 2.5], 2),
    ([5.8, 2.7, 5.1, 1.9], 2),
    ([7.1, 3.0, 5.9, 2.1], 2),
    ([6.3, 2.9, 5.6, 1.8], 2),
    ([6.5, 3.0, 5.8, 2.2], 2),
    ([7.6, 3.0, 6.6, 2.1], 2),
    ([4.9, 2.5, 4.5, 1.7], 2),
    ([7.3, 2.9, 6.3, 1.8], 2),
    ([6.7, 2.5, 5.8, 1.8], 2),
    ([7.2, 3.6, 6.1, 2.5], 2),
    ([6.5, 3.2, 5.1, 2.0], 2),
    ([6.4, 2.7, 5.3, 1.9], 2),
    ([6.8, 3.0, 5.5, 2.1], 2),
    ([5.7, 2.5, 5.0, 2.0], 2),
    ([5.8, 2.8, 5.1, 2.4], 2),
    ([6.4, 3.2, 5.3, 2.3], 2),
    ([6.5, 3.0, 5.5, 1.8], 2),
    ([7.7, 3.8, 6.7, 2.2], 2),
    ([7.7, 2.6, 6.9, 2.3], 2),
    ([6.0, 2.2, 5.0, 1.5], 2),
    ([6.9, 3.2, 5.7, 2.3], 2),
    ([5.6, 2.8, 4.9, 2.0], 2),
    ([7.7, 2.8, 6.7, 2.0], 2),
    ([6.3, 2.7, 4.9, 1.8], 2),
    ([6.7, 3.3, 5.7, 2.1], 2),
    ([7.2, 3.2, 6.0, 1.8], 2),
    ([6.2, 2.8, 4.8, 1.8], 2),
    ([6.1, 3.0, 4.9, 1.8], 2),
    ([6.4, 2.8, 5.6, 2.1], 2),
    ([7.2, 3.0, 5.8, 1.6], 2),
    ([7.4, 2.8, 6.1, 1.9], 2),
    ([7.9, 3.8, 6.4, 2.0], 2),
    ([6.4, 2.8, 5.6, 2.2], 2),
    ([6.3, 2.8, 5.1, 1.5], 2),
    ([6.1, 2.6, 5.6, 1.4], 2),
    ([7.7, 3.0, 6.1, 2.3], 2),
    ([6.3, 3.4, 5.6, 2.4], 2),
    ([6.4, 3.1, 5.5, 1.8], 2),
    ([6.0, 3.0, 4.8, 1.8], 2),
    ([6.9, 3.1, 5.4, 2.1], 2),
    ([6.7, 3.1, 5.6, 2.4], 2),
    ([6.9, 3.1, 5.1, 2.3], 2),
    ([5.8, 2.7, 5.1, 1.9], 2),
    ([6.8, 3.2, 5.9, 2.3], 2),
    ([6.7, 3.3, 5.7, 2.5], 2),
    ([6.7, 3.0, 5.2, 2.3], 2),
    ([6.3, 2.5, 5.0, 1.9], 2),
    ([6.5, 3.0, 5.2, 2.0], 2),
    ([6.2, 3.4, 5.4, 2.3], 2),
    ([5.9, 3.0, 5.1, 1.8], 2),
]


def describe_dataset(data: list) -> None:
    """Print a summary of the dataset."""
    X = [row[0] for row in data]
    y = [row[1] for row in data]
    n = len(data)

    print(f"  Total samples:         {n}")
    print(f"  Features per sample:   {len(X[0])}")
    print(f"  Number of classes:     {len(set(y))}")
    for cid, cname in enumerate(CLASS_NAMES):
        count = sum(1 for label in y if label == cid)
        print(f"    * {cname:>12} (class {cid}):  {count} samples")

    # Per-class feature ranges
    print(f"\n  Feature ranges (min - max per class):")
    print(f"  {'Feature':<20} {'Setosa':>22} {'Versicolor':>22} {'Virginica':>22}")
    features_per_class = {i: [] for i in range(3)}
    for vec, label in data:
        features_per_class[label].append(vec)

    for fi, fname in enumerate(["sepal_length", "sepal_width", "petal_length", "petal_width"]):
        parts = []
        for cid in range(3):
            vals = [v[fi] for v in features_per_class[cid]]
            parts.append(f"{min(vals):4.1f} - {max(vals):4.1f}")
        print(f"  {fname:<20} {parts[0]:>22} {parts[1]:>22} {parts[2]:>22}")


def run_pipeline(k: int = 3, test_ratio: float = 0.2) -> dict:
    """
    Full supervised learning pipeline:
      Load → Split → Train → Predict → Evaluate

    Returns a dict of results.
    """
    X = [row[0] for row in IRIS_DATA]
    y = [row[1] for row in IRIS_DATA]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio)
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Testing samples:    {len(X_test)}")
    print()

    # Train
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, num_classes=3)

    print(f"  Accuracy (k={k}):     {acc:.2%}")
    print()
    print("  Confusion Matrix:")
    print_confusion_matrix(cm, CLASS_NAMES)
    print()

    # Per-class metrics
    print("  Per-class metrics:")
    for cid, cname in enumerate(CLASS_NAMES):
        total = sum(1 for actual in y_test if actual == cid)
        correct = cm[cid][cid]
        precision = correct / sum(cm[r][cid] for r in range(3)) if sum(cm[r][cid] for r in range(3)) > 0 else 0
        recall = correct / total if total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"    * {cname:<12}  precision={precision:.2%}  recall={recall:.2%}  f1={f1:.2%}  support={total}")

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def compare_k_values(k_values: list[int] = None) -> None:
    """Compare classifier accuracy across different k values."""
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 15]

    X = [row[0] for row in IRIS_DATA]
    y = [row[1] for row in IRIS_DATA]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    print(f"  {'k':>3}  |  {'Accuracy':>10}  |  {'Correct/Total':>15}")
    print(f"  {'-'*3}--+--{'-'*10}--+--{'-'*15}")

    best_k = None
    best_acc = 0.0

    for k in k_values:
        model = KNNClassifier(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy(y_test, y_pred)
        correct = sum(1 for t, p in zip(y_test, y_pred) if t == p)
        total = len(y_test)
        marker = " <-- BEST" if acc > best_acc else ""
        print(f"  {k:>3}  |  {acc:>8.2%}  |  {correct:>3}/{total:<3}{marker}")
        if acc > best_acc:
            best_k = k
            best_acc = acc

    print(f"\n  Best k: {best_k} ({best_acc:.2%} accuracy)")


def interactive_predict() -> None:
    """
    Let the user type in flower measurements and get a prediction.
    """
    X = [row[0] for row in IRIS_DATA]
    y = [row[1] for row in IRIS_DATA]

    model = KNNClassifier(k=5)
    model.fit(X, y)

    print("\n  Enter flower measurements and I'll tell you the species!")
    print("  (Press Ctrl+C or type 'quit' to stop)\n")

    while True:
        try:
            raw = input("  Measurements [sepal_len sepal_wid petal_len petal_wid]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        parts = raw.replace(",", " ").split()
        if len(parts) != 4:
            print("  [!] Need exactly 4 numbers -- sepal length, sepal width, petal length, petal width")
            continue

        try:
            features = [float(p) for p in parts]
        except ValueError:
            print("  [!] Couldn't parse numbers. Try: 5.1 3.5 1.4 0.2")
            continue

        result = model.predict_with_confidence([features])[0]
        prediction = result["prediction"]
        confidence = result["confidence"]
        species = CLASS_NAMES[prediction]

        print(f"  => Predicted species:  {species} (class {prediction})")
        print(f"    Confidence:         {confidence:.0%} (k={model.k})")

        if confidence < 0.6:
            print("    [!] Low confidence -- this sample is close to a decision boundary!")
        print()


def main() -> None:
    HR = "=" * 60
    print(HR)
    print("  Project 2: Data Classification Using AI")
    print("  DecodeLabs Industrial Training Kit 2026")
    print(HR)

    # Step 1: Explore the dataset
    print("\n[ PHASE 1: DATASET EXPLORATION ]")
    print("-" * 60)
    describe_dataset(IRIS_DATA)

    # Step 2: Run the pipeline
    print("\n[ PHASE 2: TRAINING & EVALUATION (k=3, 80/20 split) ]")
    print("-" * 60)
    run_pipeline(k=3)

    # Step 3: Compare hyperparameters
    print("[ PHASE 3: HYPERPARAMETER COMPARISON ]")
    print("-" * 60)
    compare_k_values()

    # Step 4: Interactive mode
    print("\n[ PHASE 4: INTERACTIVE PREDICTION ]")
    print("-" * 60)
    interactive_predict()

    print("\n" + HR)
    print("  Classification pipeline complete!")
    print(HR)


if __name__ == "__main__":
    main()

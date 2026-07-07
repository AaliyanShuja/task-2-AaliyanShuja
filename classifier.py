"""
k-Nearest Neighbors (k-NN) Classifier — from scratch.

No external ML libraries. Pure Python + math.
This implements the full supervised learning pipeline:

    1. Store all training samples
    2. For a new point, compute distance to every training sample
    3. Pick the k closest neighbors
    4. Majority vote → predicted class
"""

import math
from collections import Counter


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two feature vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class KNNClassifier:
    """
    A from-scratch k-Nearest Neighbors classifier.

    Usage:
        model = KNNClassifier(k=3)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._X: list[list[float]] = []
        self._y: list[int] = []

    # ─── Training ────────────────────────────────────────────

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        """
        "Train" by memorising the dataset.
        k-NN is lazy — there is no real training step.
        """
        if len(features) != len(labels):
            raise ValueError("Number of feature vectors and labels must match")
        if len(features) == 0:
            raise ValueError("Cannot train on empty data")
        self._X = features
        self._y = labels

    # ─── Prediction ──────────────────────────────────────────

    def predict(self, features: list[list[float]]) -> list[int]:
        """Predict labels for one or more samples."""
        return [self._predict_single(x) for x in features]

    def _predict_single(self, x: list[float]) -> int:
        """Predict the label of a single feature vector."""

        # 1. Compute distance from x to every training sample
        distances = [
            (euclidean_distance(x, x_train), label)
            for x_train, label in zip(self._X, self._y)
        ]

        # 2. Sort by distance (ascending) and take k closest
        neighbors = sorted(distances, key=lambda pair: pair[0])[: self.k]

        # 3. Majority vote among the k neighbors
        votes = [label for _, label in neighbors]
        most_common = Counter(votes).most_common(1)

        return most_common[0][0]

    # ─── Confidence score ────────────────────────────────────

    def predict_with_confidence(
        self, features: list[list[float]]
    ) -> list[dict]:
        """
        Predict with a confidence score (0.0 – 1.0).

        Returns a list of dicts: {"prediction": int, "confidence": float}
        """
        return [self._predict_single_confidence(x) for x in features]

    def _predict_single_confidence(self, x: list[float]) -> dict:
        distances = [
            (euclidean_distance(x, x_train), label)
            for x_train, label in zip(self._X, self._y)
        ]
        neighbors = sorted(distances, key=lambda pair: pair[0])[: self.k]
        votes = [label for _, label in neighbors]
        tally = Counter(votes)
        winner, count = tally.most_common(1)[0]
        return {"prediction": winner, "confidence": count / self.k}


# ─── Evaluation helpers ──────────────────────────────────────

def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Fraction of correct predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between true and predicted labels")
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> list[list[int]]:
    """
    Build an N×N confusion matrix.

    Rows   = actual class
    Columns = predicted class

    Entry [i][j] = number of samples from class i predicted as class j.
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual][predicted] += 1
    return matrix


def print_confusion_matrix(matrix: list[list[int]], class_names: list[str]) -> None:
    """Pretty-print a confusion matrix."""
    n = len(matrix)
    # Column widths
    name_width = max(len(name) for name in class_names) + 2
    col_width = max(len(str(matrix[i][j])) for i in range(n) for j in range(n)) + 2

    # Header
    header = " " * (name_width + 2)
    for name in class_names:
        header += f"{name:>{col_width}}"
    print(header)

    # Rows
    for i, name in enumerate(class_names):
        row = f"  {name:>{name_width}}"
        for j in range(n):
            row += f"{matrix[i][j]:>{col_width}}"
        print(row)


def train_test_split(
    X: list[list[float]],
    y: list[int],
    test_ratio: float = 0.2,
    seed: int | None = 42,
) -> tuple:
    """
    Split data into training and testing sets.

    Returns: (X_train, X_test, y_train, y_test)
    """
    if len(X) != len(y):
        raise ValueError("Features and labels length mismatch")
    if not (0 < test_ratio < 1):
        raise ValueError("test_ratio must be between 0 and 1")

    # Pair and shuffle deterministically
    paired = list(zip(X, y))
    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(paired)
    else:
        import random

        random.shuffle(paired)

    split_idx = int(len(paired) * (1 - test_ratio))
    train = paired[:split_idx]
    test = paired[split_idx:]

    X_train = [p[0] for p in train]
    y_train = [p[1] for p in train]
    X_test = [p[0] for p in test]
    y_test = [p[1] for p in test]

    return X_train, X_test, y_train, y_test

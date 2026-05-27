import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any,
    path: str | Path,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)


def write_timestamped_json(
    data: Any,
    output_dir: str | Path = "output",
    prefix: str = "result",
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path(output_dir) / f"{prefix}_{timestamp}.json"
    write_json(data, path)
    return path


def quadratic_weighted_kappa(y_true, y_pred, num_classes=4):
    """
    Compute Quadratic Weighted Kappa (QWK)

    Parameters
    ----------
    y_true : list or np.array
        Ground truth labels (integer labels)
    y_pred : list or np.array
        Predicted labels (integer labels)
    num_classes : int (optional)
        Number of classes. If None, inferred from data.

    Returns
    -------
    kappa : float
        Quadratic weighted kappa score
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    assert y_true.shape == y_pred.shape

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    # 1. Confusion matrix (O)
    O = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    # 2. Expected matrix (E)
    hist_true = np.bincount(y_true, minlength=num_classes)
    hist_pred = np.bincount(y_pred, minlength=num_classes)

    E = np.outer(hist_true, hist_pred) / len(y_true)

    # 3. Weight matrix (W)
    W = np.zeros((num_classes, num_classes), dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # 4. QWK
    numerator = np.sum(W * O)
    denominator = np.sum(W * E)

    return 1.0 - numerator / denominator

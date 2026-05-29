import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)


def _threshold_metric(y_true, y_pred, threshold_class, scorer):
    true_binary = y_true >= threshold_class
    pred_binary = y_pred >= threshold_class
    if true_binary.sum() == 0:
        return np.nan
    return float(scorer(true_binary, pred_binary, zero_division=0))


def compute_spike_metrics(y_true, y_pred, y_proba=None) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present_labels = sorted(np.unique(y_true).tolist())
    balanced_accuracy = recall_score(
        y_true,
        y_pred,
        labels=present_labels,
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
        ),
        "mean_absolute_class_error": float(np.mean(np.abs(y_true - y_pred))),
        "n": int(len(y_true)),
    }

    for name, threshold_class in {
        "gte_5pct": 1,
        "gte_10pct": 2,
        "gte_20pct": 3,
    }.items():
        metrics[f"recall_{name}"] = _threshold_metric(
            y_true, y_pred, threshold_class, recall_score
        )
        metrics[f"precision_{name}"] = _threshold_metric(
            y_true, y_pred, threshold_class, precision_score
        )
        metrics[f"f1_{name}"] = _threshold_metric(
            y_true, y_pred, threshold_class, f1_score
        )

    if y_proba is not None:
        metrics["multiclass_log_loss"] = float(
            log_loss(y_true, y_proba, labels=[0, 1, 2, 3])
        )

    return metrics

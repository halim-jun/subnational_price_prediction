import math
import warnings

from src.model.spike.metrics import compute_spike_metrics


def test_compute_spike_metrics_includes_severity_and_threshold_recalls():
    metrics = compute_spike_metrics(
        y_true=[0, 1, 2, 3, 3],
        y_pred=[0, 1, 1, 2, 3],
        y_proba=[
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.80, 0.07, 0.03],
            [0.20, 0.50, 0.25, 0.05],
            [0.05, 0.10, 0.70, 0.15],
            [0.02, 0.03, 0.15, 0.80],
        ],
    )

    assert math.isclose(metrics["macro_f1"], 7 / 12)
    assert math.isclose(metrics["mean_absolute_class_error"], 0.4)
    assert math.isclose(metrics["recall_gte_5pct"], 1.0)
    assert math.isclose(metrics["recall_gte_10pct"], 2 / 3)
    assert math.isclose(metrics["recall_gte_20pct"], 1 / 2)
    assert metrics["n"] == 5


def test_compute_spike_metrics_does_not_warn_for_single_class_groups():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = compute_spike_metrics(y_true=[0, 0], y_pred=[0, 1])

    assert caught == []
    assert metrics["n"] == 2

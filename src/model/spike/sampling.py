import numpy as np
from sklearn.utils.class_weight import compute_sample_weight


def balanced_sample_weight(y, max_weight: float | None = 10.0):
    """Return inverse-frequency sample weights, optionally capped."""
    weights = compute_sample_weight(class_weight="balanced", y=y)
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
    return weights

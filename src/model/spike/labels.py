import numpy as np
import pandas as pd

from src.model.spike.config import SPIKE_THRESHOLDS


def add_spike_labels(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    group_col: str = "admin2",
) -> pd.DataFrame:
    """Add horizon-aware month-over-month spike severity labels.

    The returned rows are base dates only. Rows without a future price at the
    requested horizon are dropped because their label cannot be observed.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    required = {group_col, "year", "month", target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    labeled = df.sort_values([group_col, "year", "month"]).copy()
    base_date = pd.to_datetime(
        {
            "year": labeled["year"].astype(int),
            "month": labeled["month"].astype(int),
            "day": 1,
        }
    )
    labeled["base_date"] = base_date
    labeled["label_date"] = base_date + pd.DateOffset(months=horizon)
    future_prices = labeled[[group_col, "base_date", target]].rename(
        columns={"base_date": "label_date", target: "_future_price"}
    )
    labeled = labeled.merge(
        future_prices,
        on=[group_col, "label_date"],
        how="left",
        validate="many_to_one",
    )

    pct_col = f"{target}_spike_pct_change_h{horizon}"
    class_col = f"{target}_spike_class_h{horizon}"
    labeled[pct_col] = labeled["_future_price"] / labeled[target] - 1

    thresholds = list(SPIKE_THRESHOLDS)
    conditions = [
        labeled[pct_col] < thresholds[0],
        labeled[pct_col] < thresholds[1],
        labeled[pct_col] < thresholds[2],
    ]
    labeled[class_col] = np.select(conditions, [0, 1, 2], default=3).astype(int)
    labeled["label_year"] = labeled["label_date"].dt.year.astype(int)
    labeled["label_month"] = labeled["label_date"].dt.month.astype(int)

    return labeled.dropna(subset=["_future_price", pct_col]).drop(
        columns=["_future_price"]
    )

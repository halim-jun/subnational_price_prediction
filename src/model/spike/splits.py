import pandas as pd


def _ensure_date_columns(df: pd.DataFrame) -> None:
    if "base_date" not in df.columns:
        df["base_date"] = pd.to_datetime(
            {
                "year": df["year"].astype(int),
                "month": df["month"].astype(int),
                "day": 1,
            }
        )
    if "label_date" not in df.columns:
        df["label_date"] = pd.to_datetime(
            {
                "year": df["label_year"].astype(int),
                "month": df["label_month"].astype(int),
                "day": 1,
            }
        )


def build_spike_stcv_masks(
    df: pd.DataFrame,
    spatial_fold: dict,
    temporal_fold: dict,
    admin_col: str = "admin2",
    label_encoders: dict | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Build leakage-safe spike STCV masks.

    Training rows are allowed only when their label date is before the
    validation start date, preventing future validation prices from entering
    train labels.
    """
    _ensure_date_columns(df)
    validation_start = pd.Timestamp(
        year=int(temporal_fold["cutoff_year"]), month=1, day=1
    )
    train_admin2 = spatial_fold["train_admin2"]
    val_admin2 = spatial_fold["val_admin2"]
    if label_encoders is not None and admin_col in label_encoders:
        encoder = label_encoders[admin_col]
        train_admin2 = set(
            encoder.transform([a for a in train_admin2 if a in encoder.classes_])
        )
        val_admin2 = set(
            encoder.transform([a for a in val_admin2 if a in encoder.classes_])
        )

    train_mask = (
        df[admin_col].isin(train_admin2)
        & (df["base_date"] < validation_start)
        & (df["label_date"] < validation_start)
    )
    val_mask = (
        df[admin_col].isin(val_admin2)
        & (df["base_date"] >= validation_start)
    )
    return train_mask, val_mask

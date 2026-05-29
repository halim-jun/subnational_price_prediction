import pandas as pd

from src.model.spike.labels import add_spike_labels


def prepare_spike_frame(
    df_base: pd.DataFrame,
    target: str,
    horizon: int,
    prepare_for_run_fn=None,
) -> tuple[pd.DataFrame, list[str], str, str]:
    """Reuse the existing horizon-safe feature pipeline and add spike labels."""
    if prepare_for_run_fn is None:
        from src.model import train_model_stcv as base_model

        prepare_for_run_fn = base_model.prepare_for_run

    df_prepared, features, _ = prepare_for_run_fn(df_base, target, horizon)
    df_spike = add_spike_labels(df_prepared, target=target, horizon=horizon)
    label_col = f"{target}_spike_class_h{horizon}"
    pct_col = f"{target}_spike_pct_change_h{horizon}"
    required_cols = features + [label_col, pct_col, "base_date", "label_date"]
    df_spike = df_spike.dropna(subset=required_cols)
    return df_spike, features, label_col, pct_col

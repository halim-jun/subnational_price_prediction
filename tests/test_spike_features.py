import pandas as pd

from src.model.spike.features import prepare_spike_frame


def test_prepare_spike_frame_reuses_existing_horizon_features():
    df = pd.DataFrame(
        {
            "admin2": ["A", "A", "A", "A"],
            "year": [2024, 2024, 2024, 2024],
            "month": [1, 2, 3, 4],
            "price": [100.0, 106.0, 120.0, 118.0],
            "existing_feature": [1.0, 2.0, 3.0, 4.0],
        }
    )

    def prepare_for_run(df_base, target, horizon):
        return df_base.copy(), ["existing_feature"], []

    prepared, features, label_col, pct_col = prepare_spike_frame(
        df,
        target="price",
        horizon=1,
        prepare_for_run_fn=prepare_for_run,
    )

    assert features == ["existing_feature"]
    assert label_col == "price_spike_class_h1"
    assert pct_col == "price_spike_pct_change_h1"
    assert prepared[label_col].tolist() == [1, 2, 0]

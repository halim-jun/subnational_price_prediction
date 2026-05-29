import pandas as pd

from src.model.spike.labels import add_spike_labels


def test_add_spike_labels_uses_future_price_at_horizon_per_admin():
    df = pd.DataFrame(
        {
            "admin2": ["A", "A", "A", "A", "B", "B"],
            "year": [2024, 2024, 2024, 2024, 2024, 2024],
            "month": [1, 2, 3, 4, 1, 2],
            "price": [100.0, 104.0, 111.0, 140.0, 50.0, 60.0],
        }
    )

    labeled = add_spike_labels(df, target="price", horizon=1)

    admin_a = labeled[labeled["admin2"] == "A"].sort_values("month")
    assert admin_a["price_spike_pct_change_h1"].round(2).tolist() == [
        0.04,
        0.07,
        0.26,
    ]
    assert admin_a["price_spike_class_h1"].tolist() == [0, 1, 3]
    assert admin_a["label_year"].tolist() == [2024, 2024, 2024]
    assert admin_a["label_month"].tolist() == [2, 3, 4]


def test_add_spike_labels_drops_rows_without_future_price():
    df = pd.DataFrame(
        {
            "admin2": ["A", "A"],
            "year": [2024, 2024],
            "month": [1, 2],
            "price": [100.0, 110.0],
        }
    )

    labeled = add_spike_labels(df, target="price", horizon=2)

    assert labeled.empty


def test_add_spike_labels_requires_exact_future_month_match():
    df = pd.DataFrame(
        {
            "admin2": ["A", "A"],
            "year": [2024, 2024],
            "month": [1, 3],
            "price": [100.0, 130.0],
        }
    )

    labeled = add_spike_labels(df, target="price", horizon=1)

    assert labeled.empty

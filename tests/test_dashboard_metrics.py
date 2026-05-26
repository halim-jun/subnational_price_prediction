import math

import pandas as pd

from src.dashboard.api.data_loader import build_country_metric_summary


def test_build_country_metric_summary_returns_one_row_per_country():
    predictions = pd.DataFrame(
        [
            {
                "target": "c_maize_fao",
                "horizon": 1,
                "country_name": "KEN",
                "actual": 10.0,
                "predicted": 12.0,
            },
            {
                "target": "c_maize_fao",
                "horizon": 1,
                "country_name": "KEN",
                "actual": 20.0,
                "predicted": 19.0,
            },
            {
                "target": "c_maize_fao",
                "horizon": 1,
                "country_name": "SOM",
                "actual": 30.0,
                "predicted": 33.0,
            },
            {
                "target": "c_maize_fao",
                "horizon": 1,
                "country_name": "SOM",
                "actual": 40.0,
                "predicted": 36.0,
            },
        ]
    )

    summary = build_country_metric_summary(predictions)

    assert list(summary["country_name"]) == ["KEN", "SOM"]
    assert set(summary.columns) >= {
        "target",
        "horizon",
        "country_name",
        "rmse",
        "mae",
        "mape",
        "r2",
        "n",
    }

    kenya = summary[summary["country_name"] == "KEN"].iloc[0]
    assert kenya["n"] == 2
    assert math.isclose(kenya["rmse"], math.sqrt((4 + 1) / 2))
    assert math.isclose(kenya["mae"], 1.5)
    assert math.isclose(kenya["mape"], ((2 / 10) + (1 / 20)) / 2)

import pandas as pd

from src.model.spike.splits import build_spike_stcv_masks


def test_build_spike_stcv_masks_purges_train_rows_whose_label_enters_validation():
    df = pd.DataFrame(
        {
            "admin2": ["train_admin"] * 4 + ["val_admin"] * 2,
            "year": [2023, 2023, 2023, 2024, 2024, 2024],
            "month": [9, 10, 11, 1, 1, 2],
            "label_year": [2023, 2024, 2024, 2024, 2024, 2024],
            "label_month": [12, 1, 2, 4, 2, 3],
        }
    )
    spatial_fold = {
        "train_admin2": {"train_admin"},
        "val_admin2": {"val_admin"},
    }
    temporal_fold = {"cutoff_year": 2024}

    train_mask, val_mask = build_spike_stcv_masks(
        df,
        spatial_fold=spatial_fold,
        temporal_fold=temporal_fold,
    )

    assert df.loc[train_mask, ["year", "month"]].to_dict("records") == [
        {"year": 2023, "month": 9}
    ]
    assert df.loc[val_mask, "admin2"].tolist() == ["val_admin", "val_admin"]
    assert df.loc[train_mask, "label_date"].max() < df.loc[val_mask, "base_date"].min()


def test_build_spike_stcv_masks_accepts_encoded_admin_values():
    class FakeEncoder:
        classes_ = ["train_admin", "val_admin"]

        def transform(self, values):
            mapping = {"train_admin": 0, "val_admin": 1}
            return [mapping[value] for value in values]

    df = pd.DataFrame(
        {
            "admin2": [0, 1],
            "year": [2023, 2024],
            "month": [9, 1],
            "label_year": [2023, 2024],
            "label_month": [12, 2],
        }
    )
    spatial_fold = {
        "train_admin2": {"train_admin"},
        "val_admin2": {"val_admin"},
    }

    train_mask, val_mask = build_spike_stcv_masks(
        df,
        spatial_fold=spatial_fold,
        temporal_fold={"cutoff_year": 2024},
        label_encoders={"admin2": FakeEncoder()},
    )

    assert train_mask.tolist() == [True, False]
    assert val_mask.tolist() == [False, True]

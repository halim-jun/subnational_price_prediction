"""
Subnational Merge Runner (v2 — KEN + SOM)

Changes from v1:
  - Target: KEN + SOM only (ETH has no price data in source CSV)
  - Price cols: c_maize_fao, c_food_price_index, c_sorghum
  - Per-country spatial join (prevents cross-boundary misassignment)

Usage:
  python run_subnational_merge.py
"""

import pandas as pd
import sys
import os
import glob
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.data_pipeline.merge import subnational_merger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ISO3_LIST = ['KEN', 'SOM']


def load_price_data(data_dir):
    logger.info("Loading Price Data...")
    csv_path = os.path.join(data_dir, "worldbank_imputed_price_data/WLD_RTFP_mkt_2026-01-13.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Price file not found: {csv_path}")
        return pd.DataFrame()
    price = pd.read_csv(csv_path, low_memory=False)
    price = price[price['ISO3'].isin(ISO3_LIST)]
    logger.info(f"Price data: {len(price)} rows, countries: {price['ISO3'].unique().tolist()}")
    return price


def load_crop_data(data_dir):
    logger.info("Loading Crop Data...")
    agg_path = os.path.join(data_dir, "crop_mask/admin_mapped/admin_agg.parquet")
    if os.path.exists(agg_path):
        logger.info("Loading pre-aggregated crop data.")
        return pd.read_parquet(agg_path)

    files = glob.glob(os.path.join(data_dir, "crop_mask/admin_mapped/*.parquet"))
    if not files:
        logger.warning("No crop mask parquet files found.")
        return pd.DataFrame(columns=['shapeName_ADM2', 'value'])

    logger.info(f"Aggregating {len(files)} crop mask files...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")

    if not dfs:
        return pd.DataFrame(columns=['shapeName_ADM2', 'value'])

    df = pd.concat(dfs, ignore_index=True)
    df_matched = df.dropna(subset=['shapeID_ADM2'])
    admin_agg = df_matched.groupby(
        ['shapeISO_ADM0', 'shapeName_ADM1', 'shapeName_ADM2']
    )['value'].mean().reset_index()
    return admin_agg


def load_acled_data(data_dir):
    logger.info("Loading ACLED Data...")
    acled_path = os.path.join(data_dir, "raw/acled/Africa_aggregated_data_up_to-2026-01-03.xlsx")
    if not os.path.exists(acled_path):
        logger.error(f"ACLED file not found: {acled_path}")
        return pd.DataFrame()
    acled = pd.read_excel(acled_path)
    target_countries = ['Kenya', 'Somalia']
    acled = acled[acled['COUNTRY'].isin(target_countries)]
    logger.info(f"ACLED filtered to {len(acled)} rows")
    return acled


def main():
    data_dir = os.path.join(project_root, 'data')

    price_df = load_price_data(data_dir)
    crop_df = load_crop_data(data_dir)
    acled_df = load_acled_data(data_dir)

    logger.info("Starting Merge Process...")
    merged_df = subnational_merger.merge_datasets(
        price_df=price_df,
        crop_df=crop_df,
        acled_df=acled_df,
        data_dir=data_dir,
        iso3_list=ISO3_LIST,
    )

    output_path = os.path.join(data_dir, "processed/subnational_merged_v2_KEN_SOM.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_parquet(output_path, index=False)

    logger.info(f"Merge Complete! Saved to {output_path}")
    logger.info(f"Shape: {merged_df.shape}")
    logger.info(f"Columns: {list(merged_df.columns)}")
    print(merged_df.describe())


if __name__ == "__main__":
    main()

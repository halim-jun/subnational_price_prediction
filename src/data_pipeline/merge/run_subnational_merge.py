
import pandas as pd
import sys
import os
import glob
import logging

# Add project root to path
# This script is in src/data_pipeline/merge/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.data_pipeline.merge import subnational_merger

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_price_data(data_dir):
    logger.info("Loading Price Data...")
    csv_path = os.path.join(data_dir, "worldbank_imputed_price_data/WLD_RTFP_mkt_2026-01-13.csv")
    if not os.path.exists(csv_path):
        logger.error(f"Price file not found: {csv_path}")
        return pd.DataFrame()
        
    price = pd.read_csv(csv_path)
    target_countries = ['Kenya', 'Somalia']  # Ethiopia excluded — no price data
    
    return price[price['country'].isin(target_countries)]



def load_crop_data(data_dir):
    logger.info("Loading Crop Data...")
    # Load from parquet if available
    agg_path = os.path.join(data_dir, "crop_mask/admin_mapped/admin_agg.parquet")
    if os.path.exists(agg_path):
        logger.info("Loading pre-aggregated crop data.")
        return pd.read_parquet(agg_path)
    
    # Else verify the notebook logic - aggregating mapped files
    files = glob.glob(os.path.join(data_dir, "crop_mask/admin_mapped/*.parquet"))
    if not files:
        logger.warning("No crop mask parquet files found.")
        return pd.DataFrame(columns=['shapeName_ADM2', 'value'])
        
    logger.info(f"Aggregating {len(files)} crop mask files...")
    # Use list comprehension for better memory if possible, but concat is fine for now
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
    
    # Aggregation
    admin_agg = df_matched.groupby(['shapeISO_ADM0', 'shapeName_ADM1', 'shapeName_ADM2'])['value'].mean().reset_index()
    return admin_agg

def load_acled_data(data_dir):
    logger.info("Loading ACLED Data...")
    acled_path = os.path.join(data_dir, "raw/acled/Africa_aggregated_data_up_to-2026-01-03.xlsx")
    if not os.path.exists(acled_path):
        logger.error(f"ACLED file not found: {acled_path}")
        return pd.DataFrame()
    
    acled = pd.read_excel(acled_path)
    target_countries = ['Kenya', 'Ethiopia', 'Somalia']
    acled = acled[acled['COUNTRY'].isin(target_countries)]
    logger.info(f"ACLED filtered to {len(acled)} rows for {target_countries}")
    return acled

def main():
    data_dir = os.path.join(project_root, 'data')
    
    # Load Datasets
    price_df = load_price_data(data_dir)
    # Population loaded internally via WorldPop rasters in merge_datasets
    crop_df = load_crop_data(data_dir)
    acled_df = load_acled_data(data_dir)
    
    # Run Merge
    logger.info("Starting Merge Process...")
    merged_df = subnational_merger.merge_datasets(
        price_df=price_df,
        crop_df=crop_df,
        acled_df=acled_df,
        data_dir=data_dir,
        iso3_list=['KEN', 'SOM', 'ETH']
    )
    
    # Save Output
    output_path = os.path.join(data_dir, "processed/subnational_merged_data.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    
    logger.info(f"Merge Complete! Saved to {output_path}")
    logger.info(f"Shape: {merged_df.shape}")
    print(merged_df.head())
    print(merged_df.describe())

if __name__ == "__main__":
    main()

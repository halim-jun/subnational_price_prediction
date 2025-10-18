#!/usr/bin/env python3
"""
WFP Food Price Data Downloader

Downloads retail prices of key staples by market from WFP via HDX (Humanitarian Data Exchange).
Focuses on Eastern Africa markets with monthly granularity.
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WFPDownloader:
    def __init__(self, output_dir="data/raw/wfp"):
        # HDX direct download URLs for WFP food prices
        self.hdx_urls = {
            '2019': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/b8399f61-e9d9-4c6a-93a4-303afb2197b3/download/wfp_food_prices_global_2019.csv',
            '2020': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/587f89db-9c7e-4e42-822a-6c23f060d63c/download/wfp_food_prices_global_2020.csv',
            '2021': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/70bc3058-1ff7-41e3-b16a-3492422fcab6/download/wfp_food_prices_global_2021.csv',
            '2022': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/747fe8d0-83e7-4da7-a40e-3afdd11832c9/download/wfp_food_prices_global_2022.csv',
            '2023': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/e96b8f67-c4de-4173-a814-2f7d84c47475/download/wfp_food_prices_global_2023.csv',
            '2024': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/5867679b-7ef4-4117-84b8-2bb4ddd7817f/download/wfp_food_prices_global_2024.csv',
            '2025': 'https://data.humdata.org/dataset/31579af5-3895-4002-9ee3-c50857480785/resource/d62af4be-cff6-437b-89a3-67f8fa4c53bf/download/wfp_food_prices_global_2025.csv'
        }

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa country codes (ISO3)
        self.east_africa_countries = [
            'ETH', 'KEN', 'SOM', 'SSD', 'UGA', 'TZA',
            'RWA', 'BDI', 'DJI', 'ERI', 'MDG'
        ]

        # Key staple commodities
        self.key_commodities = [
            'Maize', 'Sorghum', 'Millet', 'Rice', 'Wheat', 'Beans',
            'Cowpeas', 'Groundnuts', 'Cassava', 'Sweet potato', 'Oil'
        ]

    def download_hdx_data(self, year='2024'):
        """
        Download WFP data from HDX for a specific year

        Args:
            year (str): Year to download (2024 or 2025)
        """
        if year not in self.hdx_urls:
            logger.error(f"Year {year} not available. Available years: {list(self.hdx_urls.keys())}")
            return pd.DataFrame()

        url = self.hdx_urls[year]

        try:
            logger.info(f"Downloading WFP data for {year} from HDX")
            logger.info(f"URL: {url}")

            response = requests.get(url, timeout=120)
            response.raise_for_status()

            # Read CSV directly from response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), low_memory=False)

            logger.info(f"Downloaded {len(df)} records for {year}")
            logger.info(f"Columns: {list(df.columns)}")

            # Clean up any header issues
            if df.columns[0].startswith('#'):
                # Remove # from first column name
                new_columns = list(df.columns)
                new_columns[0] = new_columns[0].lstrip('#')
                df.columns = new_columns
                logger.info(f"Cleaned column names: {list(df.columns)}")

            # Basic data cleaning
            df = df.dropna(subset=['date'] if 'date' in df.columns else [])

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading data for {year}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error downloading {year}: {e}")
            return pd.DataFrame()

    def filter_east_africa_data(self, df):
        """
        Filter data for Eastern Africa countries and key commodities

        Args:
            df (pd.DataFrame): Full WFP dataset
        """
        if df.empty:
            return df

        logger.info("Filtering for Eastern Africa countries and key commodities")

        # Check column names and adapt filtering
        logger.info(f"Available columns: {list(df.columns)}")

        # Common column name variations
        country_col = None
        commodity_col = None
        date_col = None

        for col in df.columns:
            if col.lower() in ['country', 'countrycode', 'country_code', 'adm0_name', 'countryiso3']:
                country_col = col
            elif col.lower() in ['commodity', 'commodityname', 'commodity_name']:
                commodity_col = col
            elif col.lower() in ['date', 'date_price', 'price_date']:
                date_col = col

        if country_col is None:
            logger.warning(f"No country column found. Available columns: {list(df.columns)}")
            return df

        # Filter by country
        if country_col:
            initial_count = len(df)
            df_filtered = df[df[country_col].isin(self.east_africa_countries)]
            logger.info(f"Filtered by country: {initial_count} -> {len(df_filtered)} records")
            df = df_filtered

        # Filter by commodity if available
        if commodity_col and not df.empty:
            initial_count = len(df)
            # Case-insensitive commodity filtering
            commodity_mask = df[commodity_col].str.contains('|'.join(self.key_commodities), case=False, na=False)
            df_filtered = df[commodity_mask]
            logger.info(f"Filtered by commodity: {initial_count} -> {len(df_filtered)} records")
            df = df_filtered

        return df

    def download_all_years(self, years=['2024', '2025'], start_date=None, end_date=None):
        """Download WFP data for specified years and filter for Eastern Africa"""
        all_data = []

        for year in years:
            if year in self.hdx_urls:
                logger.info(f"Processing {year} data...")
                df = self.download_hdx_data(year)

                if not df.empty:
                    # Filter for Eastern Africa
                    df_filtered = self.filter_east_africa_data(df)

                    if not df_filtered.empty:
                        all_data.append(df_filtered)
                        logger.info(f"Added {len(df_filtered)} records from {year}")
                    else:
                        logger.warning(f"No Eastern Africa data found in {year}")
                else:
                    logger.warning(f"No data downloaded for {year}")

                # Rate limiting
                time.sleep(2)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Apply date filtering if specified
            if start_date or end_date:
                combined_df = self.filter_by_date(combined_df, start_date, end_date)

            # Create descriptive filename
            min_year = combined_df['date'].dt.year.min()
            max_year = combined_df['date'].dt.year.max()
            year_range = f"{min_year}-{max_year}" if min_year != max_year else str(min_year)

            # Count countries and records for filename
            country_count = combined_df['countryiso3'].nunique()

            # Save with descriptive name
            descriptive_filename = f"wfp_food_prices_eastern_africa_{year_range}_{country_count}countries_{len(combined_df)}records.csv"
            output_path = self.output_dir / descriptive_filename
            combined_df.to_csv(output_path, index=False)

            # Also save timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"wfp_food_prices_backup_{timestamp}.csv"
            backup_path = self.output_dir / backup_filename
            combined_df.to_csv(backup_path, index=False)

            # Save latest version
            latest_path = self.output_dir / "wfp_food_prices_latest.csv"
            combined_df.to_csv(latest_path, index=False)

            logger.info(f"Downloaded {len(combined_df)} total records")
            logger.info(f"Data saved to {output_path}")

            # Print summary
            if 'adm0_name' in combined_df.columns:
                logger.info(f"Countries: {combined_df['adm0_name'].nunique()}")
                logger.info(f"Country list: {sorted(combined_df['adm0_name'].unique())}")

            return combined_df
        else:
            logger.warning("No data downloaded for any year")
            return pd.DataFrame()

    def filter_by_date(self, df, start_date=None, end_date=None):
        """Filter dataframe by date range"""
        if df.empty:
            return df

        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower():
                date_col = col
                break

        if date_col is None:
            logger.warning("No date column found for filtering")
            return df

        try:
            df[date_col] = pd.to_datetime(df[date_col])

            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df[date_col] >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df[date_col] <= end_date]

            logger.info(f"Date filtering applied: {len(df)} records remain")

        except Exception as e:
            logger.error(f"Error filtering by date: {e}")

        return df

    def get_market_functionality_data(self):
        """
        Download WFP Market Functionality Index (MFI) data
        This is a separate endpoint/dataset from WFP
        """
        # Note: This would require a different API endpoint
        # For now, we'll create a placeholder structure
        logger.info("Market Functionality Index download not implemented yet")
        logger.info("Requires separate WFP MFI API endpoint")
        return pd.DataFrame()

def main():
    """Main function to run the WFP downloader"""
    downloader = WFPDownloader()

    # Download historical data from 2019-2023
    logger.info("Downloading WFP historical data from HDX (2019-2023)")
    data = downloader.download_all_years(['2019', '2020', '2021', '2022', '2023'])

    if not data.empty:
        logger.info("WFP data download completed successfully")
        logger.info(f"Shape: {data.shape}")

        # Print column info
        logger.info(f"Columns: {list(data.columns)}")

        # Find and print date range
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                # Clean date column first
                data[date_col] = data[date_col].astype(str).str.strip()
                data = data[~data[date_col].str.startswith('#')]  # Remove any header rows
                data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
                data = data.dropna(subset=[date_col])  # Remove unparseable dates
                logger.info(f"Date range: {data[date_col].min()} to {data[date_col].max()}")
            except Exception as e:
                logger.error(f"Error parsing dates: {e}")
                logger.info(f"Date column sample: {data[date_col].head().tolist()}")

        # Print country info
        country_cols = [col for col in data.columns if any(x in col.lower() for x in ['country', 'adm0'])]
        if country_cols:
            country_col = country_cols[0]
            logger.info(f"Countries: {data[country_col].nunique()}")
            logger.info(f"Country breakdown: {data[country_col].value_counts().to_dict()}")

    else:
        logger.error("WFP data download failed")

if __name__ == "__main__":
    main()
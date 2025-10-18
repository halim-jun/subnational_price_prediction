#!/usr/bin/env python3
"""
ACLED Conflict Data Downloader

Downloads conflict event data from ACLED (Armed Conflict Location & Event Data Project).
Focuses on Eastern Africa with filtering by event types and spatial proximity to markets.
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

class ACLEDDownloader:
    def __init__(self, output_dir="data/raw/acled"):
        self.base_url = "https://api.acleddata.com/acled/read"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa country codes (ISO3)
        self.east_africa_countries = [
            'ETH', 'KEN', 'SOM', 'SSD', 'UGA', 'TZA',
            'RWA', 'BDI', 'DJI', 'ERI', 'MDG'
        ]

        # Event types of interest for food price analysis
        self.event_types = [
            'Battles',
            'Explosions/Remote violence',
            'Violence against civilians',
            'Riots',
            'Protests',
            'Strategic developments'
        ]

        # Sub-event types
        self.sub_event_types = [
            'Armed clash',
            'Government regains territory',
            'Non-state actor overtakes territory',
            'Attack',
            'Abduction/forced disappearance',
            'Sexual violence',
            'Violent demonstration',
            'Peaceful protest',
            'Mob violence'
        ]

    def get_country_data(self, country_code, start_date=None, end_date=None, limit=0):
        """
        Download ACLED data for a specific country

        Args:
            country_code (str): ISO3 country code
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            limit (int): Maximum number of records (0 for no limit)
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        params = {
            'iso': country_code,
            'event_date': f"{start_date}|{end_date}",
            'event_date_where': 'BETWEEN',
            'format': 'json',
            'limit': limit if limit > 0 else 0
        }

        try:
            logger.info(f"Downloading ACLED data for {country_code}")
            response = requests.get(self.base_url, params=params, timeout=120)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and len(data['data']) > 0:
                df = pd.DataFrame(data['data'])

                # Convert date column
                if 'event_date' in df.columns:
                    df['event_date'] = pd.to_datetime(df['event_date'])

                # Add derived columns for analysis
                if 'fatalities' in df.columns:
                    df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)

                # Convert coordinates to numeric
                for col in ['latitude', 'longitude']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                return df
            else:
                logger.warning(f"No ACLED data returned for {country_code}")
                return pd.DataFrame()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading ACLED data for {country_code}: {e}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing ACLED JSON for {country_code}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error for ACLED {country_code}: {e}")
            return pd.DataFrame()

    def download_all_countries(self, start_date=None, end_date=None):
        """Download ACLED data for all Eastern Africa countries"""
        all_data = []

        for country in self.east_africa_countries:
            df = self.get_country_data(country, start_date, end_date)
            if not df.empty:
                df['country_code'] = country
                all_data.append(df)

            # Rate limiting - ACLED API has usage limits
            time.sleep(2)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Create descriptive filename
            min_date = combined_df['event_date'].min()
            max_date = combined_df['event_date'].max()
            year_range = f"{min_date.year}-{max_date.year}" if min_date.year != max_date.year else str(min_date.year)

            country_count = combined_df['country_code'].nunique()
            event_count = len(combined_df)

            # Save with descriptive name
            descriptive_filename = f"acled_conflict_eastern_africa_{year_range}_{country_count}countries_{event_count}events.csv"
            output_path = self.output_dir / descriptive_filename
            combined_df.to_csv(output_path, index=False)

            # Save latest version
            latest_path = self.output_dir / "acled_conflict_latest.csv"
            combined_df.to_csv(latest_path, index=False)

            logger.info(f"Downloaded {len(combined_df)} ACLED records for {len(all_data)} countries")
            logger.info(f"Data saved to {output_path}")

            # Print summary statistics
            if 'event_type' in combined_df.columns:
                logger.info(f"Event types: {combined_df['event_type'].value_counts().to_dict()}")
            if 'fatalities' in combined_df.columns:
                logger.info(f"Total fatalities: {combined_df['fatalities'].sum()}")

            return combined_df
        else:
            logger.warning("No ACLED data downloaded for any country")
            return pd.DataFrame()

    def aggregate_by_month_location(self, df, buffer_km=50):
        """
        Aggregate conflict events by month and location for market proximity analysis

        Args:
            df (pd.DataFrame): Raw ACLED data
            buffer_km (float): Buffer distance in km for spatial aggregation
        """
        if df.empty:
            return pd.DataFrame()

        # Create year-month column
        df['year_month'] = df['event_date'].dt.to_period('M')

        # Group by location and time
        agg_functions = {
            'data_date': 'count',  # Number of events
            'fatalities': 'sum',   # Total fatalities
            'latitude': 'mean',    # Average location
            'longitude': 'mean'
        }

        # Aggregate by event type as well
        if 'event_type' in df.columns:
            event_counts = df.groupby(['year_month', 'latitude', 'longitude', 'event_type']).size().reset_index(name='event_count')
            return event_counts
        else:
            aggregated = df.groupby(['year_month', 'latitude', 'longitude']).agg(agg_functions).reset_index()
            aggregated.columns = ['year_month', 'latitude', 'longitude', 'total_events', 'total_fatalities', 'avg_lat', 'avg_lon']
            return aggregated

def main():
    """Main function to run the ACLED downloader"""
    downloader = ACLEDDownloader()

    # Download last 5 years of data
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    logger.info(f"Downloading ACLED data from {start_date} to {end_date}")
    data = downloader.download_all_countries(start_date, end_date)

    if not data.empty:
        logger.info("ACLED data download completed successfully")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Countries: {data['country_code'].nunique()}")
        logger.info(f"Date range: {data['event_date'].min()} to {data['event_date'].max()}")

        # Create aggregated version
        aggregated = downloader.aggregate_by_month_location(data)
        if not aggregated.empty:
            agg_path = downloader.output_dir / "acled_monthly_aggregated.csv"
            aggregated.to_csv(agg_path, index=False)
            logger.info(f"Aggregated data saved to {agg_path}")
    else:
        logger.error("ACLED data download failed")

if __name__ == "__main__":
    main()
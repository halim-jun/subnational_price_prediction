#!/usr/bin/env python3
"""
Open-Meteo Weather Data Downloader

Downloads historical weather data using Open-Meteo API for exact market locations
from our WFP dataset. Gets temperature, weather code, daylight duration,
precipitation, and snowfall data.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenMeteoWeatherDownloader:
    def __init__(self, output_dir="data/raw/weather"):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Weather variables we want (daily data)
        self.daily_variables = [
            'temperature_2m_max',      # Maximum daily temperature
            'temperature_2m_min',      # Minimum daily temperature
            'weather_code',            # WMO weather code
            'daylight_duration',       # Daylight duration in seconds
            'precipitation_sum',       # Daily precipitation sum
            'snowfall_sum'            # Daily snowfall sum
        ]

    def extract_market_locations(self, wfp_file_path):
        """
        Extract unique market locations from WFP data

        Args:
            wfp_file_path (str): Path to WFP CSV file
        """
        logger.info(f"Extracting market locations from {wfp_file_path}")

        try:
            # Read WFP data
            df = pd.read_csv(wfp_file_path, low_memory=False)

            # Convert date for filtering
            df['date'] = pd.to_datetime(df['date'])

            # Get unique market locations
            markets = df[['market', 'market_id', 'countryiso3', 'latitude', 'longitude']].drop_duplicates()

            # Clean coordinates - remove any invalid values
            markets = markets.dropna(subset=['latitude', 'longitude'])
            markets = markets[(markets['latitude'].between(-90, 90)) &
                            (markets['longitude'].between(-180, 180))]

            # Get date range for weather data
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            logger.info(f"Found {len(markets)} unique markets")
            logger.info(f"Date range: {min_date} to {max_date}")
            logger.info(f"Countries: {sorted(markets['countryiso3'].unique())}")

            return markets, min_date, max_date

        except Exception as e:
            logger.error(f"Error extracting market locations: {e}")
            return pd.DataFrame(), None, None

    def get_weather_for_location(self, lat, lon, start_date, end_date, location_info):
        """
        Get weather data for a specific location and date range

        Args:
            lat (float): Latitude
            lon (float): Longitude
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD)
            location_info (dict): Market information
        """

        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(self.daily_variables),
            'timezone': 'auto'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Extract daily data
            daily_data = data.get('daily', {})

            if not daily_data:
                logger.warning(f"No daily data for {location_info['market']}")
                return pd.DataFrame()

            # Convert to DataFrame
            weather_df = pd.DataFrame()

            # Add dates
            if 'time' in daily_data:
                weather_df['date'] = pd.to_datetime(daily_data['time'])

            # Add weather variables
            for var in self.daily_variables:
                if var in daily_data:
                    weather_df[var] = daily_data[var]

            # Add location information
            weather_df['market'] = location_info['market']
            weather_df['market_id'] = location_info['market_id']
            weather_df['countryiso3'] = location_info['countryiso3']
            weather_df['latitude'] = lat
            weather_df['longitude'] = lon
            weather_df['elevation'] = data.get('elevation', np.nan)
            weather_df['timezone'] = data.get('timezone', '')

            # Add derived variables
            weather_df['temperature_2m_mean'] = (weather_df['temperature_2m_max'] + weather_df['temperature_2m_min']) / 2
            weather_df['temperature_2m_range'] = weather_df['temperature_2m_max'] - weather_df['temperature_2m_min']

            # Convert daylight duration from seconds to hours
            if 'daylight_duration' in weather_df.columns:
                weather_df['daylight_duration_hours'] = weather_df['daylight_duration'] / 3600

            return weather_df

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {location_info['market']} ({lat}, {lon})")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {location_info['market']}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting weather for {location_info['market']}: {e}")
            return pd.DataFrame()

    def download_weather_for_markets(self, markets_df, start_date, end_date, max_workers=5):
        """
        Download weather data for all markets using parallel processing

        Args:
            markets_df (pd.DataFrame): Market locations
            start_date (str): Start date
            end_date (str): End date
            max_workers (int): Maximum parallel workers
        """
        logger.info(f"Downloading weather data for {len(markets_df)} markets")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Using {max_workers} parallel workers")

        all_weather_data = []
        failed_markets = []

        # Prepare tasks
        tasks = []
        for _, market in markets_df.iterrows():
            location_info = {
                'market': market['market'],
                'market_id': market['market_id'],
                'countryiso3': market['countryiso3']
            }
            tasks.append((market['latitude'], market['longitude'], start_date, end_date, location_info))

        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_market = {
                executor.submit(self.get_weather_for_location, lat, lon, start_date, end_date, info): info
                for lat, lon, start_date, end_date, info in tasks
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_market), 1):
                market_info = future_to_market[future]

                try:
                    weather_data = future.result()

                    if not weather_data.empty:
                        all_weather_data.append(weather_data)
                        if i % 10 == 0:
                            logger.info(f"Completed {i}/{len(tasks)} markets")
                    else:
                        failed_markets.append(market_info['market'])

                except Exception as e:
                    logger.error(f"Failed to get weather for {market_info['market']}: {e}")
                    failed_markets.append(market_info['market'])

                # Rate limiting
                time.sleep(0.1)

        # Combine all data
        if all_weather_data:
            combined_weather = pd.concat(all_weather_data, ignore_index=True)

            # Sort by location and date
            combined_weather = combined_weather.sort_values(['countryiso3', 'market', 'date'])

            logger.info(f"Successfully downloaded weather for {len(all_weather_data)} markets")
            logger.info(f"Total weather records: {len(combined_weather):,}")
            logger.info(f"Failed markets: {len(failed_markets)}")

            if failed_markets:
                logger.warning(f"Failed markets: {failed_markets[:10]}...")  # Show first 10

            return combined_weather
        else:
            logger.error("No weather data downloaded")
            return pd.DataFrame()

    def save_weather_data(self, weather_df, start_date, end_date):
        """Save weather data with descriptive filename"""

        if weather_df.empty:
            logger.warning("No weather data to save")
            return

        # Create descriptive filename
        country_count = weather_df['countryiso3'].nunique()
        market_count = weather_df['market'].nunique()
        record_count = len(weather_df)

        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)

        filename = f"weather_openmeteo_eastern_africa_{year_range}_{country_count}countries_{market_count}markets_{record_count}records.csv"
        output_path = self.output_dir / filename

        # Save data
        weather_df.to_csv(output_path, index=False)

        # Save latest version
        latest_path = self.output_dir / "weather_openmeteo_latest.csv"
        weather_df.to_csv(latest_path, index=False)

        logger.info(f"Weather data saved to: {output_path}")

        # Create summary
        self.create_weather_summary(weather_df, output_path.parent / f"weather_summary_{year_range}.json")

        return output_path

    def create_weather_summary(self, weather_df, summary_path):
        """Create weather data summary"""

        summary = {
            'total_records': len(weather_df),
            'countries': weather_df['countryiso3'].nunique(),
            'markets': weather_df['market'].nunique(),
            'date_range': {
                'start': weather_df['date'].min().strftime('%Y-%m-%d'),
                'end': weather_df['date'].max().strftime('%Y-%m-%d'),
                'days': (weather_df['date'].max() - weather_df['date'].min()).days
            },
            'country_breakdown': weather_df['countryiso3'].value_counts().to_dict(),
            'weather_statistics': {
                'temperature_range': {
                    'min': float(weather_df['temperature_2m_min'].min()),
                    'max': float(weather_df['temperature_2m_max'].max()),
                    'mean': float(weather_df['temperature_2m_mean'].mean())
                },
                'precipitation_total': float(weather_df['precipitation_sum'].sum()),
                'precipitation_mean_daily': float(weather_df['precipitation_sum'].mean()),
                'snowfall_days': int((weather_df['snowfall_sum'] > 0).sum()),
                'daylight_hours_range': {
                    'min': float(weather_df['daylight_duration_hours'].min()),
                    'max': float(weather_df['daylight_duration_hours'].max())
                }
            },
            'data_quality': {
                'missing_temperature': int(weather_df['temperature_2m_mean'].isna().sum()),
                'missing_precipitation': int(weather_df['precipitation_sum'].isna().sum()),
                'missing_weather_code': int(weather_df['weather_code'].isna().sum())
            }
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Weather summary saved to: {summary_path}")

def main():
    """Main function to download weather data"""
    downloader = OpenMeteoWeatherDownloader()

    # Path to WFP data
    wfp_file = "data/raw/wfp/wfp_food_prices_latest.csv"

    if not Path(wfp_file).exists():
        logger.error(f"WFP file not found: {wfp_file}")
        return

    # Extract market locations
    logger.info("Step 1: Extracting market locations from WFP data")
    markets_df, start_date, end_date = downloader.extract_market_locations(wfp_file)

    if markets_df.empty:
        logger.error("No market locations found")
        return

    # Download weather data
    logger.info("Step 2: Downloading weather data from Open-Meteo")
    weather_df = downloader.download_weather_for_markets(
        markets_df,
        str(start_date),
        str(end_date),
        max_workers=3  # Conservative to avoid rate limiting
    )

    # Save data
    logger.info("Step 3: Saving weather data")
    if not weather_df.empty:
        output_path = downloader.save_weather_data(weather_df, str(start_date), str(end_date))

        logger.info("✅ Weather data download completed successfully!")
        logger.info(f"📊 {len(weather_df):,} weather records")
        logger.info(f"🏪 {weather_df['market'].nunique()} markets")
        logger.info(f"🌍 {weather_df['countryiso3'].nunique()} countries")
        logger.info(f"📅 {(weather_df['date'].max() - weather_df['date'].min()).days} days")
        logger.info(f"💾 Saved to: {output_path}")
    else:
        logger.error("❌ Weather data download failed")

if __name__ == "__main__":
    main()
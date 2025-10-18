#!/usr/bin/env python3
"""
Open-Meteo Weather Data Downloader v2 - With Rate Limiting Protection

Fixed version that handles 429 rate limiting errors properly with:
- Exponential backoff retry logic
- Progress saving and resuming
- Sequential requests (no parallel)
- Longer delays between requests
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenMeteoWeatherDownloaderV2:
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

        # Rate limiting settings
        self.base_delay = 2.0          # Base delay between requests (seconds)
        self.max_retries = 5           # Maximum retry attempts
        self.backoff_factor = 2.0      # Exponential backoff multiplier
        self.progress_save_interval = 10  # Save progress every N markets

    def extract_market_locations(self, wfp_file_path):
        """Extract unique market locations from WFP data"""
        logger.info(f"Extracting market locations from {wfp_file_path}")

        try:
            df = pd.read_csv(wfp_file_path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'])

            # Get unique market locations
            markets = df[['market', 'market_id', 'countryiso3', 'latitude', 'longitude']].drop_duplicates()
            markets = markets.dropna(subset=['latitude', 'longitude'])
            markets = markets[(markets['latitude'].between(-90, 90)) &
                            (markets['longitude'].between(-180, 180))]

            # Add index for progress tracking
            markets = markets.reset_index(drop=True)
            markets['download_index'] = markets.index

            min_date = df['date'].min().date()
            max_date = df['date'].max().date()

            logger.info(f"Found {len(markets)} unique markets")
            logger.info(f"Date range: {min_date} to {max_date}")

            return markets, min_date, max_date

        except Exception as e:
            logger.error(f"Error extracting market locations: {e}")
            return pd.DataFrame(), None, None

    def get_weather_for_location_with_retry(self, lat, lon, start_date, end_date, location_info):
        """
        Get weather data with retry logic and exponential backoff
        """
        for attempt in range(self.max_retries):
            try:
                # Add random jitter to prevent thundering herd
                delay = self.base_delay + random.uniform(0, 1)
                if attempt > 0:
                    # Exponential backoff
                    delay = delay * (self.backoff_factor ** attempt)
                    logger.info(f"Retry {attempt} for {location_info['market']} after {delay:.1f}s delay")

                time.sleep(delay)

                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'start_date': start_date,
                    'end_date': end_date,
                    'daily': ','.join(self.daily_variables),
                    'timezone': 'auto'
                }

                response = requests.get(self.base_url, params=params, timeout=30)

                # Handle different HTTP status codes
                if response.status_code == 429:
                    # Rate limited - wait longer and retry
                    wait_time = delay * 2
                    logger.warning(f"Rate limited for {location_info['market']}. Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()
                daily_data = data.get('daily', {})

                if not daily_data:
                    logger.warning(f"No daily data for {location_info['market']}")
                    return pd.DataFrame()

                # Convert to DataFrame
                weather_df = pd.DataFrame()

                if 'time' in daily_data:
                    weather_df['date'] = pd.to_datetime(daily_data['time'])

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
                weather_df['download_index'] = location_info['download_index']

                # Add derived variables
                weather_df['temperature_2m_mean'] = (weather_df['temperature_2m_max'] + weather_df['temperature_2m_min']) / 2
                weather_df['temperature_2m_range'] = weather_df['temperature_2m_max'] - weather_df['temperature_2m_min']

                if 'daylight_duration' in weather_df.columns:
                    weather_df['daylight_duration_hours'] = weather_df['daylight_duration'] / 3600

                logger.info(f"✅ Downloaded weather for {location_info['market']} ({location_info['countryiso3']}) - {len(weather_df)} records")
                return weather_df

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {location_info['market']} (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Unexpected error for {location_info['market']}: {e}")
                return pd.DataFrame()

        logger.error(f"Failed to download weather for {location_info['market']} after {self.max_retries} attempts")
        return pd.DataFrame()

    def save_progress(self, weather_data_list, completed_indices, output_file):
        """Save current progress"""
        if weather_data_list:
            combined_df = pd.concat(weather_data_list, ignore_index=True)
            combined_df.to_csv(output_file, index=False)

            progress_file = self.output_dir / "download_progress.json"
            progress = {
                'completed_indices': completed_indices,
                'last_updated': datetime.now().isoformat(),
                'records_downloaded': len(combined_df),
                'markets_completed': len(weather_data_list)
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

            logger.info(f"Progress saved: {len(weather_data_list)} markets, {len(combined_df)} records")

    def load_progress(self):
        """Load previous progress if exists"""
        progress_file = self.output_dir / "download_progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            logger.info(f"Found previous progress: {progress['markets_completed']} markets completed")
            return set(progress['completed_indices'])
        return set()

    def download_weather_sequential(self, markets_df, start_date, end_date):
        """
        Download weather data sequentially with rate limiting protection
        """
        logger.info(f"Starting sequential download for {len(markets_df)} markets")
        logger.info(f"Rate limiting settings: {self.base_delay}s base delay, {self.max_retries} max retries")

        # Load previous progress
        completed_indices = self.load_progress()

        # Filter out already completed markets
        remaining_markets = markets_df[~markets_df['download_index'].isin(completed_indices)]
        logger.info(f"Resuming download: {len(remaining_markets)} markets remaining")

        all_weather_data = []
        failed_markets = []

        # Load existing data if resuming
        temp_output_file = self.output_dir / "weather_download_in_progress.csv"
        if temp_output_file.exists():
            try:
                existing_df = pd.read_csv(temp_output_file)
                if not existing_df.empty:
                    # Group by market to recreate the list structure
                    for market_name in existing_df['market'].unique():
                        market_data = existing_df[existing_df['market'] == market_name]
                        all_weather_data.append(market_data)
                    logger.info(f"Loaded {len(all_weather_data)} markets from previous session")
            except Exception as e:
                logger.warning(f"Could not load previous session data: {e}")

        # Process remaining markets
        for i, (_, market) in enumerate(remaining_markets.iterrows(), 1):
            location_info = {
                'market': market['market'],
                'market_id': market['market_id'],
                'countryiso3': market['countryiso3'],
                'download_index': market['download_index']
            }

            logger.info(f"Processing {i}/{len(remaining_markets)}: {market['market']} ({market['countryiso3']})")

            weather_data = self.get_weather_for_location_with_retry(
                market['latitude'], market['longitude'],
                str(start_date), str(end_date),
                location_info
            )

            if not weather_data.empty:
                all_weather_data.append(weather_data)
                completed_indices.add(market['download_index'])
            else:
                failed_markets.append(market['market'])

            # Save progress periodically
            if i % self.progress_save_interval == 0:
                self.save_progress(all_weather_data, list(completed_indices), temp_output_file)

        # Final save
        if all_weather_data:
            self.save_progress(all_weather_data, list(completed_indices), temp_output_file)

            combined_weather = pd.concat(all_weather_data, ignore_index=True)
            combined_weather = combined_weather.sort_values(['countryiso3', 'market', 'date'])

            logger.info(f"✅ Download completed!")
            logger.info(f"Successfully downloaded: {len(all_weather_data)} markets")
            logger.info(f"Total weather records: {len(combined_weather):,}")
            logger.info(f"Failed markets: {len(failed_markets)}")

            return combined_weather
        else:
            logger.error("❌ No weather data downloaded")
            return pd.DataFrame()

    def save_weather_data(self, weather_df, start_date, end_date):
        """Save weather data with descriptive filename"""
        if weather_df.empty:
            logger.warning("No weather data to save")
            return

        country_count = weather_df['countryiso3'].nunique()
        market_count = weather_df['market'].nunique()
        record_count = len(weather_df)

        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)

        filename = f"weather_openmeteo_eastern_africa_{year_range}_{country_count}countries_{market_count}markets_{record_count}records.csv"
        output_path = self.output_dir / filename

        weather_df.to_csv(output_path, index=False)

        # Save latest version
        latest_path = self.output_dir / "weather_openmeteo_latest.csv"
        weather_df.to_csv(latest_path, index=False)

        # Clean up temporary file
        temp_file = self.output_dir / "weather_download_in_progress.csv"
        if temp_file.exists():
            temp_file.unlink()

        progress_file = self.output_dir / "download_progress.json"
        if progress_file.exists():
            progress_file.unlink()

        logger.info(f"Weather data saved to: {output_path}")
        return output_path

def main():
    """Main function with improved rate limiting"""
    downloader = OpenMeteoWeatherDownloaderV2()

    wfp_file = "data/raw/wfp/wfp_food_prices_latest.csv"

    if not Path(wfp_file).exists():
        logger.error(f"WFP file not found: {wfp_file}")
        return

    logger.info("🌤️ Open-Meteo Weather Downloader v2 - Rate Limit Protected")
    logger.info("=" * 60)

    # Extract market locations
    markets_df, start_date, end_date = downloader.extract_market_locations(wfp_file)

    if markets_df.empty:
        logger.error("No market locations found")
        return

    # Download weather data with rate limiting protection
    weather_df = downloader.download_weather_sequential(
        markets_df, start_date, end_date
    )

    # Save data
    if not weather_df.empty:
        output_path = downloader.save_weather_data(weather_df, str(start_date), str(end_date))

        logger.info("🎉 Weather data download completed successfully!")
        logger.info(f"📊 {len(weather_df):,} weather records")
        logger.info(f"🏪 {weather_df['market'].nunique()} markets")
        logger.info(f"🌍 {weather_df['countryiso3'].nunique()} countries")
        logger.info(f"📅 {(weather_df['date'].max() - weather_df['date'].min()).days} days")
        logger.info(f"💾 Saved to: {output_path}")
    else:
        logger.error("❌ Weather data download failed")

if __name__ == "__main__":
    main()
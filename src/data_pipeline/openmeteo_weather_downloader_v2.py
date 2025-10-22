#!/usr/bin/env python3
"""
Open-Meteo Weather Data Downloader v2 - With Rate Limiting Protection

This script downloads historical weather data for Eastern Africa markets using Open-Meteo Archive API.

KEY FEATURES:
- Country-specific downloads (choose one country at a time to save time)
- Extended date range (automatically downloads from 1 year before WFP data start date)
- Rate limiting protection with exponential backoff
- Progress saving and resuming (can stop and continue anytime)
- Descriptive filenames with country and exact date range

USAGE:
    Method 1 - Command line with country filter (RECOMMENDED):
        python openmeteo_weather_downloader_v2.py --country KEN
        python openmeteo_weather_downloader_v2.py --country ETH
        
    Method 2 - In Python code:
        from openmeteo_weather_downloader_v2 import main
        main(country_filter='KEN')
        
    Method 3 - Download all countries (NOT RECOMMENDED - takes very long):
        python openmeteo_weather_downloader_v2.py

AVAILABLE COUNTRY CODES:
    KEN - Kenya          RWA - Rwanda         BDI - Burundi
    ETH - Ethiopia       TZA - Tanzania       DJI - Djibouti
    UGA - Uganda         SOM - Somalia        ERI - Eritrea
    SSD - South Sudan

OUTPUT FILENAME FORMAT:
    weather_{COUNTRY}_{START-DATE}_to_{END-DATE}_{N}markets_{M}records.csv
    
    Example: weather_KEN_2018-01-08_to_2025-02-28_57markets_146723records.csv

DOWNLOADED WEATHER VARIABLES (Daily):
    - temperature_2m_max/min: Daily max/min temperature (°C)
    - precipitation_sum: Daily total precipitation (mm)
    - snowfall_sum: Daily total snowfall (cm)
    - weather_code: WMO weather code
    - daylight_duration: Daylight duration (seconds)
    + Derived variables: mean temperature, temperature range, daylight hours

TIMING:
    - ~3-5 seconds per market (with rate limiting protection)
    - Kenya (~57 markets): ~3-5 minutes
    - Ethiopia (~50 markets): ~3-5 minutes
    - Smaller countries: ~2-3 minutes

RESUME SUPPORT:
    If download is interrupted, simply run the same command again.
    Progress is saved every 10 markets in 'download_progress.json'.

TECHNICAL DETAILS:
    - API: Open-Meteo Archive API (https://archive-api.open-meteo.com)
    - Rate limiting: 2 second base delay between requests
    - Retry logic: Up to 5 attempts with exponential backoff
    - Data resolution: Daily
    - Timezone: Auto-detected for each location
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

    def extract_market_locations(self, wfp_file_path, country_filter=None):
        """
        Extract unique market locations from WFP data
        
        Args:
            wfp_file_path: Path to WFP CSV file
            country_filter: ISO3 country code to filter (e.g., 'KEN', 'ETH') or None for all
        """
        logger.info(f"Extracting market locations from {wfp_file_path}")
        if country_filter:
            logger.info(f"Filtering for country: {country_filter}")

        try:
            df = pd.read_csv(wfp_file_path, low_memory=False)
            df['date'] = pd.to_datetime(df['date'])

            # Filter by country if specified
            if country_filter:
                df = df[df['countryiso3'] == country_filter.upper()]
                if df.empty:
                    logger.error(f"No data found for country: {country_filter}")
                    available_countries = pd.read_csv(wfp_file_path)['countryiso3'].unique()
                    logger.info(f"Available countries: {', '.join(available_countries)}")
                    return pd.DataFrame(), None, None

            # Get unique market locations
            markets = df[['market', 'market_id', 'countryiso3', 'latitude', 'longitude']].drop_duplicates()
            markets = markets.dropna(subset=['latitude', 'longitude'])
            markets = markets[(markets['latitude'].between(-90, 90)) &
                            (markets['longitude'].between(-180, 180))]

            # Add index for progress tracking
            markets = markets.reset_index(drop=True)
            markets['download_index'] = markets.index

            # Get date range and extend 1 year before
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            # Extend start date by 1 year
            min_date_extended = min_date.replace(year=min_date.year - 1)

            logger.info(f"Found {len(markets)} unique markets")
            logger.info(f"Original date range: {min_date} to {max_date}")
            logger.info(f"Extended date range (1 year before): {min_date_extended} to {max_date}")

            return markets, min_date_extended, max_date

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

    def save_progress(self, weather_data_list, completed_indices, output_file, country_code=None):
        """
        Save current progress with country-specific filenames
        
        Args:
            weather_data_list: List of weather dataframes
            completed_indices: Set of completed market indices
            output_file: Path to temporary output CSV file
            country_code: Country ISO3 code for filename (e.g., 'KEN')
        """
        if weather_data_list:
            combined_df = pd.concat(weather_data_list, ignore_index=True)
            combined_df.to_csv(output_file, index=False)

            # Country-specific progress filename
            if country_code:
                progress_file = self.output_dir / f"download_progress_{country_code}.json"
            else:
                progress_file = self.output_dir / "download_progress.json"
                
            progress = {
                'country': country_code,
                'completed_indices': completed_indices,
                'last_updated': datetime.now().isoformat(),
                'records_downloaded': len(combined_df),
                'markets_completed': len(weather_data_list)
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

            logger.info(f"Progress saved: {len(weather_data_list)} markets, {len(combined_df)} records")

    def load_progress(self, country_code=None):
        """
        Load previous progress if exists
        
        Args:
            country_code: Country ISO3 code for filename (e.g., 'KEN')
            
        Returns:
            Set of completed market indices
        """
        # Country-specific progress filename
        if country_code:
            progress_file = self.output_dir / f"download_progress_{country_code}.json"
        else:
            progress_file = self.output_dir / "download_progress.json"
            
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            logger.info(f"Found previous progress for {country_code}: {progress['markets_completed']} markets completed")
            return set(progress['completed_indices'])
        return set()

    def download_weather_sequential(self, markets_df, start_date, end_date):
        """
        Download weather data sequentially with rate limiting protection
        """
        logger.info(f"Starting sequential download for {len(markets_df)} markets")
        logger.info(f"Rate limiting settings: {self.base_delay}s base delay, {self.max_retries} max retries")

        # Detect country code from markets data (use first market's country)
        country_code = None
        if not markets_df.empty and 'countryiso3' in markets_df.columns:
            unique_countries = markets_df['countryiso3'].unique()
            if len(unique_countries) == 1:
                country_code = unique_countries[0]
                logger.info(f"Detected single country: {country_code}")
            else:
                logger.warning(f"Multiple countries detected: {', '.join(unique_countries)}")
                country_code = "mixed"

        # Load previous progress (country-specific)
        completed_indices = self.load_progress(country_code)

        # Filter out already completed markets
        remaining_markets = markets_df[~markets_df['download_index'].isin(completed_indices)]
        logger.info(f"Resuming download: {len(remaining_markets)} markets remaining")

        all_weather_data = []
        failed_markets = []

        # Load existing data if resuming (country-specific temp file)
        if country_code:
            temp_output_file = self.output_dir / f"weather_download_in_progress_{country_code}.csv"
        else:
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
                self.save_progress(all_weather_data, list(completed_indices), temp_output_file, country_code)

        # Final save
        if all_weather_data:
            self.save_progress(all_weather_data, list(completed_indices), temp_output_file, country_code)

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
        """
        Save weather data with descriptive filename
        
        Filename format: weather_{country}_{YYYY-MM-DD}_to_{YYYY-MM-DD}_{N}markets_{M}records.csv
        """
        if weather_df.empty:
            logger.warning("No weather data to save")
            return

        # Get country information
        countries = weather_df['countryiso3'].unique()
        country_str = '_'.join(sorted(countries)) if len(countries) <= 3 else f"{len(countries)}countries"
        
        market_count = weather_df['market'].nunique()
        record_count = len(weather_df)

        # Format dates as YYYY-MM-DD
        start_date_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        end_date_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # Create descriptive filename with country and exact date range
        filename = f"weather_{country_str}_{start_date_str}_to_{end_date_str}_{market_count}markets_{record_count}records.csv"
        output_path = self.output_dir / filename

        weather_df.to_csv(output_path, index=False)

        # Save country-specific latest version
        if len(countries) == 1:
            latest_path = self.output_dir / f"weather_{countries[0]}_latest.csv"
        else:
            latest_path = self.output_dir / "weather_latest.csv"
        weather_df.to_csv(latest_path, index=False)

        # Clean up temporary files (country-specific)
        if len(countries) == 1:
            country_code = countries[0]
            temp_file = self.output_dir / f"weather_download_in_progress_{country_code}.csv"
            progress_file = self.output_dir / f"download_progress_{country_code}.json"
        else:
            # For mixed countries or legacy files
            temp_file = self.output_dir / "weather_download_in_progress.csv"
            progress_file = self.output_dir / "download_progress.json"
            
        if temp_file.exists():
            temp_file.unlink()
            logger.info(f"Cleaned up temporary file: {temp_file.name}")

        if progress_file.exists():
            progress_file.unlink()
            logger.info(f"Cleaned up progress file: {progress_file.name}")

        logger.info(f"Weather data saved to: {output_path}")
        logger.info(f"Latest copy saved to: {latest_path}")
        return output_path

def main(country_filter=None):
    """
    Main function with improved rate limiting
    
    Args:
        country_filter: ISO3 country code to filter (e.g., 'KEN', 'ETH', 'UGA') or None for all countries
        
    Example:
        # Download for Kenya only
        main(country_filter='KEN')
        
        # Download for all countries (not recommended, takes very long)
        main()
    """
    downloader = OpenMeteoWeatherDownloaderV2()

    wfp_file = "data/raw/wfp/wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"

    if not Path(wfp_file).exists():
        logger.error(f"WFP file not found: {wfp_file}")
        return

    logger.info("🌤️ Open-Meteo Weather Downloader v2 - Rate Limit Protected")
    logger.info("=" * 60)
    
    if country_filter:
        logger.info(f"🎯 Target country: {country_filter}")
    else:
        logger.warning("⚠️  No country filter set - will download ALL countries (this may take very long!)")

    # Extract market locations (with optional country filter)
    markets_df, start_date, end_date = downloader.extract_market_locations(
        wfp_file, 
        country_filter=country_filter
    )

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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download weather data from Open-Meteo for specific countries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download weather data for Kenya only
  python src/data_pipeline/openmeteo_weather_downloader_v2.py --country KEN
  
  # Download for Ethiopia
  python src/data_pipeline/openmeteo_weather_downloader_v2.py --country ETH
  
  # Download for multiple countries (not recommended in one run)
  python src/data_pipeline/openmeteo_weather_downloader_v2.py
  
Available country codes:
  KEN (Kenya), ETH (Ethiopia), UGA (Uganda), RWA (Rwanda),
  TZA (Tanzania), SOM (Somalia), SSD (South Sudan), etc.
        """
    )
    
    parser.add_argument(
        '--country', '-c',
        type=str,
        default=None,
        help='ISO3 country code (e.g., KEN, ETH, UGA). If not specified, downloads all countries.'
    )
    
    args = parser.parse_args()
    
    main(country_filter=args.country)
#!/usr/bin/env python3
"""
Download weather data specifically for Kenya markets only
"""

import pandas as pd
import sys
import os
sys.path.append('src')

from data_pipeline.openmeteo_weather_downloader_v2 import OpenMeteoWeatherDownloaderV2

def download_kenya_weather():
    """Download weather for Kenya markets only"""

    downloader = OpenMeteoWeatherDownloaderV2()

    print("🇰🇪 KENYA WEATHER DATA DOWNLOAD")
    print("=" * 40)
    print("Downloading weather for Kenya markets only")

    # Load Kenya markets
    kenya_file = "data/raw/weather/kenya_markets_for_download.csv"
    kenya_markets_df = pd.read_csv(kenya_file)

    if kenya_markets_df.empty:
        print("❌ No Kenya market locations found")
        return

    print(f"📍 Found {len(kenya_markets_df)} Kenya markets")

    # Load WFP data to get date range
    wfp_file = "data/raw/wfp/wfp_food_prices_latest.csv"
    wfp_df = pd.read_csv(wfp_file, low_memory=False)
    wfp_df['date'] = pd.to_datetime(wfp_df['date'])

    start_date = wfp_df['date'].min().date()
    end_date = wfp_df['date'].max().date()

    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"⏱️ Estimated download time: ~{len(kenya_markets_df) * 5} minutes")

    # Show sample markets
    print(f"\n🏪 Kenya markets to download:")
    for i, (_, market) in enumerate(kenya_markets_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {market['market']} ({market['latitude']:.2f}, {market['longitude']:.2f})")
    if len(kenya_markets_df) > 10:
        print(f"   ... and {len(kenya_markets_df) - 10} more markets")

    # Download weather data
    weather_df = downloader.download_weather_sequential(
        kenya_markets_df, start_date, end_date
    )

    if not weather_df.empty:
        # Save Kenya weather data
        country_count = weather_df['countryiso3'].nunique()
        market_count = weather_df['market'].nunique()
        record_count = len(weather_df)

        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)

        filename = f"weather_openmeteo_kenya_{year_range}_{market_count}markets_{record_count}records.csv"
        output_path = downloader.output_dir / filename

        weather_df.to_csv(output_path, index=False)

        # Also save as Kenya latest
        kenya_latest_path = downloader.output_dir / "weather_kenya_latest.csv"
        weather_df.to_csv(kenya_latest_path, index=False)

        print(f"\n✅ SUCCESS! Kenya weather data downloaded:")
        print(f"   📊 {len(weather_df):,} weather records")
        print(f"   🏪 {weather_df['market'].nunique()} markets")
        print(f"   📅 Date range: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}")
        print(f"   💾 Saved to: {output_path}")

        # Show weather summary
        print(f"\n🌡️ Kenya Weather Summary:")
        print(f"   Temperature range: {weather_df['temperature_2m_min'].min():.1f}°C to {weather_df['temperature_2m_max'].max():.1f}°C")
        print(f"   Average daily temp: {weather_df['temperature_2m_mean'].mean():.1f}°C")
        print(f"   Total precipitation: {weather_df['precipitation_sum'].sum():.1f}mm")
        print(f"   Rainy days: {(weather_df['precipitation_sum'] > 0).sum():,}")

        # Show market coverage by region
        print(f"\n🗺️ Market coverage:")
        major_cities = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']
        for city in major_cities:
            city_markets = weather_df[weather_df['market'].str.contains(city, case=False, na=False)]
            if not city_markets.empty:
                print(f"   {city}: {city_markets['market'].nunique()} markets")

        return True
    else:
        print("❌ Failed to download Kenya weather data")
        return False

if __name__ == "__main__":
    success = download_kenya_weather()

    if success:
        print(f"\n🎉 Kenya weather data ready!")
        print(f"💡 You can now analyze Kenya food prices with weather data")
    else:
        print(f"\n❌ Kenya weather download failed")
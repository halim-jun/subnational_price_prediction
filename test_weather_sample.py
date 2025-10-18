#!/usr/bin/env python3
"""
Test Open-Meteo weather downloader with a small sample
"""

import pandas as pd
import sys
import os
sys.path.append('src')

from data_pipeline.openmeteo_weather_downloader import OpenMeteoWeatherDownloader

def test_sample_markets():
    """Test with a small sample of markets first"""

    downloader = OpenMeteoWeatherDownloader()

    # Load WFP data and get a small sample
    wfp_file = "data/raw/wfp/wfp_food_prices_latest.csv"

    print("🧪 Testing Open-Meteo weather downloader with sample data")
    print("=" * 60)

    # Extract market locations
    markets_df, start_date, end_date = downloader.extract_market_locations(wfp_file)

    if markets_df.empty:
        print("❌ No market locations found")
        return

    print(f"📍 Found {len(markets_df)} total markets")

    # Take a small sample for testing - one market per country
    sample_markets = markets_df.groupby('countryiso3').first().reset_index()
    print(f"🧪 Testing with {len(sample_markets)} sample markets (one per country)")

    for _, market in sample_markets.iterrows():
        print(f"   {market['countryiso3']}: {market['market']} ({market['latitude']:.3f}, {market['longitude']:.3f})")

    # Test with a shorter date range first (just 2024)
    test_start = "2024-01-01"
    test_end = "2024-03-31"  # Just 3 months for testing

    print(f"\n🌤️ Testing weather download for {test_start} to {test_end}")

    # Download weather data
    weather_df = downloader.download_weather_for_markets(
        sample_markets,
        test_start,
        test_end,
        max_workers=2  # Conservative for testing
    )

    if not weather_df.empty:
        print(f"\n✅ SUCCESS! Weather data downloaded:")
        print(f"   📊 {len(weather_df):,} records")
        print(f"   🏪 {weather_df['market'].nunique()} markets")
        print(f"   🌍 Countries: {', '.join(sorted(weather_df['countryiso3'].unique()))}")
        print(f"   📅 Date range: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}")

        # Show sample data
        print(f"\n📋 Sample weather data:")
        print(weather_df[['market', 'countryiso3', 'date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']].head(10))

        # Show weather statistics
        print(f"\n🌡️ Temperature statistics:")
        print(f"   Min temp: {weather_df['temperature_2m_min'].min():.1f}°C")
        print(f"   Max temp: {weather_df['temperature_2m_max'].max():.1f}°C")
        print(f"   Mean temp: {weather_df['temperature_2m_mean'].mean():.1f}°C")

        print(f"\n🌧️ Precipitation statistics:")
        print(f"   Total precipitation: {weather_df['precipitation_sum'].sum():.1f}mm")
        print(f"   Average daily: {weather_df['precipitation_sum'].mean():.1f}mm")
        print(f"   Rainy days: {(weather_df['precipitation_sum'] > 0).sum()}")

        # Save sample data
        sample_path = downloader.output_dir / "weather_sample_test.csv"
        weather_df.to_csv(sample_path, index=False)
        print(f"\n💾 Sample data saved to: {sample_path}")

        return True
    else:
        print("❌ No weather data downloaded")
        return False

if __name__ == "__main__":
    success = test_sample_markets()
    if success:
        print("\n🎉 Weather downloader test PASSED!")
        print("Ready to download full dataset")
    else:
        print("\n❌ Weather downloader test FAILED!")
        print("Check API connection and parameters")
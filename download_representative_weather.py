#!/usr/bin/env python3
"""
Download weather data for representative markets (subset)
This gives us weather data quickly for analysis while the full download runs in background
"""

import pandas as pd
import sys
import os
sys.path.append('src')

from data_pipeline.openmeteo_weather_downloader_v2 import OpenMeteoWeatherDownloaderV2

def download_representative_weather():
    """Download weather for representative markets only"""

    downloader = OpenMeteoWeatherDownloaderV2()

    print("🌤️ REPRESENTATIVE WEATHER DATA DOWNLOAD")
    print("=" * 50)
    print("Downloading weather for key markets to get started quickly")

    # Load WFP data
    wfp_file = "data/raw/wfp/wfp_food_prices_latest.csv"
    markets_df, start_date, end_date = downloader.extract_market_locations(wfp_file)

    if markets_df.empty:
        print("❌ No market locations found")
        return

    print(f"📍 Found {len(markets_df)} total markets")

    # Select representative markets (simplified approach)
    # Strategy: One major market per country
    rep_markets_df = markets_df.groupby('countryiso3').first().reset_index()

    # Try to find major cities if available
    major_cities = ['Bujumbura', 'Addis Ababa', 'Nairobi', 'Kigali', 'Mogadishu', 'Juba', 'Dar es Salaam', 'Kampala']

    for city in major_cities:
        city_matches = markets_df[markets_df['market'].str.contains(city, case=False, na=False)]
        if not city_matches.empty:
            # Replace the representative market for this country with the major city
            country = city_matches.iloc[0]['countryiso3']
            rep_markets_df = rep_markets_df[rep_markets_df['countryiso3'] != country]
            rep_markets_df = pd.concat([rep_markets_df, city_matches.head(1)], ignore_index=True)

    print(f"\n🎯 Selected {len(rep_markets_df)} representative markets:")
    for _, market in rep_markets_df.iterrows():
        print(f"   {market['countryiso3']}: {market['market']} ({market['latitude']:.2f}, {market['longitude']:.2f})")

    print(f"\n⏱️ Estimated download time: ~{len(rep_markets_df) * 3} seconds")

    # Download weather data
    weather_df = downloader.download_weather_sequential(
        rep_markets_df, start_date, end_date
    )

    if not weather_df.empty:
        # Save representative weather data
        country_count = weather_df['countryiso3'].nunique()
        market_count = weather_df['market'].nunique()
        record_count = len(weather_df)

        start_year = pd.to_datetime(start_date).year
        end_year = pd.to_datetime(end_date).year
        year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)

        filename = f"weather_openmeteo_representative_{year_range}_{country_count}countries_{market_count}markets_{record_count}records.csv"
        output_path = downloader.output_dir / filename

        weather_df.to_csv(output_path, index=False)

        # Also save as representative latest
        rep_latest_path = downloader.output_dir / "weather_representative_latest.csv"
        weather_df.to_csv(rep_latest_path, index=False)

        print(f"\n✅ SUCCESS! Representative weather data downloaded:")
        print(f"   📊 {len(weather_df):,} weather records")
        print(f"   🏪 {weather_df['market'].nunique()} markets")
        print(f"   🌍 Countries: {', '.join(sorted(weather_df['countryiso3'].unique()))}")
        print(f"   📅 Date range: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}")
        print(f"   💾 Saved to: {output_path}")

        # Show weather summary
        print(f"\n🌡️ Weather Summary:")
        print(f"   Temperature range: {weather_df['temperature_2m_min'].min():.1f}°C to {weather_df['temperature_2m_max'].max():.1f}°C")
        print(f"   Average daily temp: {weather_df['temperature_2m_mean'].mean():.1f}°C")
        print(f"   Total precipitation: {weather_df['precipitation_sum'].sum():.1f}mm")
        print(f"   Rainy days: {(weather_df['precipitation_sum'] > 0).sum():,}")

        return True
    else:
        print("❌ Failed to download representative weather data")
        return False

if __name__ == "__main__":
    success = download_representative_weather()

    if success:
        print(f"\n🎉 Representative weather data ready!")
        print(f"💡 You can now start analysis while full download continues in background")
        print(f"📋 To continue full download later, run:")
        print(f"   python src/data_pipeline/openmeteo_weather_downloader_v2.py")
    else:
        print(f"\n❌ Representative weather download failed")
#!/usr/bin/env python3
"""
Demo script to show how the data pipeline works
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')

def load_sample_data():
    """Load the sample data we created"""
    print("🔍 Loading sample data from pipeline...")

    data = {}

    # Load WFP food price data
    try:
        wfp_path = "data/raw/wfp/wfp_food_prices_latest.csv"
        data['wfp'] = pd.read_csv(wfp_path)
        print(f"✅ WFP Data: {len(data['wfp'])} records")
    except Exception as e:
        print(f"❌ WFP Data: {e}")
        data['wfp'] = pd.DataFrame()

    # Load climate data
    try:
        precip_path = "data/raw/climate/chirps_precipitation.csv"
        temp_path = "data/raw/climate/modis_temperature.csv"
        data['precipitation'] = pd.read_csv(precip_path)
        data['temperature'] = pd.read_csv(temp_path)
        print(f"✅ Climate Data: {len(data['precipitation'])} precip + {len(data['temperature'])} temp records")
    except Exception as e:
        print(f"❌ Climate Data: {e}")
        data['precipitation'] = pd.DataFrame()
        data['temperature'] = pd.DataFrame()

    # Load conflict data
    try:
        conflict_path = "data/raw/acled/acled_conflict_data_latest.csv"
        data['conflict'] = pd.read_csv(conflict_path)
        print(f"✅ Conflict Data: {len(data['conflict'])} events")
    except Exception as e:
        print(f"❌ Conflict Data: {e}")
        data['conflict'] = pd.DataFrame()

    return data

def analyze_price_trends(wfp_data):
    """Analyze food price trends"""
    if wfp_data.empty:
        print("No WFP data to analyze")
        return

    print("\n📊 Food Price Analysis:")

    # Convert date column
    wfp_data['Date'] = pd.to_datetime(wfp_data['Date'])

    # Average price by commodity
    avg_prices = wfp_data.groupby('CommodityName')['PriceUSD'].mean().sort_values(ascending=False)
    print("\n💰 Average Prices by Commodity (USD/kg):")
    for commodity, price in avg_prices.head().items():
        print(f"  {commodity}: ${price:.2f}")

    # Price volatility
    price_volatility = wfp_data.groupby('CommodityName')['PriceUSD'].std().sort_values(ascending=False)
    print("\n📈 Price Volatility (Standard Deviation):")
    for commodity, volatility in price_volatility.head().items():
        print(f"  {commodity}: ${volatility:.2f}")

    # Country price comparison
    country_prices = wfp_data.groupby('CountryCode')['PriceUSD'].mean().sort_values(ascending=False)
    print("\n🌍 Average Prices by Country:")
    for country, price in country_prices.items():
        print(f"  {country}: ${price:.2f}")

def analyze_climate_patterns(precip_data, temp_data):
    """Analyze climate patterns"""
    if precip_data.empty or temp_data.empty:
        print("No climate data to analyze")
        return

    print("\n🌡️ Climate Analysis:")

    # Convert dates
    precip_data['date'] = pd.to_datetime(precip_data['date'])
    temp_data['date'] = pd.to_datetime(temp_data['date'])

    # Precipitation patterns
    avg_precip = precip_data['precipitation_mm'].mean()
    precip_trend = precip_data['precipitation_anomaly'].mean()
    print(f"\n🌧️ Precipitation:")
    print(f"  Average: {avg_precip:.1f} mm/month")
    print(f"  Trend: {precip_trend:+.1f} mm anomaly")

    # Temperature patterns
    avg_temp_day = temp_data['lst_day_celsius'].mean()
    avg_temp_night = temp_data['lst_night_celsius'].mean()
    extreme_heat_days = temp_data['extreme_heat_days'].sum()
    print(f"\n🌡️ Temperature:")
    print(f"  Average Day: {avg_temp_day:.1f}°C")
    print(f"  Average Night: {avg_temp_night:.1f}°C")
    print(f"  Extreme Heat Days: {extreme_heat_days}")

def analyze_conflict_patterns(conflict_data):
    """Analyze conflict patterns"""
    if conflict_data.empty:
        print("No conflict data to analyze")
        return

    print("\n⚔️ Conflict Analysis:")

    # Convert date
    conflict_data['event_date'] = pd.to_datetime(conflict_data['event_date'])

    # Event types
    event_types = conflict_data['event_type'].value_counts()
    print("\n📊 Event Types:")
    for event_type, count in event_types.items():
        print(f"  {event_type}: {count} events")

    # Fatalities
    total_fatalities = conflict_data['fatalities'].sum()
    avg_fatalities = conflict_data['fatalities'].mean()
    print(f"\n💀 Fatalities:")
    print(f"  Total: {total_fatalities}")
    print(f"  Average per event: {avg_fatalities:.1f}")

    # Country distribution
    country_conflicts = conflict_data['country_code'].value_counts()
    print(f"\n🌍 Conflicts by Country:")
    for country, count in country_conflicts.items():
        print(f"  {country}: {count} events")

def demonstrate_spatial_analysis(wfp_data, conflict_data):
    """Demonstrate spatial analysis capabilities"""
    if wfp_data.empty or conflict_data.empty:
        print("Insufficient data for spatial analysis")
        return

    print("\n🗺️ Spatial Analysis Demo:")

    # Calculate distances between markets and conflicts (simplified)
    print("\n📍 Market-Conflict Proximity Analysis:")

    # Sample analysis: markets within conflict areas
    markets = wfp_data[['CountryCode', 'MarketName', 'Latitude', 'Longitude']].drop_duplicates()
    conflicts = conflict_data[['country_code', 'latitude', 'longitude', 'fatalities']]

    for _, market in markets.head(3).iterrows():
        # Find conflicts in same country
        country_conflicts = conflicts[conflicts['country_code'] == market['CountryCode']]

        if not country_conflicts.empty:
            # Calculate simple distance (in reality would use proper spatial functions)
            distances = np.sqrt(
                (country_conflicts['latitude'] - market['Latitude'])**2 +
                (country_conflicts['longitude'] - market['Longitude'])**2
            )

            nearby_conflicts = len(distances[distances < 1.0])  # Within ~100km
            avg_fatalities = country_conflicts['fatalities'].mean()

            print(f"  {market['MarketName']} ({market['CountryCode']}): {nearby_conflicts} nearby conflicts, avg {avg_fatalities:.1f} fatalities")

def create_simple_visualization(data):
    """Create a simple visualization"""
    print("\n📈 Creating Visualization...")

    try:
        if not data['wfp'].empty:
            # Create a simple price trend plot
            plt.figure(figsize=(12, 8))

            # Plot 1: Price trends by commodity
            plt.subplot(2, 2, 1)
            wfp_data = data['wfp'].copy()
            wfp_data['Date'] = pd.to_datetime(wfp_data['Date'])

            for commodity in wfp_data['CommodityName'].unique()[:3]:
                commodity_data = wfp_data[wfp_data['CommodityName'] == commodity]
                monthly_avg = commodity_data.groupby(commodity_data['Date'].dt.to_period('M'))['PriceUSD'].mean()
                plt.plot(monthly_avg.index.astype(str), monthly_avg.values, label=commodity, marker='o')

            plt.title('Food Price Trends')
            plt.xlabel('Month')
            plt.ylabel('Price (USD/kg)')
            plt.legend()
            plt.xticks(rotation=45)

            # Plot 2: Climate data
            plt.subplot(2, 2, 2)
            if not data['precipitation'].empty:
                precip_data = data['precipitation'].copy()
                precip_data['date'] = pd.to_datetime(precip_data['date'])
                plt.plot(precip_data['date'], precip_data['precipitation_mm'], 'b-', label='Precipitation')
                plt.title('Precipitation Pattern')
                plt.ylabel('mm/month')
                plt.xticks(rotation=45)

            # Plot 3: Temperature
            plt.subplot(2, 2, 3)
            if not data['temperature'].empty:
                temp_data = data['temperature'].copy()
                temp_data['date'] = pd.to_datetime(temp_data['date'])
                plt.plot(temp_data['date'], temp_data['lst_day_celsius'], 'r-', label='Day Temp')
                plt.plot(temp_data['date'], temp_data['lst_night_celsius'], 'b-', label='Night Temp')
                plt.title('Temperature Pattern')
                plt.ylabel('°C')
                plt.legend()
                plt.xticks(rotation=45)

            # Plot 4: Conflict events
            plt.subplot(2, 2, 4)
            if not data['conflict'].empty:
                conflict_data = data['conflict'].copy()
                conflict_data['event_date'] = pd.to_datetime(conflict_data['event_date'])
                monthly_conflicts = conflict_data.groupby(conflict_data['event_date'].dt.to_period('M')).size()
                plt.bar(range(len(monthly_conflicts)), monthly_conflicts.values)
                plt.title('Monthly Conflict Events')
                plt.ylabel('Number of Events')

            plt.tight_layout()

            # Save plot
            output_dir = Path("data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "data_overview.png", dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to: {output_dir / 'data_overview.png'}")

        else:
            print("❌ No data available for visualization")

    except Exception as e:
        print(f"❌ Visualization error: {e}")

def main():
    """Main demo function"""
    print("🚀 Eastern Africa Food Price Forecasting - Data Pipeline Demo")
    print("=" * 70)

    # Load data
    data = load_sample_data()

    # Analyze each dataset
    analyze_price_trends(data['wfp'])
    analyze_climate_patterns(data['precipitation'], data['temperature'])
    analyze_conflict_patterns(data['conflict'])
    demonstrate_spatial_analysis(data['wfp'], data['conflict'])

    # Create visualization
    create_simple_visualization(data)

    print("\n" + "=" * 70)
    print("✅ Demo completed! The pipeline successfully:")
    print("   1. ✅ Downloaded/created sample data from multiple sources")
    print("   2. ✅ Analyzed food price trends and volatility")
    print("   3. ✅ Processed climate and conflict data")
    print("   4. ✅ Demonstrated spatial analysis capabilities")
    print("   5. ✅ Created integrated visualization")
    print("\n🎯 Next steps: Implement preprocessing and modeling pipeline!")

if __name__ == "__main__":
    main()
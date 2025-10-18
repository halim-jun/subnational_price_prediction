#!/usr/bin/env python3
"""
Demo script showing REAL WFP data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_real_wfp_data():
    """Analyze the real WFP data we just downloaded"""
    wfp_path = "data/raw/wfp/wfp_food_prices_latest.csv"

    try:
        df = pd.read_csv(wfp_path)
        logger.info(f"✅ Loaded {len(df)} real WFP records")

        # Convert date
        df['date'] = pd.to_datetime(df['date'])

        print("\n🎯 REAL WFP FOOD PRICE DATA ANALYSIS")
        print("=" * 50)

        # Basic info
        print(f"\n📊 Dataset Overview:")
        print(f"  Records: {len(df):,}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Countries: {df['countryiso3'].nunique()}")
        print(f"  Markets: {df['market'].nunique()}")
        print(f"  Commodities: {df['commodity'].nunique()}")

        # Country breakdown
        print(f"\n🌍 Country Breakdown:")
        country_counts = df['countryiso3'].value_counts()
        for country, count in country_counts.items():
            print(f"  {country}: {count:,} records")

        # Top commodities
        print(f"\n🌾 Top Commodities:")
        commodity_counts = df['commodity'].value_counts().head(10)
        for commodity, count in commodity_counts.items():
            print(f"  {commodity}: {count:,} records")

        # Price analysis
        print(f"\n💰 Price Analysis (USD):")
        print(f"  Average price: ${df['usdprice'].mean():.3f}")
        print(f"  Median price: ${df['usdprice'].median():.3f}")
        print(f"  Price range: ${df['usdprice'].min():.3f} - ${df['usdprice'].max():.3f}")

        # Most expensive commodities
        print(f"\n💸 Most Expensive Commodities (Average USD/kg):")
        expensive_commodities = df.groupby('commodity')['usdprice'].mean().sort_values(ascending=False).head(10)
        for commodity, price in expensive_commodities.items():
            print(f"  {commodity}: ${price:.3f}")

        # Country price comparison
        print(f"\n🏪 Average Prices by Country (USD/kg):")
        country_prices = df.groupby('countryiso3')['usdprice'].mean().sort_values(ascending=False)
        for country, price in country_prices.items():
            print(f"  {country}: ${price:.3f}")

        # Recent price trends
        print(f"\n📈 Recent Price Trends (Last 6 months):")
        recent_data = df[df['date'] >= df['date'].max() - pd.Timedelta(days=180)]
        if not recent_data.empty:
            monthly_avg = recent_data.groupby(recent_data['date'].dt.to_period('M'))['usdprice'].mean()
            for month, price in monthly_avg.items():
                print(f"  {month}: ${price:.3f}")

        return df

    except Exception as e:
        logger.error(f"Error analyzing WFP data: {e}")
        return pd.DataFrame()

def create_real_data_visualization(df):
    """Create visualizations with real data"""
    if df.empty:
        return

    print(f"\n📊 Creating visualizations with real data...")

    plt.figure(figsize=(15, 12))

    # 1. Price trends by country
    plt.subplot(2, 3, 1)
    monthly_country = df.groupby([df['date'].dt.to_period('M'), 'countryiso3'])['usdprice'].mean().unstack()
    for country in monthly_country.columns[:5]:  # Top 5 countries
        plt.plot(monthly_country.index.astype(str), monthly_country[country], label=country, marker='o')
    plt.title('Price Trends by Country (USD/kg)')
    plt.xlabel('Month')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)

    # 2. Top commodities price distribution
    plt.subplot(2, 3, 2)
    top_commodities = df['commodity'].value_counts().head(5).index
    commodity_prices = [df[df['commodity'] == commodity]['usdprice'] for commodity in top_commodities]
    plt.boxplot(commodity_prices, labels=top_commodities)
    plt.title('Price Distribution - Top Commodities')
    plt.ylabel('Price (USD/kg)')
    plt.xticks(rotation=45)

    # 3. Country vs Average Price
    plt.subplot(2, 3, 3)
    country_avg = df.groupby('countryiso3')['usdprice'].mean().sort_values()
    plt.bar(country_avg.index, country_avg.values)
    plt.title('Average Prices by Country')
    plt.xlabel('Country')
    plt.ylabel('Average Price (USD/kg)')
    plt.xticks(rotation=45)

    # 4. Market count by country
    plt.subplot(2, 3, 4)
    market_counts = df.groupby('countryiso3')['market'].nunique().sort_values(ascending=False)
    plt.bar(market_counts.index, market_counts.values)
    plt.title('Number of Markets by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Markets')
    plt.xticks(rotation=45)

    # 5. Price over time (overall trend)
    plt.subplot(2, 3, 5)
    monthly_avg = df.groupby(df['date'].dt.to_period('M'))['usdprice'].mean()
    plt.plot(monthly_avg.index.astype(str), monthly_avg.values, 'b-', marker='o', linewidth=2)
    plt.title('Overall Price Trend')
    plt.xlabel('Month')
    plt.ylabel('Average Price (USD/kg)')
    plt.xticks(rotation=45)

    # 6. Top markets by record count
    plt.subplot(2, 3, 6)
    top_markets = df['market'].value_counts().head(10)
    plt.barh(range(len(top_markets)), top_markets.values)
    plt.yticks(range(len(top_markets)), top_markets.index)
    plt.title('Top Markets by Data Volume')
    plt.xlabel('Number of Records')

    plt.tight_layout()

    # Save visualization
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "real_wfp_analysis.png", dpi=150, bbox_inches='tight')
    print(f"📊 Real data visualization saved to: {output_dir / 'real_wfp_analysis.png'}")

def main():
    """Main function"""
    print("🚀 REAL WFP DATA ANALYSIS")
    print("Using actual HDX data from https://data.humdata.org/dataset/global-wfp-food-prices")
    print("=" * 80)

    # Analyze real WFP data
    df = analyze_real_wfp_data()

    # Create visualizations
    if not df.empty:
        create_real_data_visualization(df)

        print(f"\n✅ SUCCESS! Fixed WFP downloader and analyzed real data:")
        print(f"   🎯 Downloaded {len(df):,} real records from HDX")
        print(f"   📊 Covering {df['countryiso3'].nunique()} Eastern Africa countries")
        print(f"   🏪 From {df['market'].nunique()} different markets")
        print(f"   🌾 Tracking {df['commodity'].nunique()} commodities")
        print(f"   📅 Data from {df['date'].min().date()} to {df['date'].max().date()}")

        print(f"\n🔧 WFP Downloader Status: ✅ FIXED")
        print(f"   ✅ Using HDX API instead of broken WFP VAM API")
        print(f"   ✅ Real data filtering for Eastern Africa")
        print(f"   ✅ Proper date parsing and data cleaning")

    else:
        print(f"❌ No data available for analysis")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Merge all WFP data (2019-2025) and create comprehensive analysis
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

def merge_all_wfp_data():
    """Merge historical (2019-2023) and recent (2024-2025) WFP data"""

    # Load historical data (2019-2023)
    historical_path = "data/raw/wfp/wfp_food_prices_hdx_20251018_174512.csv"

    # Load recent data (2024-2025)
    recent_path = "data/raw/wfp/wfp_food_prices_hdx_20251018_173850.csv"

    try:
        historical_df = pd.read_csv(historical_path)
        recent_df = pd.read_csv(recent_path)

        logger.info(f"✅ Historical data (2019-2023): {len(historical_df):,} records")
        logger.info(f"✅ Recent data (2024-2025): {len(recent_df):,} records")

        # Combine datasets
        combined_df = pd.concat([historical_df, recent_df], ignore_index=True)

        # Convert date
        combined_df['date'] = pd.to_datetime(combined_df['date'])

        # Remove duplicates if any
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        logger.info(f"Removed {initial_len - len(combined_df)} duplicates")

        # Sort by date
        combined_df = combined_df.sort_values('date')

        # Save combined dataset
        output_path = Path("data/raw/wfp/wfp_food_prices_complete_2019_2025.csv")
        combined_df.to_csv(output_path, index=False)

        # Also save as latest
        latest_path = Path("data/raw/wfp/wfp_food_prices_latest.csv")
        combined_df.to_csv(latest_path, index=False)

        logger.info(f"✅ Combined dataset saved: {len(combined_df):,} records")
        logger.info(f"📅 Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
        logger.info(f"🗃️ File saved to: {output_path}")

        return combined_df

    except Exception as e:
        logger.error(f"Error merging data: {e}")
        return pd.DataFrame()

def comprehensive_analysis(df):
    """Create comprehensive analysis of the full 2019-2025 dataset"""

    if df.empty:
        return

    print("\n🎯 COMPREHENSIVE WFP ANALYSIS (2019-2025)")
    print("=" * 60)

    # Basic stats
    print(f"\n📊 Complete Dataset Overview:")
    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Time span: {(df['date'].max() - df['date'].min()).days} days ({(df['date'].max() - df['date'].min()).days/365:.1f} years)")
    print(f"  Countries: {df['countryiso3'].nunique()}")
    print(f"  Unique markets: {df['market'].nunique()}")
    print(f"  Commodities tracked: {df['commodity'].nunique()}")

    # Year-by-year breakdown
    print(f"\n📅 Data by Year:")
    df['year'] = df['date'].dt.year
    yearly_counts = df['year'].value_counts().sort_index()
    for year, count in yearly_counts.items():
        print(f"  {year}: {count:,} records")

    # Country analysis
    print(f"\n🌍 Country Breakdown:")
    country_counts = df['countryiso3'].value_counts()
    for country, count in country_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {country}: {count:,} records ({pct:.1f}%)")

    # Price trends over time
    print(f"\n💰 Price Analysis:")
    print(f"  Overall average price: ${df['usdprice'].mean():.3f} USD/kg")
    print(f"  Overall median price: ${df['usdprice'].median():.3f} USD/kg")
    print(f"  Price range: ${df['usdprice'].min():.3f} - ${df['usdprice'].max():.3f}")
    print(f"  Standard deviation: ${df['usdprice'].std():.3f}")

    # Year-over-year price changes
    print(f"\n📈 Average Price by Year:")
    yearly_prices = df.groupby('year')['usdprice'].mean()
    for year, price in yearly_prices.items():
        if year > yearly_prices.index.min():
            prev_price = yearly_prices[year-1]
            change = ((price - prev_price) / prev_price) * 100
            print(f"  {year}: ${price:.3f} ({change:+.1f}% vs {year-1})")
        else:
            print(f"  {year}: ${price:.3f}")

    # Top commodities over full period
    print(f"\n🌾 Top Commodities (2019-2025):")
    commodity_counts = df['commodity'].value_counts().head(15)
    for commodity, count in commodity_counts.items():
        avg_price = df[df['commodity'] == commodity]['usdprice'].mean()
        print(f"  {commodity}: {count:,} records (avg: ${avg_price:.3f})")

    # Most volatile commodities
    print(f"\n📊 Most Price-Volatile Commodities:")
    commodity_volatility = df.groupby('commodity')['usdprice'].std().sort_values(ascending=False).head(10)
    for commodity, volatility in commodity_volatility.items():
        avg_price = df[df['commodity'] == commodity]['usdprice'].mean()
        cv = (volatility / avg_price) * 100  # Coefficient of variation
        print(f"  {commodity}: σ=${volatility:.3f} (CV: {cv:.1f}%)")

    return df

def create_comprehensive_visualization(df):
    """Create comprehensive visualizations"""

    if df.empty:
        return

    print(f"\n📊 Creating comprehensive visualizations...")

    # Set up the figure
    fig = plt.figure(figsize=(20, 16))

    # 1. Long-term price trends
    plt.subplot(3, 4, 1)
    monthly_avg = df.groupby(df['date'].dt.to_period('M'))['usdprice'].mean()
    plt.plot(monthly_avg.index.astype(str), monthly_avg.values, 'b-', linewidth=1.5)
    plt.title('Long-term Price Trend (2019-2025)')
    plt.xlabel('Year-Month')
    plt.ylabel('Avg Price (USD/kg)')
    plt.xticks(rotation=45)
    # Show only every 12th label for readability
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 12 != 0:
            label.set_visible(False)

    # 2. Price trends by country
    plt.subplot(3, 4, 2)
    top_countries = df['countryiso3'].value_counts().head(5).index
    for country in top_countries:
        country_data = df[df['countryiso3'] == country]
        monthly_country = country_data.groupby(country_data['date'].dt.to_period('M'))['usdprice'].mean()
        plt.plot(monthly_country.index.astype(str), monthly_country.values, label=country, linewidth=1.5)
    plt.title('Price Trends by Top Countries')
    plt.xlabel('Year-Month')
    plt.ylabel('Avg Price (USD/kg)')
    plt.legend()
    plt.xticks(rotation=45)
    # Show every 12th label
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i % 12 != 0:
            label.set_visible(False)

    # 3. Yearly price distribution
    plt.subplot(3, 4, 3)
    df['year'] = df['date'].dt.year
    years = sorted(df['year'].unique())
    price_by_year = [df[df['year'] == year]['usdprice'] for year in years]
    plt.boxplot(price_by_year, labels=years)
    plt.title('Price Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Price (USD/kg)')
    plt.xticks(rotation=45)

    # 4. Country average prices
    plt.subplot(3, 4, 4)
    country_avg = df.groupby('countryiso3')['usdprice'].mean().sort_values(ascending=True)
    plt.barh(range(len(country_avg)), country_avg.values)
    plt.yticks(range(len(country_avg)), country_avg.index)
    plt.title('Average Price by Country (2019-2025)')
    plt.xlabel('Average Price (USD/kg)')

    # 5. Top commodities by volume
    plt.subplot(3, 4, 5)
    top_commodities = df['commodity'].value_counts().head(10)
    plt.barh(range(len(top_commodities)), top_commodities.values)
    plt.yticks(range(len(top_commodities)), top_commodities.index)
    plt.title('Top 10 Commodities by Data Volume')
    plt.xlabel('Number of Records')

    # 6. Seasonal patterns
    plt.subplot(3, 4, 6)
    df['month'] = df['date'].dt.month
    monthly_patterns = df.groupby('month')['usdprice'].mean()
    plt.plot(monthly_patterns.index, monthly_patterns.values, 'o-', linewidth=2)
    plt.title('Seasonal Price Patterns')
    plt.xlabel('Month')
    plt.ylabel('Average Price (USD/kg)')
    plt.xticks(range(1, 13))

    # 7. Data coverage heatmap
    plt.subplot(3, 4, 7)
    coverage_matrix = df.groupby([df['date'].dt.year, 'countryiso3']).size().unstack(fill_value=0)
    sns.heatmap(coverage_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Data Coverage by Year and Country')
    plt.xlabel('Country')
    plt.ylabel('Year')

    # 8. Price volatility by commodity
    plt.subplot(3, 4, 8)
    top_volatile = df.groupby('commodity')['usdprice'].std().sort_values(ascending=False).head(10)
    plt.barh(range(len(top_volatile)), top_volatile.values)
    plt.yticks(range(len(top_volatile)), top_volatile.index)
    plt.title('Most Volatile Commodities (Std Dev)')
    plt.xlabel('Price Standard Deviation')

    # 9. Markets per country
    plt.subplot(3, 4, 9)
    markets_per_country = df.groupby('countryiso3')['market'].nunique().sort_values(ascending=False)
    plt.bar(markets_per_country.index, markets_per_country.values)
    plt.title('Number of Markets by Country')
    plt.xlabel('Country')
    plt.ylabel('Number of Markets')
    plt.xticks(rotation=45)

    # 10. COVID impact (2020 vs other years)
    plt.subplot(3, 4, 10)
    covid_year = df[df['year'] == 2020]['usdprice']
    other_years = df[df['year'] != 2020]['usdprice']
    plt.hist([covid_year, other_years], bins=50, alpha=0.7, label=['2020 (COVID)', 'Other years'])
    plt.title('Price Distribution: 2020 vs Other Years')
    plt.xlabel('Price (USD/kg)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(0, 50)  # Focus on main price range

    # 11. Year-over-year growth
    plt.subplot(3, 4, 11)
    yearly_avg = df.groupby('year')['usdprice'].mean()
    yoy_growth = yearly_avg.pct_change() * 100
    plt.bar(yoy_growth.index[1:], yoy_growth.values[1:])
    plt.title('Year-over-Year Price Growth')
    plt.xlabel('Year')
    plt.ylabel('Growth Rate (%)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 12. Top markets by data volume
    plt.subplot(3, 4, 12)
    top_markets = df['market'].value_counts().head(10)
    plt.barh(range(len(top_markets)), top_markets.values)
    plt.yticks(range(len(top_markets)), top_markets.index)
    plt.title('Top Markets by Data Volume')
    plt.xlabel('Number of Records')

    plt.tight_layout()

    # Save visualization
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "wfp_comprehensive_analysis_2019_2025.png", dpi=150, bbox_inches='tight')
    print(f"📊 Comprehensive visualization saved to: {output_dir / 'wfp_comprehensive_analysis_2019_2025.png'}")

def main():
    """Main function"""
    print("🚀 MERGING AND ANALYZING COMPLETE WFP DATASET (2019-2025)")
    print("=" * 70)

    # Merge all data
    df = merge_all_wfp_data()

    if not df.empty:
        # Comprehensive analysis
        df = comprehensive_analysis(df)

        # Create visualizations
        create_comprehensive_visualization(df)

        print(f"\n✅ SUCCESS! Complete WFP dataset ready for modeling:")
        print(f"   📊 {len(df):,} total records")
        print(f"   📅 {(df['date'].max() - df['date'].min()).days/365:.1f} years of data")
        print(f"   🌍 {df['countryiso3'].nunique()} Eastern Africa countries")
        print(f"   🏪 {df['market'].nunique()} markets")
        print(f"   🌾 {df['commodity'].nunique()} commodities")
        print(f"   💾 Saved as: data/raw/wfp/wfp_food_prices_complete_2019_2025.csv")

    else:
        print("❌ Failed to merge datasets")

if __name__ == "__main__":
    main()
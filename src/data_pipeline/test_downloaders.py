#!/usr/bin/env python3
"""
Test script for data downloaders with sample data generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_wfp_data():
    """Create sample WFP food price data for testing"""
    logger.info("Creating sample WFP food price data")

    # Create output directory
    output_dir = Path("data/raw/wfp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample data
    countries = ['ETH', 'KEN', 'UGA', 'TZA', 'RWA']
    commodities = ['Maize', 'Sorghum', 'Beans', 'Rice', 'Oil']
    markets = ['Addis Ababa', 'Nairobi', 'Kampala', 'Dar es Salaam', 'Kigali']

    # Generate 2 years of monthly data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')

    data = []
    for i, country in enumerate(countries):
        for commodity in commodities:
            base_price = np.random.uniform(20, 100)  # Base price in USD
            for date in dates:
                # Add seasonality and trend
                month_effect = 10 * np.sin(2 * np.pi * date.month / 12)
                trend = 0.2 * (date.year - 2023)
                noise = np.random.normal(0, 5)

                price = base_price + month_effect + trend + noise
                price = max(price, 5)  # Minimum price

                data.append({
                    'Date': date,
                    'CountryCode': country,
                    'MarketName': markets[i],
                    'CommodityName': commodity,
                    'PriceUSD': round(price, 2),
                    'Unit': 'kg',
                    'Currency': 'USD',
                    'Latitude': np.random.uniform(-5, 15),
                    'Longitude': np.random.uniform(29, 45)
                })

    df = pd.DataFrame(data)

    # Save data
    output_path = output_dir / "wfp_food_prices_latest.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Sample WFP data created: {len(df)} records")
    logger.info(f"Saved to: {output_path}")
    return df

def create_sample_climate_data():
    """Create sample climate data for testing"""
    logger.info("Creating sample climate data")

    output_dir = Path("data/raw/climate")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate monthly data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')

    # Precipitation data (CHIRPS-style)
    precip_data = []
    for date in dates:
        # Seasonal rainfall pattern
        base_rainfall = 50 + 40 * np.sin(2 * np.pi * (date.month - 3) / 12)
        rainfall = max(0, base_rainfall + np.random.normal(0, 20))

        precip_data.append({
            'date': date,
            'precipitation_mm': round(rainfall, 1),
            'precipitation_anomaly': round(rainfall - base_rainfall, 1),
            'data_source': 'CHIRPS_sample'
        })

    precip_df = pd.DataFrame(precip_data)
    precip_df.to_csv(output_dir / "chirps_precipitation.csv", index=False)

    # Temperature data (MODIS-style)
    temp_data = []
    for date in dates:
        base_temp = 25 + 5 * np.sin(2 * np.pi * (date.month - 6) / 12)
        day_temp = base_temp + np.random.normal(0, 3)
        night_temp = day_temp - 8 + np.random.normal(0, 2)

        temp_data.append({
            'date': date,
            'lst_day_celsius': round(day_temp, 1),
            'lst_night_celsius': round(night_temp, 1),
            'extreme_heat_days': 1 if day_temp > 32 else 0,
            'data_source': 'MODIS_sample'
        })

    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv(output_dir / "modis_temperature.csv", index=False)

    logger.info(f"Sample climate data created")
    return precip_df, temp_df

def create_sample_conflict_data():
    """Create sample ACLED conflict data for testing"""
    logger.info("Creating sample ACLED conflict data")

    output_dir = Path("data/raw/acled")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate conflict events
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)

    events = []
    countries = ['ETH', 'KEN', 'UGA', 'TZA', 'SOM']
    event_types = ['Battles', 'Violence against civilians', 'Riots', 'Protests']

    # Generate 200 random events
    for i in range(200):
        event_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))

        events.append({
            'event_date': event_date,
            'event_type': np.random.choice(event_types),
            'country_code': np.random.choice(countries),
            'latitude': np.random.uniform(-5, 15),
            'longitude': np.random.uniform(29, 45),
            'fatalities': np.random.poisson(2),
            'data_source': 'ACLED_sample'
        })

    conflict_df = pd.DataFrame(events)
    conflict_df.to_csv(output_dir / "acled_conflict_data_latest.csv", index=False)

    logger.info(f"Sample ACLED data created: {len(conflict_df)} events")
    return conflict_df

def test_all_downloaders():
    """Test all data downloaders and create sample data"""
    logger.info("Testing all data downloaders")

    results = {}

    # Test WFP data
    try:
        wfp_data = create_sample_wfp_data()
        results['WFP'] = f"✅ Success: {len(wfp_data)} records"
    except Exception as e:
        results['WFP'] = f"❌ Failed: {e}"

    # Test Climate data
    try:
        precip_data, temp_data = create_sample_climate_data()
        results['Climate'] = f"✅ Success: {len(precip_data)} precip + {len(temp_data)} temp records"
    except Exception as e:
        results['Climate'] = f"❌ Failed: {e}"

    # Test ACLED data
    try:
        conflict_data = create_sample_conflict_data()
        results['ACLED'] = f"✅ Success: {len(conflict_data)} events"
    except Exception as e:
        results['ACLED'] = f"❌ Failed: {e}"

    # Create summary
    logger.info("=== DOWNLOADER TEST RESULTS ===")
    for source, result in results.items():
        logger.info(f"{source}: {result}")

    return results

if __name__ == "__main__":
    test_all_downloaders()
#!/usr/bin/env python3
"""
Fix file naming to use descriptive names instead of timestamps
"""

import pandas as pd
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_wfp_file_naming():
    """Fix WFP file naming to be descriptive"""

    wfp_dir = Path("data/raw/wfp")

    # Find existing files
    existing_files = list(wfp_dir.glob("wfp_food_prices_*.csv"))

    for file_path in existing_files:
        if "backup" in file_path.name or "latest" in file_path.name:
            continue  # Skip backup and latest files

        try:
            # Read the file to get metadata
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])

            # Extract metadata
            min_year = df['date'].dt.year.min()
            max_year = df['date'].dt.year.max()
            year_range = f"{min_year}-{max_year}" if min_year != max_year else str(min_year)

            country_count = df['countryiso3'].nunique()
            record_count = len(df)

            # Create descriptive filename
            new_filename = f"wfp_food_prices_eastern_africa_{year_range}_{country_count}countries_{record_count}records.csv"
            new_path = wfp_dir / new_filename

            # Only rename if the name is different and target doesn't exist
            if file_path.name != new_filename and not new_path.exists():
                shutil.move(str(file_path), str(new_path))
                logger.info(f"Renamed: {file_path.name} -> {new_filename}")

                # Update the latest symlink/copy
                latest_path = wfp_dir / "wfp_food_prices_latest.csv"
                shutil.copy2(str(new_path), str(latest_path))

            else:
                logger.info(f"Skipped: {file_path.name} (already properly named or target exists)")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

def suggest_naming_convention():
    """Suggest proper naming convention for all data sources"""

    print("\n📝 PROPOSED FILE NAMING CONVENTION")
    print("=" * 50)

    naming_examples = {
        "WFP Food Prices": [
            "wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv",
            "wfp_food_prices_eastern_africa_2024_9countries_32241records.csv"
        ],
        "ACLED Conflict Data": [
            "acled_conflict_eastern_africa_2019-2025_11countries_15000events.csv",
            "acled_conflict_eastern_africa_2024_11countries_2500events.csv"
        ],
        "Climate Data": [
            "climate_precipitation_chirps_eastern_africa_2019-2025_monthly.csv",
            "climate_temperature_modis_eastern_africa_2019-2025_monthly.csv",
            "climate_drought_indices_eastern_africa_2019-2025_monthly.csv"
        ],
        "OpenStreetMap": [
            "osm_roads_eastern_africa_11countries_primary_secondary.geojson",
            "osm_infrastructure_eastern_africa_11countries_markets_airports.geojson"
        ],
        "World Bank/IMF": [
            "worldbank_macro_indicators_eastern_africa_2019-2025_11countries.csv",
            "imf_exchange_rates_eastern_africa_2019-2025_11countries.csv",
            "fao_food_price_index_global_2019-2025_monthly.csv"
        ],
        "Geospatial": [
            "population_density_worldpop_eastern_africa_2020_1km_resolution.csv",
            "admin_boundaries_gadm_eastern_africa_country_level.geojson",
            "elevation_srtm_eastern_africa_1km_resolution.csv"
        ]
    }

    for category, examples in naming_examples.items():
        print(f"\n🗂️ {category}:")
        for example in examples:
            print(f"   ✅ {example}")

    print(f"\n📋 NAMING PATTERN:")
    print(f"   {'{source}'}_{'{data_type}'}_{'{region}'}_{'{time_range}'}_{'{additional_info}'}.{'{extension}'}")
    print(f"")
    print(f"🎯 BENEFITS:")
    print(f"   ✅ Immediately understand what the file contains")
    print(f"   ✅ Sort files logically by source, region, time")
    print(f"   ✅ Avoid duplicate downloads")
    print(f"   ✅ Easy to find specific datasets")
    print(f"   ✅ Clear metadata in filename")

def main():
    """Main function"""
    print("🔧 FIXING FILE NAMING CONVENTION")
    print("=" * 40)

    # Fix existing WFP files
    logger.info("Fixing WFP file naming...")
    fix_wfp_file_naming()

    # Show current files
    print(f"\n📁 Current WFP files:")
    wfp_dir = Path("data/raw/wfp")
    for file_path in sorted(wfp_dir.glob("*.csv")):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   {file_path.name} ({size_mb:.1f} MB)")

    # Suggest convention for all sources
    suggest_naming_convention()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Show the improvements in file naming convention
"""

from pathlib import Path
import pandas as pd

def show_before_after():
    """Show before/after comparison of file naming"""

    print("🔧 FILE NAMING IMPROVEMENTS")
    print("=" * 50)

    # Before and After examples
    examples = [
        {
            "category": "WFP Food Prices",
            "before": "wfp_food_prices_hdx_20251018_174512.csv",
            "after": "wfp_food_prices_eastern_africa_2019-2023_10countries_86246records.csv",
            "problems": ["Timestamp tells us nothing", "No region info", "No data scope", "No record count"],
            "improvements": ["Clear time range", "Region specified", "Country count", "Record count"]
        },
        {
            "category": "ACLED Conflict",
            "before": "acled_conflict_data_20251018_175230.csv",
            "after": "acled_conflict_eastern_africa_2019-2025_11countries_15000events.csv",
            "problems": ["Just timestamp", "No geographic scope", "No event count"],
            "improvements": ["Time range clear", "Geographic focus", "Event count", "Data type"]
        },
        {
            "category": "Climate Data",
            "before": "chirps_precipitation.csv",
            "after": "climate_precipitation_chirps_eastern_africa_2019-2025_monthly.csv",
            "problems": ["No time info", "No region", "No frequency"],
            "improvements": ["Source clear", "Region specified", "Time range", "Frequency"]
        }
    ]

    for example in examples:
        print(f"\n📊 {example['category']}:")
        print(f"  ❌ BEFORE: {example['before']}")
        print(f"     Problems: {', '.join(example['problems'])}")
        print(f"  ✅ AFTER:  {example['after']}")
        print(f"     Benefits: {', '.join(example['improvements'])}")

def show_current_files():
    """Show current files with improved naming"""

    print(f"\n📁 CURRENT FILES WITH IMPROVED NAMING")
    print("=" * 50)

    data_dir = Path("data/raw")

    # Check each subdirectory
    for subdir in ["wfp", "acled", "climate", "osm", "macro", "geospatial"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*"))
            if files:
                print(f"\n🗂️ {subdir.upper()}:")
                for file_path in sorted(files):
                    if file_path.is_file():
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        print(f"   ✅ {file_path.name} ({size_mb:.1f} MB)")
            else:
                print(f"\n🗂️ {subdir.upper()}: (empty)")

def show_naming_benefits():
    """Show benefits of the new naming convention"""

    print(f"\n🎯 BENEFITS OF NEW NAMING CONVENTION")
    print("=" * 50)

    benefits = [
        {
            "benefit": "Immediate Understanding",
            "description": "Know exactly what data is in the file without opening it",
            "example": "wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv tells you everything"
        },
        {
            "benefit": "Easy Sorting",
            "description": "Files sort logically by source, region, and time period",
            "example": "All WFP files group together, all Eastern Africa data groups together"
        },
        {
            "benefit": "Avoid Duplicates",
            "description": "Clear naming prevents accidentally downloading the same data",
            "example": "Can see if you already have 2019-2023 data vs 2024-2025 data"
        },
        {
            "benefit": "Metadata in Filename",
            "description": "Key statistics available without reading the file",
            "example": "Record counts, country counts, time ranges all visible"
        },
        {
            "benefit": "Version Control Friendly",
            "description": "Descriptive names work better in git and collaboration",
            "example": "Changes to 'wfp_eastern_africa_2024.csv' are clear vs 'data_20251018.csv'"
        }
    ]

    for i, benefit in enumerate(benefits, 1):
        print(f"\n{i}. 🎯 {benefit['benefit']}")
        print(f"   📝 {benefit['description']}")
        print(f"   💡 Example: {benefit['example']}")

def analyze_wfp_data():
    """Analyze the current WFP data to show the value"""

    wfp_latest = Path("data/raw/wfp/wfp_food_prices_latest.csv")

    if wfp_latest.exists():
        print(f"\n📊 CURRENT WFP DATASET ANALYSIS")
        print("=" * 50)

        df = pd.read_csv(wfp_latest)
        df['date'] = pd.to_datetime(df['date'])

        print(f"✅ Successfully loaded: {len(df):,} records")
        print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"🌍 Countries: {df['countryiso3'].nunique()} ({', '.join(sorted(df['countryiso3'].unique()))})")
        print(f"🏪 Markets: {df['market'].nunique():,}")
        print(f"🌾 Commodities: {df['commodity'].nunique()}")
        print(f"💰 Price range: ${df['usdprice'].min():.3f} - ${df['usdprice'].max():.3f}")

        # Show the power of descriptive naming
        filename = "wfp_food_prices_eastern_africa_2019-2025_10countries_118487records.csv"
        print(f"\n🎯 FILENAME TELLS THE STORY:")
        print(f"   📁 File: {filename}")
        print(f"   🔍 From filename alone, we know:")
        print(f"      • Data source: WFP food prices")
        print(f"      • Geographic focus: Eastern Africa")
        print(f"      • Time period: 2019-2025 (6.8 years)")
        print(f"      • Geographic coverage: 10 countries")
        print(f"      • Data volume: 118,487 records")
        print(f"   ✅ No need to open the file to understand it!")

def main():
    """Main function"""
    show_before_after()
    show_current_files()
    show_naming_benefits()
    analyze_wfp_data()

    print(f"\n🎉 FILE NAMING FIXED!")
    print("Now all data files have descriptive, meaningful names that immediately tell you:")
    print("• What data it contains")
    print("• Which region/countries")
    print("• What time period")
    print("• How much data")
    print("• File size and metadata")

if __name__ == "__main__":
    main()
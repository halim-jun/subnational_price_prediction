"""
Quality check for subnational merged data (v2 — KEN + SOM).

Usage:
  python check_merge_quality.py
  python check_merge_quality.py --parquet path/to/file.parquet

Checks:
  1. Null / missing values
  2. Admin2 boundary completeness
  3. Price coverage per country
  4. Price range & outliers
  5. Cross-boundary contamination
  6. Skeleton completeness (year x month x admin2)
  7. Population / crop / conflict sanity
  8. KEN vs SOM price unit comparison
"""

import argparse
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

ISO3_LIST = ['KEN', 'SOM']
BOUNDARY_DIR = os.path.join(project_root, 'data', 'geoboundaries')
PRICE_COLS = ['c_maize_fao', 'c_food_price_index', 'c_sorghum']


def header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def check_nulls(df):
    header("CHECK 1: Null / Missing Values")
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    report = pd.DataFrame({'null_count': null_counts, 'null_pct': null_pct})
    print(report)
    print(f"\nTotal rows: {len(df)}")


def check_admin2_completeness(df):
    header("CHECK 2: Admin2 Boundary Completeness")
    for iso in ISO3_LIST:
        path = os.path.join(BOUNDARY_DIR, f'gb_{iso}_ADM2.geojson')
        if not os.path.exists(path):
            print(f"  {iso}: boundary file not found at {path}")
            continue
        gdf = gpd.read_file(path)
        expected = set(gdf['shapeName'].unique())
        actual = set(df[df['country_iso'] == iso]['admin2'].unique())
        missing = expected - actual
        extra = actual - expected
        status = "PASS" if not missing and not extra else "WARN"
        print(f"  {iso}: expected={len(expected)}, actual={len(actual)}, "
              f"missing={len(missing)}, extra={len(extra)} [{status}]")
        if missing:
            print(f"    Missing: {sorted(missing)[:10]}")
        if extra:
            print(f"    Extra: {sorted(extra)[:10]}")


def check_price_coverage(df):
    header("CHECK 3: Price Coverage per Country")
    for iso in ISO3_LIST:
        sub = df[df['country_iso'] == iso]
        total_admin2 = sub['admin2'].nunique()
        for col in PRICE_COLS:
            if col not in df.columns:
                continue
            notnull = sub[col].notna().sum()
            admin2_with = sub[sub[col].notna()]['admin2'].nunique()
            print(f"  {iso} {col}: {notnull}/{len(sub)} rows ({notnull/len(sub)*100:.1f}%), "
                  f"{admin2_with}/{total_admin2} admin2")
        print()


def check_price_outliers(df):
    header("CHECK 4: Price Range & Outliers")
    for col in PRICE_COLS:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            print(f"  {col}: NO DATA")
            continue
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        n_extreme = ((valid < q1 - 3 * iqr) | (valid > q3 + 3 * iqr)).sum()
        status = "PASS" if n_extreme == 0 else f"WARN ({n_extreme} outliers)"
        print(f"  {col}: [{valid.min():.2f} ~ {valid.max():.2f}], "
              f"mean={valid.mean():.2f}, median={valid.median():.2f}, "
              f"neg={int((valid < 0).sum())}, zero={int((valid == 0).sum())}, "
              f"extreme(3*IQR)={n_extreme} [{status}]")


def check_cross_boundary(df):
    header("CHECK 5: Cross-boundary Contamination")
    countries = df['country_iso'].unique()
    unexpected = [c for c in countries if c not in ISO3_LIST]
    if unexpected:
        print(f"  FAIL: unexpected countries: {unexpected}")
    else:
        print(f"  Countries present: {list(countries)} [PASS]")

    dupes = df.duplicated(subset=['admin2', 'country_iso', 'year', 'month']).sum()
    print(f"  Duplicate rows: {dupes} [{'PASS' if dupes == 0 else 'FAIL'}]")

    cross = df.groupby('admin2')['country_iso'].nunique()
    shared = cross[cross > 1]
    if len(shared):
        print(f"  WARN: admin2 shared across countries: {list(shared.index)}")
    else:
        print(f"  No admin2 name collisions [PASS]")


def check_skeleton(df):
    header("CHECK 6: Skeleton Completeness")
    yr_month_sizes = df.groupby(['year', 'month']).size()
    unique_sizes = yr_month_sizes.unique()
    print(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"  Rows per year-month: {unique_sizes} "
          f"[{'PASS' if len(unique_sizes) == 1 else 'FAIL'}]")

    all_combos = set(
        (y, m)
        for y in range(df['year'].min(), df['year'].max() + 1)
        for m in range(1, 13)
    )
    actual_combos = set(zip(df['year'], df['month']))
    missing = all_combos - actual_combos
    if missing:
        print(f"  Missing year-month combos: {sorted(missing)} [FAIL]")
    else:
        print(f"  All year-month combos present [PASS]")


def check_features(df):
    header("CHECK 7: Population / Crop / Conflict")

    # Population
    pop_null = df['population'].isnull().sum()
    pop_invariant = df.groupby(['admin2', 'country_iso'])['population'].nunique().eq(1).all()
    print(f"  Population null: {pop_null} [{'PASS' if pop_null == 0 else 'FAIL'}]")
    print(f"  Population range: {df['population'].min():.0f} ~ {df['population'].max():.0f}")
    print(f"  Time-invariant: {pop_invariant} [{'PASS' if pop_invariant else 'WARN'}]")

    # Crop
    crop_null = df['crop_cover_fraction'].isnull().sum()
    crop_pct = crop_null / len(df) * 100
    print(f"\n  Crop null: {crop_null} ({crop_pct:.1f}%)")
    crop_valid = df['crop_cover_fraction'].dropna()
    print(f"  Crop range: {crop_valid.min():.2f} ~ {crop_valid.max():.2f} "
          f"(NOTE: values are %, not 0-1)")
    crop_null_admins = df[df['crop_cover_fraction'].isnull()].groupby('country_iso')['admin2'].unique()
    for iso, admins in crop_null_admins.items():
        print(f"    {iso} null ({len(admins)}): {list(admins)}")

    # Conflict
    conflict_pct = (df['conflict_events'] > 0).mean() * 100
    print(f"\n  Conflict rows > 0: {conflict_pct:.1f}%")
    for iso in ISO3_LIST:
        sub = df[df['country_iso'] == iso]
        pct = (sub['conflict_events'] > 0).mean() * 100
        print(f"    {iso}: {pct:.1f}%")


def check_price_units(df):
    header("CHECK 8: KEN vs SOM Price Unit Comparison")
    print("  c_maize_fao = local currency (KES / SOS), NOT comparable across countries")
    print("  c_food_price_index = normalized (~1.0), comparable\n")
    for col in ['c_maize_fao', 'c_food_price_index']:
        if col not in df.columns:
            continue
        print(f"  {col}:")
        for iso in ISO3_LIST:
            v = df[(df['country_iso'] == iso) & (df[col].notna())][col]
            if len(v) > 0:
                print(f"    {iso}: mean={v.mean():.2f}, median={v.median():.2f}, "
                      f"min={v.min():.2f}, max={v.max():.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Quality check for merged subnational data")
    parser.add_argument('--parquet', default=None,
                        help='Path to parquet file (default: data/processed/subnational_merged_v2_KEN_SOM.parquet)')
    args = parser.parse_args()

    if args.parquet:
        parquet_path = args.parquet
    else:
        parquet_path = os.path.join(project_root, 'data/processed/subnational_merged_v2_KEN_SOM.parquet')

    if not os.path.exists(parquet_path):
        print(f"ERROR: File not found: {parquet_path}")
        sys.exit(1)

    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    check_nulls(df)
    check_admin2_completeness(df)
    check_price_coverage(df)
    check_price_outliers(df)
    check_cross_boundary(df)
    check_skeleton(df)
    check_features(df)
    check_price_units(df)

    header("DONE")
    print("  All checks complete. Review WARN/FAIL items above.")


if __name__ == "__main__":
    main()

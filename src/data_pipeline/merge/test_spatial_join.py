
import pandas as pd
import geopandas as gpd
import sys
import os
from shapely.geometry import Point

# Add project root to path
# This script is in src/data_pipeline/merge/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.data_pipeline.merge import subnational_merger

def test_spatial_join_price():
    print("Initializing Spatial Join Test...")
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load Canonical Boundaries (Kenya only)
    boundary_dir = os.path.join(data_dir, 'geoboundaries')
    try:
        iso3_list = ['KEN']
        admin_gdf = subnational_merger.load_canonical_boundaries(iso3_list, boundary_dir)
        print(f"Loaded {len(admin_gdf)} KEN regions.")
    except Exception as e:
        print(f"Error loading boundaries: {e}")
        return

    # 2. Load Price Data Sample (Kenya)
    csv_path = os.path.join(data_dir, "worldbank_imputed_price_data/WLD_RTFP_mkt_2026-01-13.csv")
    if not os.path.exists(csv_path):
        print("Price file missing.")
        return
        
    use_cols = ['country', 'adm2_name', 'lat', 'lon']
    df = pd.read_csv(csv_path, usecols=use_cols)
    ken_prices = df[df['country'] == 'Kenya'].copy()
    
    # Get unique markets with coordinates
    markets = ken_prices[['adm2_name', 'lat', 'lon']].drop_duplicates()
    print(f"Found {len(markets)} unique market locations in Kenya.")
    
    # Filter for problematic cities
    cities = ['Nairobi', 'Mombasa', 'Garissa', 'Kisumu']
    problem_markets = markets[markets['adm2_name'].isin(cities)]
    print("\nProblematic Markets Coordinates:")
    print(problem_markets)
    
    # 3. Perform Spatial Join
    print("\nPerforming Spatial Join...")
    geometry = [Point(xy) for xy in zip(markets.lon, markets.lat)]
    markets_gdf = gpd.GeoDataFrame(markets, geometry=geometry, crs="EPSG:4326")
    
    # Ensure CRS match
    if admin_gdf.crs != markets_gdf.crs:
        admin_gdf = admin_gdf.to_crs(markets_gdf.crs)
        
    joined = gpd.sjoin(markets_gdf, admin_gdf, how="left", predicate="within")
    
    # Check results for cities
    print("\n--- Match Results for Cities ---")
    results = joined[joined['adm2_name'].isin(cities)][['adm2_name', 'shapeName', 'shapeISO']]
    print(results)
    
    # Check coverage
    matched_count = joined['shapeName'].notnull().sum()
    print(f"\nTotal Markets: {len(markets)}")
    print(f"Spatially Matched: {matched_count}")
    print(f"Unmatched: {len(markets) - matched_count}")
    
    if len(markets) - matched_count > 0:
        print("Unmatched Markets (Points outside polygons?):")
        print(joined[joined['shapeName'].isnull()]['adm2_name'].unique())

if __name__ == "__main__":
    test_spatial_join_price()


import pandas as pd
import sys
import os

# Add project root to path
# __file__ is src/data_pipeline/merge/inspect_canonical.py
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.data_pipeline.merge import subnational_merger

def inspect_canonical_names():
    print("Initializing Canonical Inspection...")
    data_dir = os.path.join(project_root, 'data')
    boundary_dir = os.path.join(data_dir, 'geoboundaries')
    
    iso3_list = ['KEN', 'SOM', 'ETH']
    
    try:
        admin_gdf = subnational_merger.load_canonical_boundaries(iso3_list, boundary_dir)
        print(f"Loaded {len(admin_gdf)} total regions.")
        print("Columns:", admin_gdf.columns.tolist())
        print("Unique ShapeISOs:", admin_gdf['shapeISO'].unique())
        print("Head:", admin_gdf.head())
        
        for iso in iso3_list:
            subset = admin_gdf[admin_gdf['shapeISO'] == iso]
            print(f"\n--- {iso} Canonical Admins ({len(subset)}) ---")
            names = sorted(subset['shapeName'].unique().tolist())
            print(names[:20]) # Print first 20
            
            # Specific checks for problem names
            if iso == 'KEN':
                print("Checking for 'Nairobi':")
                matches = [x for x in names if 'Nairobi' in str(x)]
                print(f"Contains 'Nairobi': {matches}")
                
            if iso == 'SOM':
                print("Checking for 'Banadir':")
                matches = [x for x in names if 'Banadir' in str(x)]
                print(f"Contains 'Banadir': {matches}")

            if iso == 'ETH':
                print("Checking for 'Addis':")
                matches = [x for x in names if 'Addis' in str(x)]
                print(f"Contains 'Addis': {matches}")

    except Exception as e:
        print(f"Error loading boundaries: {e}")

if __name__ == "__main__":
    inspect_canonical_names()

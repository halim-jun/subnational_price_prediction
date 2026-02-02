#!/usr/bin/env python
"""
Dynamic World Data Downloader

This script authenticates with Google Earth Engine (GEE) and downloads 
Dynamic World (LULC) data for the East Africa region.

Usage:
    python download_dynamic_world.py
"""

import ee
import os
import argparse
from datetime import datetime
# import geemap  # Removed unused dependency
               # Actually, for batch export to drive/download URL, we can use ee directly.
               # Let's stick to standard ee for now and allow direct URL download if possible,
               # or export to Drive which is standard for GEE.
               # However, the user request implies getting data "down" to the machine.
               # A common pattern is `geemap.ee_export_image` or `ee.batch.Export.image.toDrive`.
               # Given this is a local script, `geemap` is very helpful for direct download.
               # But let's check if geemap is allowed/requested. 
               # The plan didn't specify geemap, just earthengine-api.
               # We can use `ee.Image.getDownloadURL()` for small regions, but East Africa is large.
               # East Africa is HUGE. 
               # Lon [25, 52], Lat [-15, 22].
               # This is too big for a single `getDownloadURL`. 
               # We likely need to composite or export to Drive.
               # Let's try to export to Drive or Cloud Storage. 
               # BUT, the user wants to "download" it.
               # Let's assume we export to Drive and then they can download it, OR 
               # use `geemap` if I can add it, but I added `earthengine-api`.
               # Let's stick to `earthengine-api` and maybe print instructions or 
               # try to download small chunks if needed. 
               # Actually, for a "project", usually we want a composite (e.g. yearly mode).
               # Let's target a yearly mode composite for 2024 (or requested year).
               
import requests
import shutil

def initialize_ee(project_id=None):
    """Initialize Earth Engine API."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("✅ GEE Initialized successfully.")
    except Exception as e:
        print("⚠️  GEE Authorization required or Project ID missing.")
        print("   Please run: `earthengine authenticate` in your terminal.")
        print("   If you have a specific Cloud Project, run with: --project YOUR_PROJECT_ID")
        print(f"   Error: {e}")
        # raising error to stop script
        raise e

def get_east_africa_roi():
    """Define East Africa Region of Interest."""
    # Matches SPI script region: Lon [25, 52], Lat [-15, 22]
    return ee.Geometry.Rectangle([25, -15, 52, 22])

def download_dynamic_world(year=2024, output_dir='data/raw/dynamic_world'):
    """
    Download Dynamic World Mode Composite for the specified year.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    roi = get_east_africa_roi()
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    print(f"\nProcessing Dynamic World data for {year}...")
    print(f"Region: East Africa (Lon 25-52, Lat -15 to 22)")
    
    # Dynamic World V1 collection
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    
    # Filter by bounds and time
    dw_filtered = dw.filterBounds(roi).filterDate(start_date, end_date)
    
    # Create a Mode composite: most frequent class label for the year
    # The 'label' band contains class integers (0-8)
    # 0: water, 1: trees, 2: grass, 3: flooded_vegetation, 4: crops, 
    # 5: shrub_and_scrub, 6: built, 7: bare, 8: snow_and_ice
    
    # Check if collection is empty
    count = dw_filtered.size().getInfo()
    if count == 0:
        print(f"⚠️  No data found for year {year}. Skipping.")
        return

    classification = dw_filtered.select('label').mode().clip(roi)
    
    scale = 5000  # Increased to 5km to avoid memory errors
    print(f"Target Scale: {scale}m")
    
    # Generate Download URL
    params = {
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': roi,
        'format': 'GEO_TIFF'
    }
    
    output_file = os.path.join(output_dir, f'dynamic_world_east_africa_mode_{year}_{scale}m.tif')
    
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return

    print("Requesting download URL from GEE (this may take a moment)...")
    try:
        url = classification.getDownloadURL(params)
        print(f"Downloading from: {url}")
        
        # Download the file
        response = requests.get(url, stream=True)
        
        if response.status_code != 200:
             print(f"❌ Error downloading file. Status Code: {response.status_code}")
             print(f"Response: {response.text}")
             return

        with open(output_file, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
            
        print(f"✅ Download complete: {output_file}")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Note: The requested area might be too large for direct download at this scale.")
        print("Try increasing the scale (e.g., to 5000m) or using 'Export.image.toDrive' instead.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Dynamic World Data')
    parser.add_argument('--project', help='Google Cloud Project ID (required if not set in default config)')
    parser.add_argument('--start-year', type=int, default=2016, help='Start year (default: 2016)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year (default: 2025)')
    args = parser.parse_args()

    initialize_ee(project_id=args.project)
    
    # User requested 2007-2025. 
    # Dynamic World is available from mid-2015. We start from 2016 for full yearly composites.
    print(f"Downloading data from {args.start_year} to {args.end_year}")
    
    for year in range(args.start_year, args.end_year + 1):
        download_dynamic_world(year=year)

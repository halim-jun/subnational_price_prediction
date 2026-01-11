#!/usr/bin/env python3
"""
Convert SPI NetCDF files to CSV format.
This module provides functionality to batch convert NetCDF files from the SPI generation pipeline 
into CSV format for easier downstream analysis.
"""

import os
import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def convert_nc_to_csv(input_dir, output_dir):
    """
    Convert all NetCDF files in input_dir to CSV in output_dir.
    
    Args:
        input_dir (str or Path): Directory containing .nc files
        output_dir (str or Path): Directory to save .csv files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all NetCDF files
    nc_files = list(input_path.glob("*.nc"))
    
    if not nc_files:
        print(f"⚠️  No NetCDF files found in {input_dir}")
        return

    print(f"\n🔄 Converting {len(nc_files)} NetCDF files to CSV...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print()

    count = 0
    for nc_file in tqdm(nc_files, desc="Converting"):
        try:
            # Construct output filename
            csv_filename = nc_file.name.replace('.nc', '.csv')
            csv_file = output_path / csv_filename
            
            # Skip if already exists and is newer? 
            # For now, let's overwrite to ensure consistency with latest run
            
            # Open NetCDF
            ds = xr.open_dataset(nc_file)
            
            # Convert to DataFrame
            # to_dataframe() creates a MultiIndex (usually lat, lon, time)
            df = ds.to_dataframe()
            
            # Reset index to make lat, lon, time ordinary columns
            df = df.reset_index()
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            
            ds.close()
            count += 1
            
        except Exception as e:
            print(f"❌ Error converting {nc_file.name}: {e}")

    print(f"\n✅ Converted {count}/{len(nc_files)} files successfully.")

def main():
    """Main execution entry point"""
    # Default paths based on project structure
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / "data/processed/spi/05_spi_final"
    output_dir = base_dir / "data/processed/spi/06_spi_csv"
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
        
    convert_nc_to_csv(input_dir, output_dir)

if __name__ == "__main__":
    main()

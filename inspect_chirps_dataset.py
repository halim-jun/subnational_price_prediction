#!/usr/bin/env python3
"""
Inspect CHIRPS dataset structure to understand the correct way to access it
"""

import xarray as xr
import warnings
warnings.filterwarnings('ignore')

print("Inspecting CHIRPS dataset structure...")
print("=" * 70)

try:
    # Connect to IRI Data Library
    ds = xr.open_dataset(chirps_url, decode_times=False)
    
    print("\n" + "=" * 70)
    print("DATASET INFO:")
    print("=" * 70)
    print(ds)
    
    print("\n" + "=" * 70)
    print("COORDINATES:")
    print("=" * 70)
    for coord in ds.coords:
        print(f"\n{coord}:")
        print(f"  Shape: {ds[coord].shape}")
        print(f"  Values (first 5): {ds[coord].values[:5]}")
        print(f"  Values (last 5): {ds[coord].values[-5:]}")
    
    print("\n" + "=" * 70)
    print("VARIABLES:")
    print("=" * 70)
    for var in ds.data_vars:
        print(f"\n{var}:")
        print(f"  Shape: {ds[var].shape}")
        print(f"  Attributes: {ds[var].attrs}")
    
    ds.close()
    print("\n✅ Inspection complete!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()


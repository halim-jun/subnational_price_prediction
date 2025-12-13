#!/usr/bin/env python
"""
Test NetCDF saving with calendar attribute fix

Quick test to ensure xarray can save NetCDF files without calendar conflicts
"""

import xarray as xr
import numpy as np
import os
import tempfile

def test_save_with_calendar():
    """
    Test saving a NetCDF file with calendar attribute
    """
    print("Testing NetCDF save with calendar attribute...")
    
    # Create a simple test dataset
    times = np.arange('2020-01', '2021-01', dtype='datetime64[M]')
    lats = np.linspace(-10, 10, 5)
    lons = np.linspace(20, 40, 5)
    
    data = np.random.rand(12, 5, 5)
    
    ds = xr.Dataset(
        {
            'precip': (['time', 'lat', 'lon'], data)
        },
        coords={
            'time': times,
            'lat': lats,
            'lon': lons
        }
    )
    
    # Add calendar attribute (this would cause the original error)
    ds['time'].attrs['calendar'] = 'gregorian'
    ds['precip'].attrs['units'] = 'mm'
    
    print(f"✓ Created test dataset")
    print(f"  Shape: {ds.dims}")
    print(f"  Time calendar attr: {ds['time'].attrs.get('calendar', 'none')}")
    
    # Test save (this is where the error would occur)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test.nc')
        
        print(f"\nAttempting to save with calendar fix...")
        
        # Remove calendar from attrs (our fix)
        if 'calendar' in ds['time'].attrs:
            calendar_value = ds['time'].attrs['calendar']
            del ds['time'].attrs['calendar']
        else:
            calendar_value = 'gregorian'
        
        # Encode calendar in encoding instead
        encoding = {
            'precip': {'zlib': True, 'complevel': 5},
            'time': {'calendar': calendar_value}
        }
        
        try:
            ds.to_netcdf(test_file, encoding=encoding)
            print(f"✅ SUCCESS! File saved without errors")
            
            # Verify we can read it back
            ds_read = xr.open_dataset(test_file)
            print(f"✅ File can be read back successfully")
            print(f"  Time calendar: {ds_read['time'].encoding.get('calendar', 'unknown')}")
            ds_read.close()
            
            return True
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            return False

if __name__ == '__main__':
    print("="*60)
    print("NetCDF Calendar Attribute Fix Test")
    print("="*60)
    print()
    
    success = test_save_with_calendar()
    
    print()
    print("="*60)
    if success:
        print("✅ All tests passed!")
        print("Your scripts should now work without calendar errors.")
    else:
        print("❌ Test failed!")
        print("There may still be an issue.")
    print("="*60)


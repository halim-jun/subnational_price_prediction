"""
Quick script to inspect FLDAS NetCDF structure
"""
import xarray as xr
import os

# Open one sample file
fldas_dir = "./data/raw/climate/fldas/"
sample_file = os.path.join(fldas_dir, "FLDAS_NOAH01_C_GL_M.A201901.001.nc")

print(f"📂 Opening: {sample_file}")
print("=" * 80)

ds = xr.open_dataset(sample_file, engine='netcdf4')

print("\n📊 Dataset Info:")
print(ds)

print("\n📈 Variables:")
for var in ds.data_vars:
    print(f"  - {var}: {ds[var].dims} | {ds[var].shape}")

print("\n🌍 Coordinates:")
for coord in ds.coords:
    print(f"  - {coord}: {ds[coord].shape}")
    
print("\n🔍 Sample values:")
print(f"  Lat range: {ds['Y'].min().values} to {ds['Y'].max().values}")
print(f"  Lon range: {ds['X'].min().values} to {ds['X'].max().values}")

ds.close()


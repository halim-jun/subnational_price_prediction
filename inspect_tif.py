import rasterio
import os

tif_path = "data/night_lights/Kenya/VNP46A4.A2012001.h21v09.002.2025086173009.tif"

if os.path.exists(tif_path):
    try:
        with rasterio.open(tif_path) as src:
            print(f"File: {tif_path}")
            print(f"CRS: {src.crs}")
            print(f"Bounds: {src.bounds}")
            print(f"Transform: {src.transform}")
            print(f"Shape: {src.shape}")
            print(f"NoData: {src.nodata}")
            data = src.read(1)
            print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
    except Exception as e:
        print(f"Error opening TIF: {e}")
else:
    print(f"File not found: {tif_path}")

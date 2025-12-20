
import pandas as pd
import geopandas as gpd

print("Scanning for unique locations...")
unique_locs = set()
chunksize = 1000000
for chunk in pd.read_csv('data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month.csv', usecols=['lat', 'lon'], chunksize=chunksize):
    unique_locs.update(zip(chunk['lat'], chunk['lon']))
    print(f"Found {len(unique_locs)} unique locations so far...")

print(f"Total unique locations: {len(unique_locs)}")

# Convert to DataFrame
df_unique = pd.DataFrame(list(unique_locs), columns=['lat', 'lon'])

# Convert to GDF
gdf = gpd.GeoDataFrame(
    df_unique, geometry=gpd.points_from_xy(df_unique.lon, df_unique.lat), crs="EPSG:4326"
)

# Identify countries (Admin 0)
print("Identifying countries...")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
joined = gpd.sjoin(gdf, world, how="left", predicate="within")

countries = joined['iso_a3'].unique()
countries = [c for c in countries if pd.notna(c) and c != '-99']
print("Countries identified:", countries)

# Save found countries to file
with open('iso_codes.txt', 'w') as f:
    f.write(','.join(countries))

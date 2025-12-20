
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load SPI data (just lat/lon unique)
print("Loading SPI data...")
# Read only lat/lon to save memory/time
df = pd.read_csv('data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month.csv', usecols=['lat', 'lon'])
unique_points = df.drop_duplicates()
print(f"Unique points: {len(unique_points)}")

# Conver to GeoDataFrame
gdf_points = gpd.GeoDataFrame(
    unique_points, geometry=gpd.points_from_xy(unique_points.lon, unique_points.lat), crs="EPSG:4326"
)

# Load World Data
print("Loading World data...")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Sjoin
print("Joining...")
joined = gpd.sjoin(gdf_points, world, how="left", predicate="within")

# Get list of countries
countries = joined['iso_a3'].unique()
print("Countries found:", countries)

# Save to a file so I can read it
with open('found_countries.txt', 'w') as f:
    for c in countries:
        if pd.notna(c) and c != '-99':
            f.write(c + '\n')

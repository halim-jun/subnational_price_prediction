
import requests
import geopandas as gpd
import io

# Get metadata
url = "https://www.geoboundaries.org/api/current/gbOpen/ETH/ADM3/"
print(f"Fetching {url}")
r = requests.get(url)
data = r.json()
print("Metadata keys:", data.keys())

dl_url = data['gjDownloadURL']
print(f"Downloading GeoJSON from {dl_url}...")

r2 = requests.get(dl_url)
gdf = gpd.read_file(io.BytesIO(r2.content))
print("Columns:", gdf.columns)
print("First row:", gdf.iloc[0])

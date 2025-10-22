import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Download directory
output_dir = Path("data/raw/climate/chirps")
output_dir.mkdir(parents=True, exist_ok=True)

# Get file list
url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_2-monthly/tifs/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all .tiff files
tiff_files = [a['href'] for a in soup.find_all('a') if a.get('href', '').endswith('.tiff')]

print(f"Found {len(tiff_files)} files to download")

# Download each file
for i, filename in enumerate(tiff_files, 1):
    file_url = url + filename
    output_path = output_dir / filename
    
    print(f"[{i}/{len(tiff_files)}] Downloading {filename}...")
    
    file_response = requests.get(file_url)
    output_path.write_bytes(file_response.content)
    
    print(f"  Saved to {output_path}")

print("\nAll files downloaded!")


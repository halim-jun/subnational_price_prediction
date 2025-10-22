import requests
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NASA Earthdata credentials
username = os.getenv("NASA_EARTHDATA_USERNAME")
password = os.getenv("NASA_EARTHDATA_PASSWORD")

# Read URLs from file
with open('/Users/halimjun/Downloads/subset_FLDAS_NOAH01_C_GL_M_001_20251022_070311_.txt', 'r') as f:
    urls = [line.strip() for line in f if line.strip() and line.startswith('http')]

# Download directory
output_dir = Path("data/raw/climate/fldas")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(urls)} URLs")

# Create session with authentication
session = requests.Session()
session.auth = (username, password)

# Download each file
for i, url in enumerate(urls, 1):
    filename = url.split('/')[-1]
    output_path = output_dir / filename
    
    print(f"[{i}/{len(urls)}] Downloading {filename}...")
    
    response = session.get(url, allow_redirects=True)
    output_path.write_bytes(response.content)
    
    print(f"  Saved to {output_path}")

print("\nAll files downloaded!")


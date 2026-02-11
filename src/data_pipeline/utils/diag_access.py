
import os
import glob

def check_structure():
    print(f"Propagated CWD: {os.getcwd()}")
    
    # Check GeoBoundaries Glob
    boundaries_pattern = "data/geoboundaries/*_ADM2.geojson"
    print(f"Globbing: {boundaries_pattern}")
    files = glob.glob(boundaries_pattern)
    print(f"Found {len(files)} files via glob.")
    if files:
        print(f"Sample: {files[0]}")
    else:
        # List dir manually to debug
        if os.path.exists("data/geoboundaries"):
            print("Contents of data/geoboundaries:", os.listdir("data/geoboundaries")[:5])
    
    # Check Input Parquet
    input_file = "data/crop_mask/asap_mask_crop_v04.parquet"
    print(f"\nChecking input file: {input_file}")
    if os.path.exists(input_file):
        print("Input file exists.")
        try:
             # Basic read check
             size = os.path.getsize(input_file)
             print(f"File size: {size} bytes")
        except Exception as e:
            print(f"Error reading info: {e}")
    else:
        print("Input file NOT found.")

if __name__ == "__main__":
    check_structure()

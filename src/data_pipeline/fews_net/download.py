import os
import urllib.request
import ssl
import time

def download_and_merge():
    # Define countries to download
    countries = ["SO", "KE", "ET"]
    base_url = "https://fdw.fews.net/api/ipcphase/?format=csv&country_code={}&preference=best&fields=simple"
    
    
    # Output setup
    output_dir = "data/fewsnet/food_security"
    final_output_file = os.path.join(output_dir, "fewsnet_ipc_data.csv")
    
    print(f"Ensuring directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # SSL Context
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    temp_files = []
    
    # 1. Download each country
    for country in countries:
        url = base_url.format(country)
        temp_file = os.path.join(output_dir, f"temp_{country}.csv")
        print(f"Downloading {country} data from {url}...")
        
        try:
            with urllib.request.urlopen(url, context=ctx) as response:
                data = response.read()
                with open(temp_file, "wb") as f:
                    f.write(data)
            print(f"Saved {country} data to {temp_file}")
            temp_files.append(temp_file)
        except Exception as e:
            print(f"Failed to download {country}: {e}")
            # Clean up any partial downloads if needed? 
            # For now, we continue and try to merge what we have, or exit?
            # Let's simple exit to be safe.
            exit(1)
        
        # Be nice to the API
        time.sleep(1)

    # 2. Merge files
    print(f"Merging files into {final_output_file}...")
    try:
        with open(final_output_file, 'w') as outfile:
            for i, fname in enumerate(temp_files):
                with open(fname, 'r') as infile:
                    header = infile.readline()
                    # Write header only from the first file
                    if i == 0:
                        outfile.write(header)
                    
                    # Write the rest of the lines
                    for line in infile:
                        outfile.write(line)
                print(f"Merged {fname}")
        
        print(f"Successfully created {final_output_file}")
        
    except Exception as e:
        print(f"Error merging files: {e}")
        exit(1)

    # 3. Cleanup
    print("Cleaning up temporary files...")
    for fname in temp_files:
        try:
            os.remove(fname)
        except OSError as e:
            print(f"Error removing {fname}: {e}")

    print("Done.")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
    #download crop production facts
def crop_download():
    crop_url = "https://fdw.fews.net/api/cropproductionfacts/?format=csv&end_date=2026-01-01&start_date=2015-01-01&country_code=KE&country_code=ET&country_code=SO&fields=simple&start_date=2015-01-01&end_date=2026-01-01"
    crop_production_output_dir = "data/fewsnet/crop_production/crop_production.csv"
    try:
        with urllib.request.urlopen(crop_url, context=ctx) as response:
            data = response.read()
            with open(crop_production_output_dir, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"Failed to download crop production data: {e}")
        exit(1)
#download trade flow quantity value
def trade_flow_download():
    tradeflow_url = "https://fdw.fews.net/api/tradeflowquantityvalue/?format=csv&fields=fde"

    trade_flow_output_dir = "data/fewsnet/trade_flow/tradeflow.csv"
    try:
        with urllib.request.urlopen(tradeflow_url, context=ctx) as response:
            data = response.read()
            with open(trade_flow_output_dir, "wb") as f:
                f.write(data)
    except Exception as e:
        print(f"Failed to download trade flow data: {e}")
        exit(1)


if __name__ == "__main__":
    download_and_merge()
    crop_download()
    trade_flow_download()

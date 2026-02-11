import os

def merge_csvs():
    files = ["fewsnet_ipc_so.csv", "fewsnet_ipc_ke.csv", "fewsnet_ipc_et.csv"]
    output_file = "fewsnet_ipc_data.csv"
    
    print(f"Merging {files} into {output_file}")
    
    with open(output_file, 'w') as outfile:
        for i, fname in enumerate(files):
            if not os.path.exists(fname):
                print(f"Warning: {fname} does not exist, skipping.")
                continue
                
            with open(fname, 'r') as infile:
                header = infile.readline()
                if i == 0:
                    outfile.write(header)
                
                # Write the rest
                for line in infile:
                    outfile.write(line)
            
            print(f"Appended {fname}")
            
    print(f"Merge complete. Saved to {output_file}")
    
    # Clean up
    for fname in files:
        if os.path.exists(fname):
            os.remove(fname)
            print(f"Removed {fname}")

if __name__ == "__main__":
    merge_csvs()

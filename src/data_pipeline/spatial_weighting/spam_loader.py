import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

def load_spam_data(
    spam_dir: Path, 
    crop_code: str, 
    variable: str = 'H',
    tech_type: str = 'TA',
    region_filter: dict = None
) -> xr.DataArray:
    """
    Loads SPAM 2020 Grid Data (CSV format) for a specific crop and technology.
    Based on Readme_SPAM2020V2r0.txt conventions.
    
    Args:
        spam_dir (Path): Directory containing 'spam2020V2r0_global_{variable}_{tech_type}.csv'
        crop_code (str): 4-letter SPAM crop code (e.g., 'MAIZ', 'WHEA'). Case-insensitive.
        variable (str): SPAM variable type: 'H' (Harvested), 'A' (Physical), 'P' (Production), 'Y' (Yield). Default 'H'.
        tech_type (str): Technology type: 'TA' (Total), 'TI' (Irrigated), 'TR' (Rainfed). Default 'TA'.
        region_filter (dict): Dictionary for filtering, e.g., {'ADM0_NAME': 'Ethiopia'} or {'FIPS0': 'ET'}.
        
    Returns:
        xr.DataArray: Gridded data with coords (lat, lon).
    """
    
    # 1. Resolve File Name
    # Convention: spam2020V2r0_global_v_t.csv
    # v = variable code (H, A, P, Y) -- Readme says naming uses full word? 
    # Readme: "spam2020V2r0_global_harvested_area.zip" -> zipname.
    # Readme: "File names ... spam2020V2r0_global_v_t.csv" where v=variable.
    # v options: *_A_* physical area ?? Usage text is confusing.
    # Let's check the actual file system name we have: "spam2020V2r0_global_H_TA.csv"
    # So v='H' in the filename.
    
    filename = f"spam2020V2r0_global_{variable}_{tech_type}.csv"
    file_path = spam_dir / filename
    
    if not file_path.exists():
        # Fallback: maybe the folder name is the variable name but file is H_TA?
        # User path: data/raw/spam2020/Global_CSV/spam2020V2r0_global_harvested_area/spam2020V2r0_global_H_TA.csv
        # If spam_dir is the dataset root, we might need to search?
        # Assuming spam_dir is the folder CONTAINING the csv.
        raise FileNotFoundError(f"SPAM file not found: {file_path}")
        
    # 2. Resolve Column Name
    # Readme: "each pixel has ... 46 fields for 46 crops: similar to SPAM notation: crop_T, where T = A, I, or R"
    # T depends on tech_type?
    # TA (Total) -> suffix _A
    # TI (Irrigated) -> suffix _I
    # TR (Rainfed) -> suffix _R
    
    suffix_map = {
        'TA': 'A',
        'TI': 'I',
        'TR': 'R'
    }
    
    if tech_type not in suffix_map:
        raise ValueError(f"Unknown tech_type: {tech_type}. Must be TA, TI, or TR.")
        
    final_suffix = suffix_map[tech_type]
    target_col = f"{crop_code.upper()}_{final_suffix}"
    
    # 3. Load Data
    # Read header first to validation
    header_df = pd.read_csv(file_path, nrows=0)
    if target_col not in header_df.columns:
        raise ValueError(f"Column {target_col} not found in {filename}. check crop code.")
        
    # Columns to load
    use_cols = ['x', 'y', target_col]
    
    # Add filtering columns
    filter_col = None
    filter_val = None
    if region_filter:
        filter_col = list(region_filter.keys())[0]
        filter_val = list(region_filter.values())[0]
        if filter_col not in header_df.columns:
            raise ValueError(f"Filter column {filter_col} not found in file. Available: {list(header_df.columns)}")
        use_cols.append(filter_col)
        
    # Read CSV
    df = pd.read_csv(file_path, usecols=use_cols)
    
    # 4. Filter
    if filter_col:
        df = df[df[filter_col] == filter_val]
        if df.empty:
            print(f"Warning: No data found for {filter_col}={filter_val}")
            
    # 5. Convert to Xarray
    # Index: y (lat), x (lon)
    ds = df.set_index(['y', 'x']).to_xarray()
    da = ds[target_col]
    
    # Rename coords to standard (lat, lon)
    da = da.rename({'x': 'lon', 'y': 'lat'})
     
    # Attributes
    units = 'ha' if variable in ['H', 'A'] else 'mt' if variable == 'P' else 'kg/ha'
    da.attrs['units'] = units
    da.attrs['long_name'] = f'{crop_code} {variable} ({tech_type})'
    
    return da

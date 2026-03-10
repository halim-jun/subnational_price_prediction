
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
from pathlib import Path
from shapely.geometry import Point
import difflib

# Configure logging
_log_dir = Path(__file__).parent
_log_file = _log_dir / 'merge_pipeline.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# Manual overrides for Admin1-level city names -> representative Admin2 name
# Used as fallback when spatial join is not possible (e.g., Population data lacks coordinates)
MANUAL_OVERRIDES = {
    # Kenya
    'Nairobi': 'Starehe',
    'Mombasa': 'Mvita',
    'Garissa': 'Garissa Township',
    'Tana River': 'Galole',
    'Kwale': 'Msambweni',
    # Somalia
    'Banadir': 'HAMAR WEYNE',
    'Gaalkacyo': 'GALKAYO',
    # Ethiopia
    'Addis Abeba': 'Central',
    'Addis Ababa': 'Central',
}

try:
    from thefuzz import process, fuzz
    HAS_THEFUZZ = True
except ImportError:
    HAS_THEFUZZ = False
    logger.warning("thefuzz not found. Using difflib for string matching (less accurate).")

try:
    from rasterstats import zonal_stats
    HAS_RASTERSTATS = True
except ImportError:
    HAS_RASTERSTATS = False
    logger.warning("rasterstats not found. Zonal statistics will fail.")

def load_canonical_boundaries(iso3_list, data_dir):
    """
    Loads Admin2 boundaries for specified countries from GeoJSON files.
    
    Args:
        iso3_list (list): List of ISO3 codes (e.g., ['KEN', 'SOM', 'ETH']).
        data_dir (str): Absolute Path to directory containing geoboundaries.
        
    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame with columns ['shapeName', 'shapeISO', 'geometry'].
    """
    gdfs = []
    data_path = Path(data_dir)
    
    for iso in iso3_list:
        # File pattern: gb_{ISO}_ADM2.geojson
        file_name = f"gb_{iso}_ADM2.geojson"
        file_path = data_path / file_name
        
        if not file_path.exists():
            # Try searching for the file
            found = list(data_path.glob(f"*{iso}*ADM2*.geojson"))
            if found:
                file_path = found[0]
            else:
                logger.error(f"Could not find Admin2 boundary for {iso} at {file_path}")
                continue
                
        try:
            gdf = gpd.read_file(file_path)
            # Ensure necessary columns
            if 'shapeName' not in gdf.columns:
                 # Fallback for different column names
                if 'ADM2_NAME' in gdf.columns:
                    gdf['shapeName'] = gdf['ADM2_NAME']
                elif 'Name' in gdf.columns:
                    gdf['shapeName'] = gdf['Name']
            
            # Ensure shapeISO is populated
            if 'shapeISO' not in gdf.columns:
                gdf['shapeISO'] = iso
            else:
                # If column exists but has empty values, fill them
                gdf['shapeISO'] = gdf['shapeISO'].replace('', np.nan).fillna(iso)
                
            # Keep only relevant columns
            gdf = gdf[['shapeName', 'shapeISO', 'geometry']]
            gdfs.append(gdf)
            logger.info(f"Loaded {iso} boundaries: {len(gdf)} regions")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            
    if not gdfs:
        raise ValueError(f"No boundary files loaded from {data_dir}")
        
    full_gdf = pd.concat(gdfs, ignore_index=True)
    return full_gdf

# ... (fuzzy_match_names and spatial_join_acled remain unchanged) ...

def fuzzy_match_names(series, choices, threshold=80):
    """
    Maps a series of names to a list of canonical choices using fuzzy matching.
    
    Args:
        series (pd.Series): Series of names to match.
        choices (list): List of canonical names.
        threshold (int): Score threshold (0-100) for acceptance.
        
    Returns:
        dict: Mapping {original_name: canonical_name}
    """
    mapping = {}
    unique_names = series.dropna().unique()
    choices_map = {c.lower(): c for c in choices} # Lowercase map for exact match check
    
    for name in unique_names:
        name_clean = str(name).strip()
        name_lower = name_clean.lower()
        
        # 1. Exact Match (Case-insensitive)
        if name_lower in choices_map:
            mapping[name] = choices_map[name_lower]
            continue
            
        # 2. Fuzzy Match
        if HAS_THEFUZZ:
            # extractOne returns (match, score)
            match, score = process.extractOne(name_clean, choices)
            if score >= threshold:
                mapping[name] = match
            else:
                logger.warning(f"No match found for '{name}' (Best: {match}, Score: {score})")
                mapping[name] = None # Or keep original? Better to be explicit about failure
        else:
            # Difflib fallback
            matches = difflib.get_close_matches(name_clean, choices, n=1, cutoff=threshold/100)
            if matches:
                mapping[name] = matches[0]
            else:
                logger.warning(f"No match found for '{name}' (difflib)")
                mapping[name] = None
                
    return mapping

def spatial_join_points(df, admin_gdf, lon_col, lat_col):
    """
    Spatially joins point data (any df with lat/lon) to Admin2 polygons.
    
    Args:
        df (pd.DataFrame): Data with latitude and longitude columns.
        admin_gdf (gpd.GeoDataFrame): Admin2 boundaries.
        lon_col (str): Column name for longitude.
        lat_col (str): Column name for latitude.
        
    Returns:
        pd.DataFrame: Data with 'admin2_canonical' column added.
    """
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col])
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    # Ensure CRS match
    admin_aligned = admin_gdf.to_crs("EPSG:4326") if admin_gdf.crs != "EPSG:4326" else admin_gdf
    
    joined = gpd.sjoin(points_gdf, admin_aligned, how="left", predicate="within")
    joined.rename(columns={'shapeName': 'admin2_canonical', 'shapeISO': 'admin0_canonical'}, inplace=True)
    
    # Log unmatched
    unmatched = joined['admin2_canonical'].isna().sum()
    if unmatched > 0:
        logger.warning(f"Spatial join: {unmatched}/{len(df)} points unmatched (outside polygons).")
    
    return pd.DataFrame(joined.drop(columns='geometry'))


def spatial_join_acled(acled_df, admin_gdf):
    """
    Spatially joins ACLED events to Admin2 regions.
    Wrapper around spatial_join_points for ACLED-specific column names.
    """
    return spatial_join_points(acled_df, admin_gdf, 'CENTROID_LONGITUDE', 'CENTROID_LATITUDE')

def process_worldpop_population(admin_gdf, raster_path, country_iso):
    """
    Computes population for each Admin2 region using zonal statistics on WorldPop raster.
    
    Args:
        admin_gdf (gpd.GeoDataFrame): Canonical boundaries (must contain shapeISO).
        raster_path (str): Path to the .tif file.
        country_iso (str): ISO3 code to filter admin_gdf (e.g. 'KEN').
        
    Returns:
        pd.DataFrame: DataFrame with ['admin2_canonical', 'population', 'country_iso'].
    """
    if not HAS_RASTERSTATS:
        logger.error("rasterstats not installed. Cannot process WorldPop data.")
        return pd.DataFrame()
        
    # Filter for specific country
    country_gdf = admin_gdf[admin_gdf['shapeISO'] == country_iso].copy()
    if country_gdf.empty:
        logger.warning(f"No boundaries found for {country_iso} in admin_gdf.")
        return pd.DataFrame()
        
    logger.info(f"Processing WorldPop for {country_iso} with {len(country_gdf)} regions...")
    
    # Run zonal stats (sum)
    # rasterstats returns specific order matching the input vector
    # all_touched=True ensures small polygons that don't cover a pixel center still get value
    stats = zonal_stats(country_gdf, raster_path, stats="sum", all_touched=True)
    
    # Add results to dataframe
    country_gdf['population'] = [s['sum'] for s in stats]
    
    # Select relevant columns
    result = country_gdf[['shapeName', 'population', 'shapeISO']].rename(
        columns={'shapeName': 'admin2_canonical', 'shapeISO': 'country_iso'}
    )
    
    # Drop existing index/geometry to return pure DataFrame
    return pd.DataFrame(result.drop(columns='geometry', errors='ignore'))

def create_master_skeleton(years, months, admin_gdf):
    """
    Creates a master DataFrame with all combinations of Year, Month, Admin2.
    """
    records = []
    
    # Get unique regions from canonical source
    regions = admin_gdf[['shapeName', 'shapeISO']].drop_duplicates()
    
    for year in years:
        for month in months:
            # Repeat regions for this timestep
            # Use pandas cross join or simple loop
            # Loop is safer for small data
            
            # Create a localized copy
            step_df = regions.copy()
            step_df['year'] = year
            step_df['month'] = month
            records.append(step_df)
            
    master_df = pd.concat(records, ignore_index=True)
    # Rename columns to match target schema
    master_df.rename(columns={'shapeName': 'admin2', 'shapeISO': 'country_iso'}, inplace=True)
    
    return master_df

def merge_datasets(price_df, crop_df, acled_df, data_dir, iso3_list=['KEN', 'SOM', 'ETH'], pop_df=None):
    """
    Main function to merge all datasets.
    Uses spatial join (lat/lon) for Price and ACLED data.
    Uses fuzzy matching + manual overrides for Population data.
    Uses direct name matching for Crop data (same GeoBoundaries source).
    """
    # 1. Load Canonical Boundaries
    boundary_dir = Path(data_dir) / 'geoboundaries'
    logger.info(f"Loading canonical boundaries from {boundary_dir}...")
    admin_gdf = load_canonical_boundaries(iso3_list, str(boundary_dir))
    canonical_names = admin_gdf['shapeName'].unique().tolist()
    
    # ── 2. Standardize Price Data (SPATIAL JOIN per country) ──
    # Join each country's price data against its own boundaries to prevent
    # cross-boundary misassignment (e.g., SOM markets falling in ETH polygons).
    logger.info("Standardizing Price Data via Spatial Join (per-country)...")
    if 'lat' in price_df.columns and 'lon' in price_df.columns:
        price_joined_parts = []
        for iso in iso3_list:
            iso_prices = price_df[price_df['ISO3'] == iso].copy()
            if iso_prices.empty:
                logger.warning(f"No price data for {iso} in source CSV.")
                continue
            iso_bounds = admin_gdf[admin_gdf['shapeISO'] == iso]
            iso_with_coords = iso_prices.dropna(subset=['lat', 'lon'])
            if iso_with_coords.empty:
                logger.warning(f"No price data with coordinates for {iso}.")
                continue
            joined = spatial_join_points(iso_with_coords, iso_bounds, 'lon', 'lat')
            matched = joined['admin2_canonical'].notna().sum()
            logger.info(f"Price spatial join {iso}: {matched}/{len(joined)} matched ({matched/len(joined)*100:.1f}%)")
            price_joined_parts.append(joined)
        if price_joined_parts:
            price_joined = pd.concat(price_joined_parts, ignore_index=True)
        else:
            logger.error("No price data matched any boundaries.")
            price_joined = price_df.copy()
            price_joined['admin2_canonical'] = np.nan
    else:
        # Fallback to fuzzy matching if no coordinates
        logger.warning("Price data has no lat/lon columns. Falling back to fuzzy matching.")
        price_mapping = fuzzy_match_names(price_df['adm2_name'], canonical_names)
        price_joined = price_df.copy()
        price_joined['admin2_canonical'] = price_joined['adm2_name'].map(price_mapping)
    
    # Aggregate Price: mean across markets in same admin2/year/month
    # Use c_maize_fao (available for both KEN and SOM) + c_food_price_index
    price_cols = ['c_maize_fao', 'c_food_price_index', 'c_sorghum']
    # Keep only columns that exist and are numeric
    price_cols = [c for c in price_cols if c in price_joined.columns]
    price_cols = price_joined[price_cols].select_dtypes(include=np.number).columns.tolist()
    logger.info(f"Price columns kept: {price_cols}")
    
    price_agg = price_joined.dropna(subset=['admin2_canonical']).groupby(
        ['year', 'month', 'admin2_canonical']
    )[price_cols].mean().reset_index()
    
    # ── 3. Standardize Population Data (WORLDPOP ZONAL STATS) ──
    logger.info("Standardizing Population Data via WorldPop Zonal Stats...")
    pop_dfs = []
    
    # Map ISO to raster file
    raster_map = {
        'KEN': 'ken_pop_2020_1km.tif',
        'SOM': 'som_pop_2020_1km.tif',
        'ETH': 'eth_pop_2020_1km.tif'
    }
    
    for iso in iso3_list:
        if iso not in raster_map:
            continue
        # Check both naming conventions (downloaded vs potential variants)
        # We downloaded as {iso}_ppp_2020.tif
        raster_path = Path(data_dir) / 'population_worldpop' / raster_map[iso]
        
        if raster_path.exists():
            iso_pop = process_worldpop_population(admin_gdf, str(raster_path), iso)
            if not iso_pop.empty:
                pop_dfs.append(iso_pop)
        else:
            logger.warning(f"Population raster not found for {iso}: {raster_path}")

    if pop_dfs:
        pop_agg = pd.concat(pop_dfs, ignore_index=True)
        # Sum if duplicates exist (shouldn't happen with canonical admin2)
        pop_agg = pop_agg.groupby(['admin2_canonical'])['population'].sum().reset_index()
    else:
        logger.warning("No population data processed.")
        pop_agg = pd.DataFrame(columns=['admin2_canonical', 'population'])
    
    # ── 4. Standardize Crop Data (DIRECT NAME MATCH) ──
    logger.info("Standardizing Crop Data...")
    crop_df = crop_df.copy()
    # Filter to target countries only
    if 'shapeISO_ADM0' in crop_df.columns:
        crop_df = crop_df[crop_df['shapeISO_ADM0'].isin(iso3_list)]
        logger.info(f"Crop data filtered to {iso3_list}: {len(crop_df)} rows")
    
    canonical_set = set(canonical_names)
    crop_df['admin2_canonical'] = crop_df['shapeName_ADM2'].where(
        crop_df['shapeName_ADM2'].isin(canonical_set)
    )
    unmatched_crop = crop_df['admin2_canonical'].isna().sum()
    if unmatched_crop > 0:
        logger.warning(f"Crop data: {unmatched_crop} rows with unmatched Admin2 names. Falling back to fuzzy match.")
        remaining_crop = crop_df.loc[crop_df['admin2_canonical'].isna(), 'shapeName_ADM2']
        crop_fallback = fuzzy_match_names(remaining_crop, canonical_names)
        crop_df.loc[crop_df['admin2_canonical'].isna(), 'admin2_canonical'] = remaining_crop.map(crop_fallback)
    
    crop_agg = crop_df.dropna(subset=['admin2_canonical']).groupby(
        ['admin2_canonical']
    )['value'].mean().reset_index().rename(columns={'value': 'crop_cover_fraction'})
    
    # ── 5. Connect ACLED (SPATIAL JOIN) ──
    logger.info("Joining ACLED Data via Spatial Join...")
    acled_joined = spatial_join_acled(acled_df, admin_gdf)
    
    acled_joined['year'] = pd.to_datetime(acled_joined['WEEK']).dt.year
    acled_joined['month'] = pd.to_datetime(acled_joined['WEEK']).dt.month
    
    acled_agg = acled_joined.dropna(subset=['admin2_canonical']).groupby(
        ['year', 'month', 'admin2_canonical']
    ).agg({
        'FATALITIES': 'sum',
        'EVENTS': 'count'
    }).reset_index().rename(columns={'EVENTS': 'conflict_events', 'FATALITIES': 'conflict_fatalities'})

    # ── 6. Create Master Skeleton ──
    years = sorted(price_df['year'].unique())
    months = sorted(price_df['month'].unique())
    
    logger.info(f"Creating skeleton for {len(years)} years, {len(months)} months, {len(canonical_names)} regions...")
    master = create_master_skeleton(years, months, admin_gdf)
    
    # ── 7. Merge All ──
    logger.info("Merging all datasets...")
    
    merged = pd.merge(master, price_agg,
                      left_on=['year', 'month', 'admin2'],
                      right_on=['year', 'month', 'admin2_canonical'], how='left')
    if 'admin2_canonical' in merged.columns:
        merged.drop(columns=['admin2_canonical'], inplace=True)
    
    merged = pd.merge(merged, pop_agg,
                      left_on='admin2', right_on='admin2_canonical', how='left')
    if 'admin2_canonical' in merged.columns:
        merged.drop(columns=['admin2_canonical'], inplace=True)
    
    merged = pd.merge(merged, crop_agg,
                      left_on='admin2', right_on='admin2_canonical', how='left')
    if 'admin2_canonical' in merged.columns:
        merged.drop(columns=['admin2_canonical'], inplace=True)
    
    merged = pd.merge(merged, acled_agg,
                      left_on=['year', 'month', 'admin2'],
                      right_on=['year', 'month', 'admin2_canonical'], how='left')
    if 'admin2_canonical' in merged.columns:
        merged.drop(columns=['admin2_canonical'], inplace=True)
    
    merged['conflict_events'] = merged['conflict_events'].fillna(0)
    merged['conflict_fatalities'] = merged['conflict_fatalities'].fillna(0)

    # Crop cover: fill NULL with 0 (urban/desert admin2 with no cropland)
    crop_null_count = merged['crop_cover_fraction'].isna().sum()
    if crop_null_count > 0:
        merged['crop_cover_fraction'] = merged['crop_cover_fraction'].fillna(0)
        logger.info(f"Filled {crop_null_count} crop_cover_fraction NULLs with 0 (urban/desert areas)")

    logger.info(f"Final merged shape: {merged.shape}")
    return merged


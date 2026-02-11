import xarray as xr
import numpy as np

def aggregate_climate_by_weights(
    ds_climate: xr.DataArray, 
    da_weights: xr.DataArray, 
    region_mask: xr.DataArray = None
) -> xr.DataArray:
    """
    Aggregates gridded climate data using spatial weights (e.g. Harvested Area).
    
    Args:
        ds_climate (xr.DataArray): Climate data (time, lat, lon) or (time, y, x).
                                   Must have CRS or readable lat/lon.
        da_weights (xr.DataArray): Weight map (lat, lon).
        region_mask (xr.DataArray, optional): Binary mask for region of interest. 
                                              If None, assumes weights are already masked.
        
    Returns:
        xr.DataArray: Time series of weighted indices.
    """
    
    # 1. Align Grids
    # We use the climate grid as the target resolution usually, 
    # as it's often coarser (Chirps 0.05) vs SPAM (0.0833 or 5min ~ 0.0833).
    # Actually SPAM (5 min) is approx 10km. CHIRPS (0.05 deg) is approx 5.5km.
    # So CHIRPS is finer! We should regrid Weights (SPAM) to Climate (CHIRPS).
    
    # Ensure dimensions are named lat/lon
    ds_climate = _standardize_dims(ds_climate)
    da_weights = _standardize_dims(da_weights)
    
    # Reproject/Regrid Weights to match Climate Grid
    # Interpolation: 'nearest' conserves totals better for categorical, 
    # but for continuous area, 'linear' or 'conservative' is best.
    # xarray interp is linear by default.
    weights_aligned = da_weights.interp_like(ds_climate, method='linear')
    
    # Handle NaN in weights (ocean/desert)
    weights_aligned = weights_aligned.fillna(0.0)
    
    # Apply Mask if provided
    if region_mask is not None:
        region_mask_aligned = region_mask.interp_like(ds_climate, method='nearest')
        weights_aligned = weights_aligned.where(region_mask_aligned)
        
    # 2. Weighted Mean Calculation
    # Formula: Sum(Climate * Weight) / Sum(Weight)
    
    # Broadcast weights to time dimension
    # (xarray handles this automatically)
    
    weighted_sum = (ds_climate * weights_aligned).sum(dim=['lat', 'lon'])
    total_weight = weights_aligned.sum(dim=['lat', 'lon'])
    
    weighted_index = weighted_sum / total_weight
    
    return weighted_index

def _standardize_dims(da):
    """Rename coords to lat/lon if they are x/y or latitude/longitude."""
    rename = {}
    if 'latitude' in da.coords: rename['latitude'] = 'lat'
    if 'longitude' in da.coords: rename['longitude'] = 'lon'
    if 'y' in da.coords: rename['y'] = 'lat'
    if 'x' in da.coords: rename['x'] = 'lon'
    
    if rename:
        da = da.rename(rename)
    return da

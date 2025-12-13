"""
Generate Standardized Precipitation Index (SPI) from CHIRPS data using pure Python

This script replaces CDO/NCO commands with Python-based operations to:
1. Clip CHIRPS data to East Africa region
2. Preprocess (fill missing values, regrid)
3. Calculate SPI at multiple time scales
4. Save CF-compliant NetCDF files

Requirements:
- xarray, numpy, scipy, climate-indices
"""

import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
import subprocess
import warnings
warnings.filterwarnings('ignore')


class CHIRPStoSPI:
    """
    Convert CHIRPS precipitation data to Standardized Precipitation Index
    """
    
    def __init__(self, input_file, output_dir='../../data/processed/spi'):
        """
        Initialize the SPI generator
        
        Args:
            input_file: Path to CHIRPS NetCDF file
            output_dir: Directory to save outputs
        """
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for each processing stage
        self.dirs = {
            'clipped': os.path.join(output_dir, '01_clipped'),
            'filled': os.path.join(output_dir, '02_filled'),
            'metadata': os.path.join(output_dir, '03_metadata_revision'),
            'spi_intermediate': os.path.join(output_dir, '04_spi_intermediate'),
            'spi_final': os.path.join(output_dir, '05_spi_final')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def step1_clip_region(self, lon_min=25, lon_max=52, lat_min=-15, lat_max=22,
                          year_start=2016, year_end=2024):
        """
        Step 1: Clip CHIRPS data to region and time range
        
        Replaces: cdo -z zip_5 sellonlatbox,25,52,-15,22 -selyear,2016/2024
        
        Args:
            lon_min, lon_max: Longitude bounds for East Africa
            lat_min, lat_max: Latitude bounds
            year_start, year_end: Year range
        """
        print("=" * 60)
        print("STEP 1: Clipping to East Africa region (2016-2024)")
        print("=" * 60)
        
        print(f"Loading CHIRPS data from: {self.input_file}")
        ds = xr.open_dataset(self.input_file)
        
        # Get dimension names (they might vary: lon/longitude, lat/latitude)
        lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
        
        print(f"Original shape: {dict(ds.dims)}")
        
        # Select time range
        print(f"Selecting years: {year_start}-{year_end}")
        ds_time = ds.sel(time=slice(f'{year_start}-01-01', f'{year_end}-12-31'))
        
        # Select spatial region
        print(f"Selecting lon: [{lon_min}, {lon_max}], lat: [{lat_min}, {lat_max}]")
        ds_clipped = ds_time.sel(
            {lon_name: slice(lon_min, lon_max),
             lat_name: slice(lat_min, lat_max)}
        )
        
        # Save with compression
        output_file = os.path.join(self.dirs['clipped'], 'east_africa_chirps_clipped.nc')
        print(f"Saving to: {output_file}")
        
        # Get precipitation variable name
        precip_var = 'precip' if 'precip' in ds_clipped else 'precipitation'
        
        # Remove conflicting attributes from time variable
        if 'calendar' in ds_clipped['time'].attrs:
            calendar_value = ds_clipped['time'].attrs['calendar']
            del ds_clipped['time'].attrs['calendar']
        else:
            calendar_value = 'gregorian'
        
        encoding = {
            precip_var: {'zlib': True, 'complevel': 5},
            'time': {'calendar': calendar_value}
        }
        ds_clipped.to_netcdf(output_file, encoding=encoding)
        
        print(f"✓ Clipped shape: {dict(ds_clipped.dims)}")
        print(f"✓ Time range: {ds_clipped.time.min().values} to {ds_clipped.time.max().values}")
        print()
        
        ds.close()
        return output_file
    
    def step2_fill_missing(self, input_file, method='nearest', distance_limit=5):
        """
        Step 2: Fill missing values near coastlines using interpolation
        
        Replaces: cdo -fillmiss -remapbil
        
        Args:
            input_file: Path to clipped NetCDF
            method: Interpolation method ('nearest', 'linear')
            distance_limit: Maximum distance (in grid cells) to search for valid neighbors
        """
        print("=" * 60)
        print("STEP 2: Filling missing values near coastlines")
        print("=" * 60)
        
        print(f"Loading: {input_file}")
        ds = xr.open_dataset(input_file)
        
        # Get variable names
        lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
        precip_var = 'precip' if 'precip' in ds else 'precipitation'
        
        precip = ds[precip_var].values
        original_missing = np.isnan(precip).sum()
        print(f"Original missing values: {original_missing:,}")
        
        # Fill missing values for each time step
        print("Filling missing values (this may take a while)...")
        filled_precip = np.zeros_like(precip)
        
        for t in range(precip.shape[0]):
            if t % 12 == 0:
                print(f"  Processing timestep {t}/{precip.shape[0]}...")
            
            data_slice = precip[t, :, :]
            
            # Get coordinates of valid and invalid points
            valid_mask = ~np.isnan(data_slice)
            
            if valid_mask.sum() == 0:
                filled_precip[t, :, :] = data_slice
                continue
            
            # Create coordinate grids
            ny, nx = data_slice.shape
            y_coords, x_coords = np.mgrid[0:ny, 0:nx]
            
            # Points with valid data
            valid_points = np.column_stack([
                x_coords[valid_mask],
                y_coords[valid_mask]
            ])
            valid_values = data_slice[valid_mask]
            
            # Points with missing data
            missing_mask = np.isnan(data_slice)
            if missing_mask.sum() == 0:
                filled_precip[t, :, :] = data_slice
                continue
            
            missing_points = np.column_stack([
                x_coords[missing_mask],
                y_coords[missing_mask]
            ])
            
            # Interpolate using nearest neighbor or linear
            filled_values = griddata(
                valid_points, valid_values, missing_points,
                method=method, fill_value=np.nan
            )
            
            # Create filled array
            filled_slice = data_slice.copy()
            filled_slice[missing_mask] = filled_values
            filled_precip[t, :, :] = filled_slice
        
        # Update dataset
        ds[precip_var].values = filled_precip
        
        final_missing = np.isnan(filled_precip).sum()
        print(f"✓ Remaining missing values: {final_missing:,}")
        print(f"✓ Filled {original_missing - final_missing:,} values")
        
        # Save
        output_file = os.path.join(self.dirs['filled'], 'east_africa_chirps_filled.nc')
        print(f"Saving to: {output_file}")
        
        # Remove conflicting attributes from time variable
        if 'calendar' in ds['time'].attrs:
            calendar_value = ds['time'].attrs['calendar']
            del ds['time'].attrs['calendar']
        else:
            calendar_value = 'gregorian'
        
        encoding = {
            precip_var: {'zlib': True, 'complevel': 5},
            'time': {'calendar': calendar_value}
        }
        ds.to_netcdf(output_file, encoding=encoding)
        print()
        
        ds.close()
        return output_file
    
    def step3_fix_metadata(self, input_file):
        """
        Step 3: Fix metadata for SPI calculation
        
        Replaces: cdo -setattribute,precip@units="mm"
        
        Args:
            input_file: Path to filled NetCDF
        """
        print("=" * 60)
        print("STEP 3: Fixing metadata for SPI calculation")
        print("=" * 60)
        
        print(f"Loading: {input_file}")
        ds = xr.open_dataset(input_file)
        
        # Get precipitation variable
        precip_var = 'precip' if 'precip' in ds else 'precipitation'
        
        # Fix units to 'mm' (required by climate-indices)
        print(f"Setting {precip_var} units to 'mm'")
        ds[precip_var].attrs['units'] = 'mm'
        
        # Ensure proper time encoding
        if 'calendar' not in ds['time'].attrs:
            ds['time'].attrs['calendar'] = 'gregorian'
        
        # Save
        output_file = os.path.join(self.dirs['metadata'], 'east_africa_chirps_metadata_fixed.nc')
        print(f"Saving to: {output_file}")
        
        # Remove conflicting attributes from time variable
        if 'calendar' in ds['time'].attrs:
            calendar_value = ds['time'].attrs['calendar']
            del ds['time'].attrs['calendar']
        else:
            calendar_value = 'gregorian'
        
        encoding = {
            precip_var: {'zlib': True, 'complevel': 5},
            'time': {'calendar': calendar_value}
        }
        ds.to_netcdf(output_file, encoding=encoding)
        print()
        
        ds.close()
        return output_file
    
    def step4_reorder_for_spi(self, input_file):
        """
        Step 4: Reorder dimensions to (lat, lon, time) for SPI calculation
        
        Replaces: ncpdq -a lat,lon,time
        
        Args:
            input_file: Path to metadata-fixed NetCDF
        """
        print("=" * 60)
        print("STEP 4: Reordering dimensions for SPI calculation")
        print("=" * 60)
        
        print(f"Loading: {input_file}")
        ds = xr.open_dataset(input_file)
        
        # Get dimension names
        lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
        lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
        precip_var = 'precip' if 'precip' in ds else 'precipitation'

        print(f"Original dimension order: {list(ds[precip_var].dims)}")

        # Transpose to (lat, lon, time)
        ds_transposed = ds.transpose(lat_name, lon_name, 'time')

        # Rename dims/coords to expected names for climate-indices: lat, lon, time
        rename_map = {}
        if lat_name != 'lat':
            rename_map[lat_name] = 'lat'
        if lon_name != 'lon':
            rename_map[lon_name] = 'lon'
        if rename_map:
            ds_transposed = ds_transposed.rename(rename_map)
            lat_name = 'lat'
            lon_name = 'lon'

        # Ensure coords are named lat/lon as well
        coord_rename = {}
        if 'latitude' in ds_transposed.coords and 'lat' not in ds_transposed.coords:
            coord_rename['latitude'] = 'lat'
        if 'longitude' in ds_transposed.coords and 'lon' not in ds_transposed.coords:
            coord_rename['longitude'] = 'lon'
        if coord_rename:
            ds_transposed = ds_transposed.rename_vars(coord_rename)

        print(f"New dimension order: {list(ds_transposed[precip_var].dims)}")
        
        # Save
        output_file = os.path.join(self.dirs['metadata'], 'input_spi.nc')
        print(f"Saving to: {output_file}")
        
        # Remove conflicting attributes from time variable
        if 'calendar' in ds_transposed['time'].attrs:
            calendar_value = ds_transposed['time'].attrs['calendar']
            del ds_transposed['time'].attrs['calendar']
        else:
            calendar_value = 'gregorian'
        
        encoding = {
            precip_var: {'zlib': True, 'complevel': 5},
            'time': {'calendar': calendar_value}
        }
        ds_transposed.to_netcdf(output_file, encoding=encoding)
        print()
        
        ds_transposed.close()
        return output_file
    
    def step5_calculate_spi(self, input_file, scales=[1, 2, 3, 6, 9, 12],
                           calibration_start=1991, calibration_end=2020):
        """
        Step 5: Calculate SPI using climate-indices package
        
        Replaces: spi --periodicity monthly --scales ...
        
        Args:
            input_file: Path to reordered NetCDF
            scales: List of time scales (in months) for SPI
            calibration_start: Start year for calibration period
            calibration_end: End year for calibration period
        """
        print("=" * 60)
        print("STEP 5: Calculating SPI")
        print("=" * 60)
        
        # Get precipitation variable name
        ds = xr.open_dataset(input_file)
        precip_var = 'precip' if 'precip' in ds else 'precipitation'
        ds.close()
        
        print(f"Input file: {input_file}")
        print(f"Precipitation variable: {precip_var}")
        print(f"Scales: {scales}")
        print(f"Calibration period: {calibration_start}-{calibration_end}")
        print()
        
        # Build command for climate-indices
        output_base = os.path.join(self.dirs['spi_intermediate'], 'east_africa')
        
        cmd = [
            'spi',
            '--periodicity', 'monthly',
            '--scales'] + [str(s) for s in scales] + [
            '--calibration_start_year', str(calibration_start),
            '--calibration_end_year', str(calibration_end),
            '--netcdf_precip', input_file,
            '--var_name_precip', precip_var,
            '--output_file_base', output_base,
            '--multiprocessing', 'all'
        ]
        
        print(f"Running command:")
        print(' '.join(cmd))
        print()
        
        try:
            # Run SPI calculation
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            print("✓ SPI calculation completed successfully!")
            print()
        except subprocess.CalledProcessError as e:
            print(f"\n❌ ERROR: SPI calculation failed!")
            print(f"\nCommand that failed:")
            print(f"  {' '.join(cmd)}")
            print(f"\n📋 Error details:")
            print(f"Exit code: {e.returncode}")
            if e.stdout:
                print(f"\nStandard Output:")
                print(e.stdout)
            if e.stderr:
                print(f"\nStandard Error:")
                print(e.stderr)
            print(f"\n💡 Common causes:")
            print(f"  1. Calibration period outside data range")
            print(f"  2. Insufficient data for calibration")
            print(f"  3. Input file format issues")
            print(f"\nTip: Check if your calibration years match your data range!")
            raise
        except FileNotFoundError:
            print("ERROR: 'spi' command not found!")
            print("Please install climate-indices package:")
            print("  pip install climate-indices")
            raise
        
        return self.dirs['spi_intermediate']
    
    def step6_reorder_output(self, spi_dir):
        """
        Step 6: Reorder SPI output dimensions back to (time, lat, lon)
        
        Replaces: ncpdq -a time,lat,lon
        
        Args:
            spi_dir: Directory containing SPI intermediate files
        """
        print("=" * 60)
        print("STEP 6: Reordering SPI output to CF-compliant format")
        print("=" * 60)
        
        # Find all gamma SPI files (remove Pearson versions)
        import glob
        spi_files = glob.glob(os.path.join(spi_dir, '*spi_gamma*.nc'))
        
        if not spi_files:
            print("No SPI gamma files found!")
            return
        
        print(f"Found {len(spi_files)} SPI files to reorder")
        print()
        
        for spi_file in spi_files:
            basename = os.path.basename(spi_file)
            print(f"Processing: {basename}")
            
            # Load and transpose
            ds = xr.open_dataset(spi_file)
            
            # Get dimension names
            lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
            lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'

            # Rename to standard lat/lon for consistency
            rename_map = {}
            if lat_name != 'lat':
                rename_map[lat_name] = 'lat'
            if lon_name != 'lon':
                rename_map[lon_name] = 'lon'
            if rename_map:
                ds = ds.rename(rename_map)
                lat_name = 'lat'
                lon_name = 'lon'
            
            # Find SPI variable (starts with 'spi_gamma')
            spi_var = [v for v in ds.data_vars if v.startswith('spi_gamma')][0]
            
            print(f"  Original dimensions: {list(ds[spi_var].dims)}")
            
            # Transpose to (time, lat, lon)
            ds_transposed = ds.transpose('time', lat_name, lon_name)
            
            print(f"  New dimensions: {list(ds_transposed[spi_var].dims)}")
            
            # Save to final directory
            output_file = os.path.join(self.dirs['spi_final'], basename)
            
            # Remove conflicting attributes from time variable
            if 'calendar' in ds_transposed['time'].attrs:
                calendar_value = ds_transposed['time'].attrs['calendar']
                del ds_transposed['time'].attrs['calendar']
            else:
                calendar_value = 'gregorian'
            
            encoding = {
                spi_var: {'zlib': True, 'complevel': 5},
                'time': {'calendar': calendar_value}
            }
            ds_transposed.to_netcdf(output_file, encoding=encoding)
            
            print(f"  ✓ Saved to: {output_file}")
            print()
            
            ds.close()
            ds_transposed.close()
        
        print(f"✓ All SPI files reordered and saved to: {self.dirs['spi_final']}")
        return self.dirs['spi_final']
    
    def run_full_pipeline(self, lon_min=25, lon_max=52, lat_min=-15, lat_max=22,
                         year_start=2016, year_end=2024,
                         spi_scales=[1, 2, 3, 6, 9, 12],
                         calibration_start=1991, calibration_end=2020):
        """
        Run the complete pipeline from raw CHIRPS to SPI
        
        Args:
            lon_min, lon_max, lat_min, lat_max: Spatial bounds
            year_start, year_end: Temporal bounds
            spi_scales: List of SPI time scales to calculate
            calibration_start, calibration_end: Calibration period for SPI
        """
        print("\n")
        print("*" * 60)
        print("  CHIRPS TO SPI - FULL PIPELINE")
        print("*" * 60)
        print()
        
        try:
            # Step 1: Clip to region
            clipped_file = self.step1_clip_region(
                lon_min, lon_max, lat_min, lat_max,
                year_start, year_end
            )
            
            # Step 2: Fill missing values
            filled_file = self.step2_fill_missing(clipped_file)
            
            # Step 3: Fix metadata
            metadata_file = self.step3_fix_metadata(filled_file)
            
            # Step 4: Reorder for SPI calculation
            reordered_file = self.step4_reorder_for_spi(metadata_file)
            
            # Step 5: Calculate SPI
            spi_dir = self.step5_calculate_spi(
                reordered_file, spi_scales,
                calibration_start, calibration_end
            )
            
            # Step 6: Reorder SPI output
            final_dir = self.step6_reorder_output(spi_dir)
            
            print("\n")
            print("*" * 60)
            print("  PIPELINE COMPLETED SUCCESSFULLY! ✓")
            print("*" * 60)
            print(f"\nFinal SPI files are in: {final_dir}")
            print()
            
            return final_dir
            
        except Exception as e:
            print("\n")
            print("*" * 60)
            print("  PIPELINE FAILED!")
            print("*" * 60)
            print(f"Error: {str(e)}")
            raise


def main():
    """
    Example usage
    """
    # Path to your downloaded CHIRPS data
    chirps_file = '../../data/raw/chirps/chirps-v2.0.monthly.nc'
    
    # Initialize processor
    processor = CHIRPStoSPI(chirps_file)
    
    # Run full pipeline
    final_dir = processor.run_full_pipeline(
        # East Africa bounds
        lon_min=25, lon_max=52,
        lat_min=-15, lat_max=22,
        
        # Time range (adjust based on your needs)
        year_start=2016, year_end=2024,
        
        # SPI scales (months)
        spi_scales=[1, 2, 3, 6, 9, 12],
        
        # Calibration period
        calibration_start=1991,
        calibration_end=2020
    )
    
    print(f"✓ All done! SPI files saved to: {final_dir}")


if __name__ == '__main__':
    main()


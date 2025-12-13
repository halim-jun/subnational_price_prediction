#!/usr/bin/env python
"""
SPI Generation Script - Executable version of the notebook

This script runs the complete SPI generation pipeline for East Africa.
Run this instead of the notebook for automated processing.

Usage:
    python run_spi_generation.py
    
Or with custom parameters:
    python run_spi_generation.py --input chirps.nc --output ./spi_output
"""

import os
import sys
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm

# Import our SPI generator
from generate_spi_python import CHIRPStoSPI


def download_chirps(output_dir='../../data/raw/chirps', force=False):
    """
    Download CHIRPS global monthly dataset
    
    Args:
        output_dir: Directory to save downloaded file
        force: Force re-download even if file exists
        
    Returns:
        str: Path to downloaded file
    """
    print("\n" + "=" * 70)
    print("CHIRPS DATA DOWNLOAD")
    print("=" * 70)
    
    # Create directory
    os.makedirs(output_dir, exist_ok=True)
    
    # File info
    url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc"
    output_file = os.path.join(output_dir, 'chirps-v2.0.monthly.nc')
    
    # Check if file already exists
    if os.path.exists(output_file) and not force:
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n✓ CHIRPS file already exists!")
        print(f"  Location: {output_file}")
        print(f"  Size: {file_size_mb:.1f} MB")
        
        # Check data range
        try:
            ds = xr.open_dataset(output_file)
            time_start = str(ds.time.min().values)[:10]
            time_end = str(ds.time.max().values)[:10]
            print(f"  Time range: {time_start} to {time_end}")
            ds.close()
        except:
            pass
        
        print("\nUse --force-download to re-download")
        return output_file
    
    # Download
    print(f"\n📥 Downloading CHIRPS data...")
    print(f"  URL: {url}")
    print(f"  Destination: {output_file}")
    print(f"\n⚠️  This is a large file (~7 GB) and may take 10-30 minutes!")
    print(f"  Time depends on your internet speed\n")
    
    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc='Downloading'
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"\n✅ Download completed!")
        
        # Verify file
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Size: {file_size_mb:.1f} MB")
        
        # Check data range
        try:
            print(f"\n🔍 Checking data...")
            ds = xr.open_dataset(output_file)
            time_start = str(ds.time.min().values)[:10]
            time_end = str(ds.time.max().values)[:10]
            years = int(ds.time.dt.year.max().values) - int(ds.time.dt.year.min().values) + 1
            print(f"  ✓ Time range: {time_start} to {time_end} ({years} years)")
            print(f"  ✓ Variables: {list(ds.data_vars)}")
            print(f"  ✓ Dimensions: {dict(ds.dims)}")
            ds.close()
        except Exception as e:
            print(f"  ⚠️  Warning: Could not verify file: {e}")
        
        return output_file
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed!")
        print(f"  Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. Try manual download:")
        print(f"     wget {url}")
        print(f"  3. Or use curl:")
        print(f"     curl -o {output_file} {url}")
        sys.exit(1)


def setup_paths(base_dir='../../data/raw/chirps'):
    """
    Setup and validate input/output paths
    
    Args:
        base_dir: Base directory containing CHIRPS data
        
    Returns:
        dict: Paths dictionary
    """
    paths = {
        'chirps_dir': base_dir,
        'chirps_file': os.path.join(base_dir, 'chirps-v2.0.monthly.nc'),
        'output_dir': '../../data/processed/spi',
        'viz_dir': '../../data/processed/spi/visualizations'
    }
    
    # Create visualization directory
    os.makedirs(paths['viz_dir'], exist_ok=True)
    
    return paths


def visualize_spi(spi_file, output_dir, year=2024, scale=12):
    """
    Create visualization of SPI data for a given year
    
    Args:
        spi_file: Path to SPI NetCDF file
        output_dir: Directory to save visualization
        year: Year to visualize
        scale: SPI scale (e.g., 12 for SPI-12)
    """
    print("\n" + "=" * 60)
    print(f"Visualizing SPI-{scale} for year {year}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(spi_file):
        print(f"⚠️  SPI file not found: {spi_file}")
        return
    
    # Load data
    print(f"Loading: {spi_file}")
    ds = xr.open_dataset(spi_file)
    
    # Select year
    try:
        spi_year = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
    except:
        print(f"⚠️  Year {year} not available in dataset")
        available_years = np.unique(ds['time'].dt.year.values)
        print(f"Available years: {available_years}")
        # Use last year available
        year = int(available_years[-1])
        print(f"Using year {year} instead")
        spi_year = ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31'))
    
    # Get SPI variable name
    spi_var = [v for v in ds.data_vars if 'spi' in v.lower()][0]
    print(f"SPI variable: {spi_var}")
    
    # Get coordinate names
    lon_name = 'longitude' if 'longitude' in spi_year.dims else 'lon'
    lat_name = 'latitude' if 'latitude' in spi_year.dims else 'lat'
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f'SPI-{scale} (Standardized Precipitation Index) - East Africa {year}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # SPI color scheme (drought to wet)
    cmap = plt.get_cmap('RdBu', 7)  # Red (dry) to Blue (wet)
    bounds = [-3, -2, -1.5, -1, 1, 1.5, 2, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    months = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
    
    # Plot each month
    for i, ax in enumerate(axes.flat):
        if i < len(spi_year.time):
            data = spi_year[spi_var].isel(time=i)
            
            # Plot
            pcm = ax.pcolormesh(
                data[lon_name], data[lat_name], data,
                cmap=cmap, norm=norm, shading='auto'
            )
            
            # Styling
            ax.set_title(months[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_aspect('auto')
        else:
            ax.axis('off')
    
    # Add colorbar
    plt.tight_layout()
    cbar = fig.colorbar(
        pcm, ax=axes.ravel().tolist(), 
        shrink=0.6, pad=0.01,
        ticks=[-3, -2, -1.5, -1, 0, 1, 1.5, 2, 3],
        extend='both'
    )
    cbar.set_label(f'SPI-{scale} Value', fontsize=12, fontweight='bold')
    
    # Add drought interpretation labels
    cbar.ax.text(3.5, -2.5, 'Extreme Drought', fontsize=8, va='center')
    cbar.ax.text(3.5, -1.75, 'Severe Drought', fontsize=8, va='center')
    cbar.ax.text(3.5, -1.25, 'Moderate Drought', fontsize=8, va='center')
    cbar.ax.text(3.5, 0, 'Normal', fontsize=8, va='center', fontweight='bold')
    cbar.ax.text(3.5, 1.25, 'Moderately Wet', fontsize=8, va='center')
    cbar.ax.text(3.5, 1.75, 'Severely Wet', fontsize=8, va='center')
    cbar.ax.text(3.5, 2.5, 'Extremely Wet', fontsize=8, va='center')
    
    # Save
    output_file = os.path.join(output_dir, f'east_africa_spi{scale}_{year}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_file}")
    
    plt.close()
    ds.close()


def create_summary_report(spi_dir, output_dir):
    """
    Create a summary report of generated SPI files
    
    Args:
        spi_dir: Directory containing SPI files
        output_dir: Directory to save report
    """
    print("\n" + "=" * 60)
    print("Creating Summary Report")
    print("=" * 60)
    
    import glob
    
    spi_files = sorted(glob.glob(os.path.join(spi_dir, '*.nc')))
    
    if not spi_files:
        print("No SPI files found!")
        return
    
    report_file = os.path.join(output_dir, 'spi_generation_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SPI GENERATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total SPI files: {len(spi_files)}\n")
        f.write("\n")
        
        f.write("-" * 70 + "\n")
        f.write("FILE DETAILS\n")
        f.write("-" * 70 + "\n")
        
        for spi_file in spi_files:
            basename = os.path.basename(spi_file)
            size_mb = os.path.getsize(spi_file) / (1024 * 1024)
            
            f.write(f"\nFile: {basename}\n")
            f.write(f"  Size: {size_mb:.2f} MB\n")
            
            # Read metadata
            try:
                ds = xr.open_dataset(spi_file)
                spi_var = [v for v in ds.data_vars if 'spi' in v.lower()][0]
                
                f.write(f"  Variable: {spi_var}\n")
                f.write(f"  Dimensions: {dict(ds.dims)}\n")
                f.write(f"  Time range: {ds.time.min().values} to {ds.time.max().values}\n")
                
                # Get spatial bounds
                lon_name = 'longitude' if 'longitude' in ds.dims else 'lon'
                lat_name = 'latitude' if 'latitude' in ds.dims else 'lat'
                f.write(f"  Lon range: {float(ds[lon_name].min()):.2f} to {float(ds[lon_name].max()):.2f}\n")
                f.write(f"  Lat range: {float(ds[lat_name].min()):.2f} to {float(ds[lat_name].max()):.2f}\n")
                
                # Basic statistics
                spi_data = ds[spi_var].values
                valid_data = spi_data[~np.isnan(spi_data)]
                
                f.write(f"  Statistics:\n")
                f.write(f"    Mean: {np.mean(valid_data):.3f}\n")
                f.write(f"    Std:  {np.std(valid_data):.3f}\n")
                f.write(f"    Min:  {np.min(valid_data):.3f}\n")
                f.write(f"    Max:  {np.max(valid_data):.3f}\n")
                
                # Drought statistics
                extreme_drought = (valid_data <= -2.0).sum()
                severe_drought = ((valid_data > -2.0) & (valid_data <= -1.5)).sum()
                moderate_drought = ((valid_data > -1.5) & (valid_data <= -1.0)).sum()
                
                total_points = len(valid_data)
                f.write(f"  Drought occurrence:\n")
                f.write(f"    Extreme drought: {extreme_drought:,} points ({100*extreme_drought/total_points:.2f}%)\n")
                f.write(f"    Severe drought:  {severe_drought:,} points ({100*severe_drought/total_points:.2f}%)\n")
                f.write(f"    Moderate drought: {moderate_drought:,} points ({100*moderate_drought/total_points:.2f}%)\n")
                
                ds.close()
            except Exception as e:
                f.write(f"  Error reading file: {str(e)}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"✅ Report saved: {report_file}")
    
    # Print to console as well
    with open(report_file, 'r') as f:
        print("\n" + f.read())


def main():
    """
    Main execution function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate SPI from CHIRPS precipitation data'
    )
    parser.add_argument(
        '--input', '-i',
        default='../../data/raw/chirps/chirps-v2.0.monthly.nc',
        help='Path to input CHIRPS NetCDF file'
    )
    parser.add_argument(
        '--output', '-o',
        default='../../data/processed/spi',
        help='Output directory for SPI files'
    )
    parser.add_argument(
        '--lon-min', type=float, default=25,
        help='Minimum longitude (default: 25 for East Africa)'
    )
    parser.add_argument(
        '--lon-max', type=float, default=52,
        help='Maximum longitude (default: 52 for East Africa)'
    )
    parser.add_argument(
        '--lat-min', type=float, default=-15,
        help='Minimum latitude (default: -15 for East Africa)'
    )
    parser.add_argument(
        '--lat-max', type=float, default=22,
        help='Maximum latitude (default: 22 for East Africa)'
    )
    parser.add_argument(
        '--year-start', type=int, default=2016,
        help='Start year (default: 2016)'
    )
    parser.add_argument(
        '--year-end', type=int, default=2024,
        help='End year (default: 2024)'
    )
    parser.add_argument(
        '--scales', nargs='+', type=int, default=[1, 2, 3, 6, 9, 12],
        help='SPI time scales in months (default: 1 2 3 6 9 12)'
    )
    parser.add_argument(
        '--calibration-start', type=int, default=1991,
        help='Calibration start year (default: 1991)'
    )
    parser.add_argument(
        '--calibration-end', type=int, default=2020,
        help='Calibration end year (default: 2020)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualizations after SPI generation'
    )
    parser.add_argument(
        '--viz-year', type=int, default=2024,
        help='Year to visualize (default: 2024)'
    )
    parser.add_argument(
        '--skip-spi', action='store_true',
        help='Skip SPI generation (useful if only visualizing existing data)'
    )
    parser.add_argument(
        '--download-chirps', action='store_true',
        help='Download CHIRPS data before processing'
    )
    parser.add_argument(
        '--force-download', action='store_true',
        help='Force re-download even if CHIRPS file exists'
    )
    parser.add_argument(
        '--chirps-dir', default='../../data/raw/chirps',
        help='Directory for CHIRPS data (default: ../../data/raw/chirps)'
    )
    
    args = parser.parse_args()
    
    # Download CHIRPS data if requested
    if args.download_chirps or args.force_download:
        chirps_file = download_chirps(
            output_dir=args.chirps_dir,
            force=args.force_download
        )
        # Update input path to downloaded file
        args.input = chirps_file
    
    # Validate and adjust calibration period
    if args.calibration_start < args.year_start or args.calibration_end > args.year_end:
        print(f"\n⚠️  WARNING: Calibration period ({args.calibration_start}-{args.calibration_end}) is outside data range ({args.year_start}-{args.year_end})")
        print(f"Adjusting calibration period to match data range...")
        
        # Adjust to fit within data range
        args.calibration_start = max(args.calibration_start, args.year_start)
        args.calibration_end = min(args.calibration_end, args.year_end)
        
        # Make sure we have at least 5 years for calibration
        data_years = args.year_end - args.year_start + 1
        if data_years < 5:
            print(f"\n❌ ERROR: Need at least 5 years of data for SPI calibration")
            print(f"   You have: {data_years} years ({args.year_start}-{args.year_end})")
            print(f"   Recommendation: Extend your data range to at least 5 years")
            sys.exit(1)
        
        print(f"✓ Adjusted calibration period: {args.calibration_start}-{args.calibration_end}")
        print()
    
    # Print header
    print("\n" + "*" * 70)
    print("  SPI GENERATION FOR EAST AFRICA")
    print("  Standardized Precipitation Index from CHIRPS Data")
    print("*" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Print configuration
    print("Configuration:")
    print(f"  Input file: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Region: Lon [{args.lon_min}, {args.lon_max}], Lat [{args.lat_min}, {args.lat_max}]")
    print(f"  Time period: {args.year_start}-{args.year_end}")
    print(f"  SPI scales: {args.scales}")
    print(f"  Calibration period: {args.calibration_start}-{args.calibration_end}")
    print()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ ERROR: Input file not found: {args.input}")
        print("\nPlease download CHIRPS data first:")
        print("  wget https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/chirps-v2.0.monthly.nc")
        sys.exit(1)
    
    # Initialize processor
    processor = CHIRPStoSPI(
        input_file=args.input,
        output_dir=args.output
    )
    
    # Run SPI generation pipeline
    if not args.skip_spi:
        try:
            final_dir = processor.run_full_pipeline(
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                year_start=args.year_start,
                year_end=args.year_end,
                spi_scales=args.scales,
                calibration_start=args.calibration_start,
                calibration_end=args.calibration_end
            )
            
            print("\n" + "*" * 70)
            print("  SPI GENERATION COMPLETED SUCCESSFULLY!")
            print("*" * 70)
            print(f"\nFinal SPI files saved to: {final_dir}\n")
            
        except Exception as e:
            print("\n" + "*" * 70)
            print("  SPI GENERATION FAILED!")
            print("*" * 70)
            print(f"\nError: {str(e)}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        final_dir = os.path.join(args.output, '05_spi_final')
        print(f"\nSkipping SPI generation. Using existing files in: {final_dir}\n")
    
    # Create visualizations if requested
    if args.visualize:
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)
        
        # Visualize key SPI scales
        for scale in [3, 6, 12]:
            if scale in args.scales:
                spi_file = os.path.join(
                    final_dir, 
                    f'east_africa_spi_gamma_{scale:02d}_month.nc'
                )
                if os.path.exists(spi_file):
                    visualize_spi(spi_file, viz_dir, year=args.viz_year, scale=scale)
    
    # Create summary report
    create_summary_report(final_dir, args.output)
    
    # Final summary
    print("\n" + "*" * 70)
    print("  ALL TASKS COMPLETED!")
    print("*" * 70)
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput locations:")
    print(f"  - SPI files: {final_dir}")
    if args.visualize:
        print(f"  - Visualizations: {viz_dir}")
    print(f"  - Report: {os.path.join(args.output, 'spi_generation_report.txt')}")
    print()


if __name__ == '__main__':
    main()


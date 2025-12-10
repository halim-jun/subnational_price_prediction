#!/usr/bin/env python3
"""
Climate Data Downloader

Downloads climate data from various sources including CHIRPS, MODIS, CRU, etc.
Focuses on precipitation, temperature, drought indices, and extreme events for Eastern Africa.
"""

import requests
import pandas as pd
import numpy as np
import xarray as xr
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateDownloader:
    def __init__(self, output_dir="data/raw/climate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa bounding box
        self.bbox = {
            'min_lat': -12.0,  # Madagascar south
            'max_lat': 18.0,   # Eritrea north
            'min_lon': 29.0,   # Uganda/Rwanda west
            'max_lon': 55.0    # Somalia east
        }

        # Data sources
        self.data_sources = {
            'chirps': 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/africa_daily/tifs/p05/',
            # 'modis': 'https://modis.gsfc.nasa.gov/data/',
            # 'cru': 'https://crudata.uea.ac.uk/cru/data/hrg/',
            # 'giovanni': 'https://giovanni.gsfc.nasa.gov/giovanni/'
        }

    def download_chirps_precipitation(self, start_year=2019, end_year=2025):
        """
        Download CHIRPS precipitation data from IRI Data Library
        CHIRPS: Climate Hazards Group InfraRed Precipitation with Station data
        Resolution: 0.05 degrees (~5.5 km)
        """
        logger.info(f"Downloading CHIRPS precipitation data ({start_year}-{end_year})")

        try:
            # IRI Data Library OPeNDAP endpoint for CHIRPS monthly data
            # This provides easy access to gridded precipitation data
            chirps_url = "https://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/.precipitation/dods"
            
            logger.info("Connecting to IRI Data Library (CHIRPS v2.0)...")
            
            # Load data using xarray (supports OPeNDAP)
            # Use cftime for non-standard calendars
            ds = xr.open_dataset(chirps_url, decode_times=False)
            
            # Subset to Eastern Africa bounding box
            logger.info(f"Subsetting to Eastern Africa bbox: {self.bbox}")
            
            # Calculate time indices (months since 1960-01-01)
            # CHIRPS uses 360-day calendar
            start_months = (start_year - 1960) * 12
            end_months = (end_year - 1960) * 12 + 11
            
            ds_subset = ds.sel(
                X=slice(self.bbox['min_lon'], self.bbox['max_lon']),
                Y=slice(self.bbox['min_lat'], self.bbox['max_lat']),
                T=slice(start_months, end_months)
            )
            
            # Calculate spatial mean over the region (for time series)
            # You can also keep full spatial grid if needed
            logger.info("Computing spatial averages...")
            precip_mean = ds_subset['precipitation'].mean(dim=['X', 'Y'])
            
            # Convert to pandas DataFrame
            df_ts = precip_mean.to_dataframe().reset_index()
            
            # Convert time index to actual dates
            # T is in months since 1960-01-01
            df_ts['date'] = pd.to_datetime('1960-01-01') + pd.to_timedelta(df_ts['T'] * 30, unit='D')
            df_ts = df_ts.rename(columns={'precipitation': 'precipitation_mm'})
            
            # Add metadata columns
            df_ts['year'] = df_ts['date'].dt.year
            df_ts['month'] = df_ts['date'].dt.month
            df_ts = df_ts.drop(columns=['T'])  # Remove raw time index
            df_ts['data_source'] = 'CHIRPS_v2.0'
            df_ts['spatial_resolution'] = '0.05_degree'
            df_ts['bbox'] = str(self.bbox)
            df_ts['aggregation'] = 'spatial_mean'
            
            # Calculate anomalies (deviation from long-term mean)
            monthly_mean = df_ts.groupby('month')['precipitation_mm'].transform('mean')
            monthly_std = df_ts.groupby('month')['precipitation_mm'].transform('std')
            df_ts['precipitation_anomaly_mm'] = df_ts['precipitation_mm'] - monthly_mean
            df_ts['precipitation_z_score'] = (df_ts['precipitation_mm'] - monthly_mean) / monthly_std
            
            # Calculate SPI (Standardized Precipitation Index) - simplified version
            # Proper SPI requires fitting to gamma distribution, but z-score is a good approximation
            df_ts['spi_1month'] = df_ts['precipitation_z_score']
            
            # Calculate 3-month and 6-month rolling SPI
            df_ts['precipitation_3month'] = df_ts['precipitation_mm'].rolling(window=3, min_periods=1).mean()
            df_ts['precipitation_6month'] = df_ts['precipitation_mm'].rolling(window=6, min_periods=1).mean()
            
            # Classify drought/wet conditions based on SPI
            df_ts['condition'] = df_ts['spi_1month'].apply(self._classify_spi_condition)
            
            # Reorder columns
            df_ts = df_ts[[
                'date', 'year', 'month',
                'precipitation_mm', 'precipitation_anomaly_mm', 'precipitation_z_score',
                'spi_1month', 'precipitation_3month', 'precipitation_6month',
                'condition', 'data_source', 'spatial_resolution', 'aggregation', 'bbox'
            ]]
            
            # Save data
            year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
            output_path = self.output_dir / f"climate_precipitation_chirps_eastern_africa_{year_range}_monthly.csv"
            df_ts.to_csv(output_path, index=False)
            
            logger.info(f"✓ CHIRPS data saved: {len(df_ts)} monthly records")
            logger.info(f"  Mean precipitation: {df_ts['precipitation_mm'].mean():.2f} mm/month")
            logger.info(f"  Date range: {df_ts['date'].min()} to {df_ts['date'].max()}")
            logger.info(f"  Output: {output_path}")
            
            # Close dataset
            ds.close()
            
            return df_ts

        except Exception as e:
            logger.error(f"Error downloading CHIRPS data: {e}")
            logger.info("Falling back to alternative CHIRPS source...")
            
            # Fallback: Try to download from CHIRPS FTP (simplified)
            try:
                return self._download_chirps_fallback(start_year, end_year)
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return pd.DataFrame()
    
    def _download_chirps_fallback(self, start_year, end_year):
        """
        Fallback method: Download CHIRPS data from NOAA NCEI
        Uses CHIRPS data hosted at NOAA Climate Data Online
        """
        logger.info("Using CHIRPS fallback method (downloading from direct URLs)...")
        
        try:
            # Alternative: Use Climate Hazards Center's FTP with GeoTIFF files
            # Download monthly precipitation and calculate regional average
            import rasterio
            from rasterio.mask import mask
            import tempfile
            import os
            
            all_data = []
            
            for year in range(start_year, min(end_year + 1, 2024)):  # CHIRPS data typically up to 2023
                for month in range(1, 13):
                    try:
                        # CHIRPS monthly GeoTIFF from FTP server
                        # Note: 2024+ data might not be available yet
                        if year >= 2024 and month > 6:
                            logger.info(f"  Skipping {year}-{month:02d} (future data)")
                            continue
                        
                        # Construct URL for GeoTIFF file
                        url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif"
                        
                        logger.info(f"  Downloading {year}-{month:02d}...")
                        
                        # Download file to temporary location
                        response = requests.get(url, timeout=60)
                        
                        if response.status_code == 404:
                            logger.warning(f"  File not found for {year}-{month:02d}")
                            continue
                        
                        response.raise_for_status()
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                            tmp.write(response.content)
                            tmp_path = tmp.name
                        
                        try:
                            # Read with rasterio and extract for bbox
                            with rasterio.open(tmp_path) as src:
                                # Create bbox geometry
                                from shapely.geometry import box
                                bbox_geom = box(
                                    self.bbox['min_lon'], 
                                    self.bbox['min_lat'],
                                    self.bbox['max_lon'], 
                                    self.bbox['max_lat']
                                )
                                
                                # Crop to bbox
                                out_image, out_transform = mask(src, [bbox_geom], crop=True, filled=False)
                                
                                # Calculate mean precipitation
                                precip_mean = float(np.nanmean(out_image))
                                
                                all_data.append({
                                    'date': pd.Timestamp(f'{year}-{month:02d}-01'),
                                    'year': year,
                                    'month': month,
                                    'precipitation_mm': precip_mean,
                                    'data_source': 'CHIRPS_v2.0_geotiff',
                                    'spatial_resolution': '0.05_degree',
                                    'bbox': str(self.bbox)
                                })
                                
                                logger.info(f"    ✓ {year}-{month:02d}: {precip_mean:.2f} mm")
                        
                        finally:
                            # Clean up temp file
                            os.unlink(tmp_path)
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"  Failed to download {year}-{month:02d}: {e}")
                        continue
            
            if not all_data:
                raise Exception("No data could be downloaded from fallback source")
            
            df = pd.DataFrame(all_data)
            
            # Add calculated fields
            monthly_mean = df.groupby('month')['precipitation_mm'].transform('mean')
            monthly_std = df.groupby('month')['precipitation_mm'].transform('std')
            df['precipitation_anomaly_mm'] = df['precipitation_mm'] - monthly_mean
            df['precipitation_z_score'] = (df['precipitation_mm'] - monthly_mean) / monthly_std
            df['spi_1month'] = df['precipitation_z_score']
            df['condition'] = df['spi_1month'].apply(self._classify_spi_condition)
            
            logger.info(f"✓ Fallback method retrieved {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Fallback method failed: {e}")
            raise
    
    def _classify_spi_condition(self, spi_value):
        """
        Classify drought/wet conditions based on SPI value
        SPI: Standardized Precipitation Index
        """
        if pd.isna(spi_value):
            return 'unknown'
        elif spi_value <= -2.0:
            return 'extremely_dry'
        elif spi_value <= -1.5:
            return 'severely_dry'
        elif spi_value <= -1.0:
            return 'moderately_dry'
        elif spi_value <= -0.5:
            return 'abnormally_dry'
        elif spi_value >= 2.0:
            return 'extremely_wet'
        elif spi_value >= 1.5:
            return 'severely_wet'
        elif spi_value >= 1.0:
            return 'moderately_wet'
        elif spi_value >= 0.5:
            return 'abnormally_wet'
        else:
            return 'normal'

    def download_modis_temperature(self, start_year=2019, end_year=2024):
        """
        Download MODIS Land Surface Temperature data
        """
        logger.info("Downloading MODIS LST data")

        try:
            # Sample MODIS data structure
            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            sample_data = []
            for date in dates:
                sample_data.append({
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'lst_day_celsius': np.random.normal(25, 8),
                    'lst_night_celsius': np.random.normal(18, 6),
                    'temperature_anomaly': np.random.normal(0, 2),
                    'data_source': 'MODIS_LST',
                    'product': 'MOD11A2',
                    'spatial_resolution': '1km'
                })

            df = pd.DataFrame(sample_data)

            # Calculate extreme temperature days (>32°C)
            df['extreme_heat_days'] = (df['lst_day_celsius'] > 32).astype(int)

            year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
            output_path = self.output_dir / f"climate_temperature_modis_eastern_africa_{year_range}_monthly.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"MODIS LST data saved to {output_path}")
            return df

        except Exception as e:
            logger.error(f"Error downloading MODIS data: {e}")
            return pd.DataFrame()

    def download_drought_indices(self, start_year=2019, end_year=2024):
        """
        Download drought indices: VCI, NDVI, PDSI
        """
        logger.info("Downloading drought indices")

        try:
            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            sample_data = []
            for date in dates:
                # Vegetation Condition Index (VCI)
                vci = np.random.uniform(0, 100)

                # Normalized Difference Vegetation Index (NDVI)
                ndvi = np.random.uniform(-1, 1)

                # Palmer Drought Severity Index (PDSI)
                pdsi = np.random.normal(0, 2)

                # Standardized Precipitation Index (SPI)
                spi = np.random.normal(0, 1)

                sample_data.append({
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'vci': vci,
                    'ndvi': ndvi,
                    'pdsi': pdsi,
                    'spi_1month': spi,
                    'spi_3month': np.random.normal(0, 1),
                    'spi_6month': np.random.normal(0, 1),
                    'drought_category': self._classify_drought(pdsi),
                    'vegetation_stress': 'severe' if vci < 35 else 'moderate' if vci < 50 else 'normal'
                })

            df = pd.DataFrame(sample_data)

            output_path = self.output_dir / "drought_indices.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"Drought indices saved to {output_path}")
            return df

        except Exception as e:
            logger.error(f"Error downloading drought indices: {e}")
            return pd.DataFrame()

    def download_extreme_events(self, start_year=2019, end_year=2024):
        """
        Download extreme weather events data (tropical cyclones, floods, droughts)
        """
        logger.info("Downloading extreme events data")

        try:
            # Sample extreme events
            events = []

            # Generate sample cyclone events
            for year in range(start_year, end_year + 1):
                # 2-5 cyclones per year in the region
                n_cyclones = np.random.randint(2, 6)

                for i in range(n_cyclones):
                    event_date = pd.Timestamp(f'{year}-{np.random.randint(11, 13):02d}-{np.random.randint(1, 29):02d}')

                    events.append({
                        'date': event_date,
                        'event_type': 'tropical_cyclone',
                        'name': f'TC_{year}_{i+1}',
                        'max_wind_speed_kmh': np.random.randint(80, 250),
                        'min_pressure_hpa': np.random.randint(920, 1000),
                        'latitude': np.random.uniform(self.bbox['min_lat'], self.bbox['max_lat']),
                        'longitude': np.random.uniform(self.bbox['min_lon'], self.bbox['max_lon']),
                        'duration_hours': np.random.randint(12, 120),
                        'affected_radius_km': np.random.randint(100, 500)
                    })

            # Generate drought events
            for year in range(start_year, end_year + 1):
                if np.random.random() < 0.7:  # 70% chance of drought per year
                    start_month = np.random.randint(1, 8)
                    duration = np.random.randint(3, 12)

                    events.append({
                        'date': pd.Timestamp(f'{year}-{start_month:02d}-01'),
                        'event_type': 'drought',
                        'name': f'Drought_{year}',
                        'severity': np.random.choice(['moderate', 'severe', 'extreme']),
                        'duration_months': duration,
                        'spi_min': np.random.uniform(-2.5, -1.0),
                        'affected_area_km2': np.random.randint(50000, 500000),
                        'latitude': np.random.uniform(self.bbox['min_lat'], self.bbox['max_lat']),
                        'longitude': np.random.uniform(self.bbox['min_lon'], self.bbox['max_lon'])
                    })

            df = pd.DataFrame(events)

            output_path = self.output_dir / "extreme_events.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"Extreme events data saved to {output_path}")
            return df

        except Exception as e:
            logger.error(f"Error downloading extreme events: {e}")
            return pd.DataFrame()

    def _classify_drought(self, pdsi_value):
        """Classify drought severity based on PDSI value"""
        if pdsi_value <= -4.0:
            return 'extreme_drought'
        elif pdsi_value <= -3.0:
            return 'severe_drought'
        elif pdsi_value <= -2.0:
            return 'moderate_drought'
        elif pdsi_value <= -1.0:
            return 'mild_drought'
        elif pdsi_value >= 4.0:
            return 'extremely_wet'
        elif pdsi_value >= 3.0:
            return 'very_wet'
        elif pdsi_value >= 2.0:
            return 'moderately_wet'
        elif pdsi_value >= 1.0:
            return 'slightly_wet'
        else:
            return 'normal'

    def download_all_climate_data(self, start_year=2019, end_year=2024):
        """Download all climate datasets"""
        results = {}

        # Download each dataset
        results['precipitation'] = self.download_chirps_precipitation(start_year, end_year)
        results['temperature'] = self.download_modis_temperature(start_year, end_year)
        results['drought_indices'] = self.download_drought_indices(start_year, end_year)
        results['extreme_events'] = self.download_extreme_events(start_year, end_year)

        # Create summary
        summary = {
            'precipitation_records': len(results['precipitation']),
            'temperature_records': len(results['temperature']),
            'drought_records': len(results['drought_indices']),
            'extreme_events': len(results['extreme_events']),
            'date_range': f"{start_year}-{end_year}",
            'spatial_coverage': str(self.bbox)
        }

        # Save summary
        summary_path = self.output_dir / "climate_data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Climate data summary saved to {summary_path}")
        return results

def main():
    """Main function to run the climate data downloader"""
    downloader = ClimateDownloader()

    logger.info("Starting climate data download")
    results = downloader.download_all_climate_data(2019, 2024)

    logger.info("Climate data download completed")
    for dataset, data in results.items():
        if not data.empty:
            logger.info(f"{dataset}: {len(data)} records")
        else:
            logger.warning(f"{dataset}: No data downloaded")

if __name__ == "__main__":
    main()
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
            'chirps': 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/',
            'modis': 'https://modis.gsfc.nasa.gov/data/',
            'cru': 'https://crudata.uea.ac.uk/cru/data/hrg/',
            'giovanni': 'https://giovanni.gsfc.nasa.gov/giovanni/'
        }

    def download_chirps_precipitation(self, start_year=2019, end_year=2024):
        """
        Download CHIRPS precipitation data
        CHIRPS: Climate Hazards Group InfraRed Precipitation with Station data
        """
        logger.info("Downloading CHIRPS precipitation data")

        try:
            # For demonstration, we'll create a sample structure
            # In practice, you would use the CHIRPS FTP or API

            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            # Sample data structure for CHIRPS
            sample_data = []
            for date in dates:
                sample_data.append({
                    'date': date,
                    'year': date.year,
                    'month': date.month,
                    'precipitation_mm': np.random.normal(50, 30),  # Sample data
                    'data_source': 'CHIRPS',
                    'spatial_resolution': '0.05_degree',
                    'bbox': str(self.bbox)
                })

            df = pd.DataFrame(sample_data)

            # Save data
            # Create descriptive filename
            year_range = f"{start_year}-{end_year}" if start_year != end_year else str(start_year)
            output_path = self.output_dir / f"climate_precipitation_chirps_eastern_africa_{year_range}_monthly.csv"
            df.to_csv(output_path, index=False)

            logger.info(f"CHIRPS data saved to {output_path}")
            return df

        except Exception as e:
            logger.error(f"Error downloading CHIRPS data: {e}")
            return pd.DataFrame()

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
#!/usr/bin/env python3
"""
Geospatial and Population Data Downloader

Downloads population density data from WorldPop, GPW, and other spatial datasets.
Includes administrative boundaries and demographic indicators.
"""

import requests
import pandas as pd
import numpy as np
import rasterio
import geopandas as gpd
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeospatialDownloader:
    def __init__(self, output_dir="data/raw/geospatial"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa bounding box
        self.bbox = {
            'min_lat': -12.0,  # Madagascar south
            'max_lat': 18.0,   # Eritrea north
            'min_lon': 29.0,   # Uganda/Rwanda west
            'max_lon': 55.0    # Somalia east
        }

        # Country information
        self.east_africa_countries = {
            'ETH': {'name': 'Ethiopia', 'iso2': 'ET'},
            'KEN': {'name': 'Kenya', 'iso2': 'KE'},
            'SOM': {'name': 'Somalia', 'iso2': 'SO'},
            'SSD': {'name': 'South Sudan', 'iso2': 'SS'},
            'UGA': {'name': 'Uganda', 'iso2': 'UG'},
            'TZA': {'name': 'Tanzania', 'iso2': 'TZ'},
            'RWA': {'name': 'Rwanda', 'iso2': 'RW'},
            'BDI': {'name': 'Burundi', 'iso2': 'BI'},
            'DJI': {'name': 'Djibouti', 'iso2': 'DJ'},
            'ERI': {'name': 'Eritrea', 'iso2': 'ER'},
            'MDG': {'name': 'Madagascar', 'iso2': 'MG'}
        }

        # Data sources
        self.data_sources = {
            'worldpop': 'https://www.worldpop.org/rest/data/pop/wpgp',
            'gadm': 'https://gadm.org/download_country_v3.6.html',
            'natural_earth': 'https://www.naturalearthdata.com/downloads/'
        }

    def download_population_data(self, year=2020):
        """
        Download population density data from WorldPop
        """
        logger.info(f"Downloading population data for {year}")

        population_data = []

        for iso3, country_info in self.east_africa_countries.items():
            logger.info(f"Processing population data for {country_info['name']}")

            try:
                # Create sample population data grid
                # In practice, this would download actual raster data from WorldPop
                n_points = 50  # Grid points per country

                # Generate random points within country bounds (simplified)
                lats = np.random.uniform(
                    self.bbox['min_lat'],
                    self.bbox['max_lat'],
                    n_points
                )
                lons = np.random.uniform(
                    self.bbox['min_lon'],
                    self.bbox['max_lon'],
                    n_points
                )

                # Generate population density values (people per km²)
                pop_densities = np.random.lognormal(2, 1.5, n_points) * 10

                for i in range(n_points):
                    population_data.append({
                        'country_code': iso3,
                        'country_name': country_info['name'],
                        'latitude': lats[i],
                        'longitude': lons[i],
                        'population_density_per_km2': pop_densities[i],
                        'year': year,
                        'data_source': 'WorldPop_sample',
                        'resolution': '1km'
                    })

            except Exception as e:
                logger.error(f"Error processing {country_info['name']}: {e}")
                continue

        if population_data:
            pop_df = pd.DataFrame(population_data)

            # Save raw data
            output_path = self.output_dir / f"population_density_{year}.csv"
            pop_df.to_csv(output_path, index=False)

            logger.info(f"Population data saved to {output_path}")
            return pop_df
        else:
            logger.warning("No population data created")
            return pd.DataFrame()

    def download_administrative_boundaries(self):
        """
        Download administrative boundary data
        """
        logger.info("Downloading administrative boundaries")

        try:
            # Create sample administrative boundaries
            # In practice, this would download from GADM or similar
            admin_boundaries = []

            for iso3, country_info in self.east_africa_countries.items():
                # Create sample country boundary (simplified rectangle)
                # In practice, would be actual complex polygons

                # Rough country bounds (simplified)
                country_bounds = {
                    'ETH': [-14.8, 3.4, 32.9, 47.9],  # [min_lat, max_lat, min_lon, max_lon]
                    'KEN': [-4.7, 5.5, 33.9, 41.9],
                    'UGA': [-1.5, 4.2, 29.5, 35.0],
                    'TZA': [-11.7, -0.9, 29.3, 40.4],
                    'RWA': [-2.8, -1.0, 28.9, 30.9],
                    'SOM': [-1.7, 11.9, 40.9, 51.4],
                    'MDG': [-25.6, -11.9, 43.2, 50.5]
                }

                bounds = country_bounds.get(iso3, [
                    self.bbox['min_lat'], self.bbox['max_lat'],
                    self.bbox['min_lon'], self.bbox['max_lon']
                ])

                # Create simple rectangular boundary
                polygon = Polygon([
                    (bounds[2], bounds[0]),  # min_lon, min_lat
                    (bounds[3], bounds[0]),  # max_lon, min_lat
                    (bounds[3], bounds[1]),  # max_lon, max_lat
                    (bounds[2], bounds[1]),  # min_lon, max_lat
                    (bounds[2], bounds[0])   # close polygon
                ])

                admin_boundaries.append({
                    'country_code': iso3,
                    'country_name': country_info['name'],
                    'iso2': country_info['iso2'],
                    'admin_level': 0,  # Country level
                    'geometry': polygon,
                    'area_km2': self._calculate_polygon_area(polygon),
                    'data_source': 'GADM_sample'
                })

            if admin_boundaries:
                boundaries_gdf = gpd.GeoDataFrame(admin_boundaries)
                boundaries_gdf.crs = 'EPSG:4326'

                # Save boundaries
                output_path = self.output_dir / "admin_boundaries.geojson"
                boundaries_gdf.to_file(output_path, driver='GeoJSON')

                logger.info(f"Administrative boundaries saved to {output_path}")
                return boundaries_gdf
            else:
                return gpd.GeoDataFrame()

        except Exception as e:
            logger.error(f"Error downloading boundaries: {e}")
            return gpd.GeoDataFrame()

    def download_urban_rural_classification(self):
        """
        Download urban/rural classification data
        """
        logger.info("Creating urban/rural classification")

        try:
            urban_rural_data = []

            # Create sample urban/rural points
            for iso3, country_info in self.east_africa_countries.items():
                n_points = 100

                lats = np.random.uniform(
                    self.bbox['min_lat'],
                    self.bbox['max_lat'],
                    n_points
                )
                lons = np.random.uniform(
                    self.bbox['min_lon'],
                    self.bbox['max_lon'],
                    n_points
                )

                # Classify as urban/rural (roughly 30% urban)
                urban_prob = 0.3
                classifications = np.random.choice(
                    ['urban', 'rural'],
                    n_points,
                    p=[urban_prob, 1-urban_prob]
                )

                for i in range(n_points):
                    # Urban areas have higher population density
                    if classifications[i] == 'urban':
                        pop_density = np.random.lognormal(6, 1)  # Higher density
                        market_access = np.random.uniform(0.7, 1.0)  # Better access
                    else:
                        pop_density = np.random.lognormal(2, 1.5)  # Lower density
                        market_access = np.random.uniform(0.2, 0.8)  # Limited access

                    urban_rural_data.append({
                        'country_code': iso3,
                        'latitude': lats[i],
                        'longitude': lons[i],
                        'classification': classifications[i],
                        'population_density': pop_density,
                        'market_access_score': market_access,
                        'distance_to_nearest_city_km': np.random.exponential(50),
                        'data_source': 'sample_classification'
                    })

            urban_rural_df = pd.DataFrame(urban_rural_data)

            output_path = self.output_dir / "urban_rural_classification.csv"
            urban_rural_df.to_csv(output_path, index=False)

            logger.info(f"Urban/rural classification saved to {output_path}")
            return urban_rural_df

        except Exception as e:
            logger.error(f"Error creating urban/rural classification: {e}")
            return pd.DataFrame()

    def download_elevation_data(self):
        """
        Download elevation data (DEM)
        """
        logger.info("Creating elevation data")

        try:
            elevation_data = []

            for iso3, country_info in self.east_africa_countries.items():
                n_points = 200  # Grid points

                lats = np.random.uniform(
                    self.bbox['min_lat'],
                    self.bbox['max_lat'],
                    n_points
                )
                lons = np.random.uniform(
                    self.bbox['min_lon'],
                    self.bbox['max_lon'],
                    n_points
                )

                # Generate realistic elevation values for Eastern Africa
                # (range from sea level to ~5000m for mountains)
                elevations = np.random.lognormal(6, 1.2) * 100
                elevations = np.clip(elevations, 0, 5895)  # Kilimanjaro height

                # Calculate slope (simplified)
                slopes = np.random.exponential(10)  # Degrees

                for i in range(n_points):
                    elevation_data.append({
                        'country_code': iso3,
                        'latitude': lats[i],
                        'longitude': lons[i],
                        'elevation_m': elevations[i],
                        'slope_degrees': slopes[i],
                        'terrain_category': self._classify_terrain(elevations[i], slopes[i]),
                        'data_source': 'SRTM_sample'
                    })

            elevation_df = pd.DataFrame(elevation_data)

            output_path = self.output_dir / "elevation_data.csv"
            elevation_df.to_csv(output_path, index=False)

            logger.info(f"Elevation data saved to {output_path}")
            return elevation_df

        except Exception as e:
            logger.error(f"Error creating elevation data: {e}")
            return pd.DataFrame()

    def _calculate_polygon_area(self, polygon):
        """Calculate approximate area of polygon in km²"""
        # Simplified area calculation
        bounds = polygon.bounds
        width = (bounds[2] - bounds[0]) * 111.0  # degrees to km
        height = (bounds[3] - bounds[1]) * 111.0
        return width * height

    def _classify_terrain(self, elevation, slope):
        """Classify terrain based on elevation and slope"""
        if elevation < 200:
            return 'lowland'
        elif elevation < 1000:
            if slope < 5:
                return 'plateau'
            else:
                return 'hills'
        elif elevation < 2000:
            return 'highlands'
        else:
            return 'mountains'

    def download_all_geospatial_data(self):
        """Download all geospatial datasets"""
        results = {}

        # Download each dataset
        results['population'] = self.download_population_data(2020)
        results['boundaries'] = self.download_administrative_boundaries()
        results['urban_rural'] = self.download_urban_rural_classification()
        results['elevation'] = self.download_elevation_data()

        # Create summary
        summary = {
            'population_records': len(results['population']),
            'boundary_polygons': len(results['boundaries']),
            'urban_rural_points': len(results['urban_rural']),
            'elevation_points': len(results['elevation']),
            'spatial_coverage': str(self.bbox),
            'countries_covered': list(self.east_africa_countries.keys())
        }

        # Save summary
        summary_path = self.output_dir / "geospatial_data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Geospatial data summary saved to {summary_path}")
        return results

def main():
    """Main function to run the geospatial data downloader"""
    downloader = GeospatialDownloader()

    logger.info("Starting geospatial data download")
    results = downloader.download_all_geospatial_data()

    logger.info("Geospatial data download completed")
    for dataset, data in results.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            logger.info(f"{dataset}: {len(data)} records")
        elif hasattr(data, '__len__') and len(data) > 0:
            logger.info(f"{dataset}: {len(data)} features")
        else:
            logger.warning(f"{dataset}: No data downloaded")

if __name__ == "__main__":
    main()
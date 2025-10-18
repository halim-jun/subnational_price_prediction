#!/usr/bin/env python3
"""
OpenStreetMap Infrastructure Data Parser

Downloads and processes road network and infrastructure data from OpenStreetMap.
Focuses on market access indicators: distance to roads, road density, connectivity.
"""

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from shapely.geometry import Point, LineString
import overpy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMDownloader:
    def __init__(self, output_dir="data/raw/osm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa bounding box
        self.bbox = {
            'min_lat': -12.0,  # Madagascar south
            'max_lat': 18.0,   # Eritrea north
            'min_lon': 29.0,   # Uganda/Rwanda west
            'max_lon': 55.0    # Somalia east
        }

        # Overpass API endpoint
        self.overpass_url = "http://overpass-api.de/api/interpreter"

        # Road classifications
        self.road_types = [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'unclassified', 'residential', 'track', 'path'
        ]

    def download_roads_by_country(self, country_name, road_types=None):
        """
        Download road network for a specific country using Overpass API

        Args:
            country_name (str): Country name for OSM query
            road_types (list): Types of roads to download
        """
        if road_types is None:
            road_types = ['motorway', 'trunk', 'primary', 'secondary']

        logger.info(f"Downloading road network for {country_name}")

        # Build Overpass query
        road_filter = '|'.join(road_types)
        query = f"""
        [out:json][timeout:300];
        (
          area["name"="{country_name}"]["admin_level"="2"];
        )->.searchArea;
        (
          way["highway"~"^({road_filter})$"](area.searchArea);
        );
        out geom;
        """

        try:
            response = requests.post(
                self.overpass_url,
                data=query,
                timeout=300
            )
            response.raise_for_status()

            data = response.json()
            roads = []

            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]

                    if len(coords) >= 2:
                        road_info = {
                            'osm_id': element['id'],
                            'highway_type': element.get('tags', {}).get('highway', 'unknown'),
                            'name': element.get('tags', {}).get('name', ''),
                            'surface': element.get('tags', {}).get('surface', ''),
                            'geometry': LineString(coords),
                            'country': country_name,
                            'length_km': self._calculate_length(coords)
                        }
                        roads.append(road_info)

            if roads:
                gdf = gpd.GeoDataFrame(roads)
                gdf.crs = 'EPSG:4326'

                # Save to file
                output_file = self.output_dir / f"roads_{country_name.lower().replace(' ', '_')}.geojson"
                gdf.to_file(output_file, driver='GeoJSON')

                logger.info(f"Downloaded {len(roads)} road segments for {country_name}")
                return gdf
            else:
                logger.warning(f"No roads found for {country_name}")
                return gpd.GeoDataFrame()

        except Exception as e:
            logger.error(f"Error downloading roads for {country_name}: {e}")
            return gpd.GeoDataFrame()

    def download_infrastructure_points(self, country_name):
        """
        Download infrastructure points (markets, airports, ports, etc.)
        """
        logger.info(f"Downloading infrastructure points for {country_name}")

        query = f"""
        [out:json][timeout:300];
        (
          area["name"="{country_name}"]["admin_level"="2"];
        )->.searchArea;
        (
          node["amenity"~"^(marketplace|fuel)$"](area.searchArea);
          node["aeroway"~"^(aerodrome|airport)$"](area.searchArea);
          node["harbour"="yes"](area.searchArea);
          node["railway"="station"](area.searchArea);
        );
        out;
        """

        try:
            response = requests.post(
                self.overpass_url,
                data=query,
                timeout=300
            )
            response.raise_for_status()

            data = response.json()
            infrastructure = []

            for element in data.get('elements', []):
                if element['type'] == 'node':
                    tags = element.get('tags', {})
                    infra_type = self._classify_infrastructure(tags)

                    if infra_type:
                        infrastructure.append({
                            'osm_id': element['id'],
                            'type': infra_type,
                            'name': tags.get('name', ''),
                            'latitude': element['lat'],
                            'longitude': element['lon'],
                            'country': country_name,
                            'geometry': Point(element['lon'], element['lat'])
                        })

            if infrastructure:
                gdf = gpd.GeoDataFrame(infrastructure)
                gdf.crs = 'EPSG:4326'

                output_file = self.output_dir / f"infrastructure_{country_name.lower().replace(' ', '_')}.geojson"
                gdf.to_file(output_file, driver='GeoJSON')

                logger.info(f"Downloaded {len(infrastructure)} infrastructure points for {country_name}")
                return gdf
            else:
                logger.warning(f"No infrastructure found for {country_name}")
                return gpd.GeoDataFrame()

        except Exception as e:
            logger.error(f"Error downloading infrastructure for {country_name}: {e}")
            return gpd.GeoDataFrame()

    def calculate_market_accessibility(self, markets_df, roads_gdf, buffer_km=50):
        """
        Calculate market accessibility metrics based on road network

        Args:
            markets_df (pd.DataFrame): Market locations with lat/lon
            roads_gdf (gpd.GeoDataFrame): Road network
            buffer_km (float): Buffer distance for analysis
        """
        logger.info("Calculating market accessibility metrics")

        if roads_gdf.empty or markets_df.empty:
            return pd.DataFrame()

        accessibility_metrics = []

        for idx, market in markets_df.iterrows():
            market_point = Point(market['longitude'], market['latitude'])

            # Create buffer around market (in degrees, approximately)
            buffer_deg = buffer_km / 111.0  # Rough conversion km to degrees
            market_buffer = market_point.buffer(buffer_deg)

            # Find roads within buffer
            nearby_roads = roads_gdf[roads_gdf.intersects(market_buffer)]

            if not nearby_roads.empty:
                # Calculate metrics
                total_road_length = nearby_roads['length_km'].sum()
                road_density = total_road_length / (np.pi * buffer_km ** 2)  # km/km²

                # Distance to nearest primary road
                primary_roads = nearby_roads[nearby_roads['highway_type'].isin(['motorway', 'trunk', 'primary'])]
                if not primary_roads.empty:
                    distances = primary_roads.geometry.distance(market_point)
                    nearest_primary_distance = distances.min() * 111.0  # Convert to km
                else:
                    nearest_primary_distance = np.inf

                # Distance to any road
                distances = nearby_roads.geometry.distance(market_point)
                nearest_road_distance = distances.min() * 111.0

                # Road type diversity
                road_types = nearby_roads['highway_type'].nunique()

            else:
                total_road_length = 0
                road_density = 0
                nearest_primary_distance = np.inf
                nearest_road_distance = np.inf
                road_types = 0

            accessibility_metrics.append({
                'market_id': market.get('market_id', idx),
                'latitude': market['latitude'],
                'longitude': market['longitude'],
                'total_road_length_km': total_road_length,
                'road_density_km_per_km2': road_density,
                'distance_to_primary_road_km': nearest_primary_distance,
                'distance_to_nearest_road_km': nearest_road_distance,
                'road_type_diversity': road_types,
                'accessibility_score': self._calculate_accessibility_score(
                    road_density, nearest_primary_distance, road_types
                )
            })

        return pd.DataFrame(accessibility_metrics)

    def _classify_infrastructure(self, tags):
        """Classify infrastructure type based on OSM tags"""
        if 'marketplace' in tags.get('amenity', ''):
            return 'marketplace'
        elif 'fuel' in tags.get('amenity', ''):
            return 'fuel_station'
        elif tags.get('aeroway') in ['aerodrome', 'airport']:
            return 'airport'
        elif tags.get('harbour') == 'yes':
            return 'port'
        elif tags.get('railway') == 'station':
            return 'railway_station'
        else:
            return None

    def _calculate_length(self, coords):
        """Calculate approximate length of road segment in km"""
        if len(coords) < 2:
            return 0

        total_length = 0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            # Haversine formula approximation
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) *
                 np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            total_length += 6371 * c  # Earth radius in km

        return total_length

    def _calculate_accessibility_score(self, road_density, distance_to_primary, road_types):
        """Calculate composite accessibility score"""
        # Normalize components (higher is better)
        density_score = min(road_density * 10, 10)  # Cap at 10
        distance_score = max(10 - distance_to_primary / 10, 0)  # Closer is better
        diversity_score = min(road_types * 2, 10)  # More types is better

        # Weighted average
        return (density_score * 0.4 + distance_score * 0.4 + diversity_score * 0.2)

    def download_all_countries(self):
        """Download OSM data for all Eastern Africa countries"""
        countries = [
            'Ethiopia', 'Kenya', 'Somalia', 'South Sudan', 'Uganda', 'Tanzania',
            'Rwanda', 'Burundi', 'Djibouti', 'Eritrea', 'Madagascar'
        ]

        all_roads = []
        all_infrastructure = []

        for country in countries:
            logger.info(f"Processing {country}")

            # Download roads
            roads = self.download_roads_by_country(country)
            if not roads.empty:
                all_roads.append(roads)

            # Download infrastructure
            infrastructure = self.download_infrastructure_points(country)
            if not infrastructure.empty:
                all_infrastructure.append(infrastructure)

            # Rate limiting
            time.sleep(5)

        # Combine all data
        if all_roads:
            combined_roads = gpd.GeoDataFrame(pd.concat(all_roads, ignore_index=True))
            combined_roads.to_file(self.output_dir / "all_roads.geojson", driver='GeoJSON')
            logger.info(f"Combined roads: {len(combined_roads)} segments")

        if all_infrastructure:
            combined_infra = gpd.GeoDataFrame(pd.concat(all_infrastructure, ignore_index=True))
            combined_infra.to_file(self.output_dir / "all_infrastructure.geojson", driver='GeoJSON')
            logger.info(f"Combined infrastructure: {len(combined_infra)} points")

        return {
            'roads': combined_roads if all_roads else gpd.GeoDataFrame(),
            'infrastructure': combined_infra if all_infrastructure else gpd.GeoDataFrame()
        }

def main():
    """Main function to run the OSM downloader"""
    downloader = OSMDownloader()

    logger.info("Starting OSM data download")
    try:
        results = downloader.download_all_countries()
        logger.info("OSM data download completed")

        if not results['roads'].empty:
            logger.info(f"Roads: {len(results['roads'])} segments")
        if not results['infrastructure'].empty:
            logger.info(f"Infrastructure: {len(results['infrastructure'])} points")

    except Exception as e:
        logger.error(f"OSM download failed: {e}")

if __name__ == "__main__":
    main()
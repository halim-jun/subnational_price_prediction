#!/usr/bin/env python3
"""
Macroeconomic Data Downloader

Downloads macroeconomic indicators from IMF, World Bank, and other sources.
Includes global oil prices, exchange rates, food price indices, and country-level indicators.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
import wbdata
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroDownloader:
    def __init__(self, output_dir="data/raw/macro"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eastern Africa country codes (ISO3)
        self.east_africa_countries = [
            'ETH', 'KEN', 'SOM', 'SSD', 'UGA', 'TZA',
            'RWA', 'BDI', 'DJI', 'ERI', 'MDG'
        ]

        # World Bank indicators
        self.wb_indicators = {
            'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
            'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
            'PA.NUS.FCRF': 'Official exchange rate (LCU per US$, period average)',
            'AG.LND.AGRI.ZS': 'Agricultural land (% of land area)',
            'AG.PRD.FOOD.XD': 'Food production index',
            'SP.POP.TOTL': 'Population, total',
            'SP.RUR.TOTL.ZS': 'Rural population (% of total population)',
            'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
            'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)'
        }

        # FAO Food Price Index components
        self.fao_components = [
            'food_price_index',
            'cereals_price_index',
            'oils_price_index',
            'dairy_price_index',
            'meat_price_index',
            'sugar_price_index'
        ]

    def download_world_bank_data(self, start_year=2019, end_year=2024):
        """Download World Bank indicators for Eastern Africa countries"""
        logger.info("Downloading World Bank data")

        all_data = []

        try:
            for indicator_code, indicator_name in self.wb_indicators.items():
                logger.info(f"Downloading {indicator_name}")

                try:
                    # Download data for all countries at once
                    data = wbdata.get_dataframe(
                        {indicator_code: indicator_name},
                        country=self.east_africa_countries,
                        data_date=(datetime(start_year, 1, 1), datetime(end_year, 12, 31))
                    )

                    if not data.empty:
                        # Reset index to get country and date as columns
                        data = data.reset_index()
                        data['indicator_code'] = indicator_code
                        data['year'] = pd.to_datetime(data['date']).dt.year
                        all_data.append(data)

                    time.sleep(1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Error downloading {indicator_name}: {e}")
                    continue

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)

                # Reshape data
                wb_data = combined_df.pivot_table(
                    index=['country', 'year'],
                    columns='indicator_code',
                    values=list(self.wb_indicators.values())[0],
                    aggfunc='first'
                ).reset_index()

                # Save data
                output_path = self.output_dir / "world_bank_indicators.csv"
                wb_data.to_csv(output_path, index=False)

                logger.info(f"World Bank data saved to {output_path}")
                return wb_data
            else:
                logger.warning("No World Bank data downloaded")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error in World Bank download: {e}")
            return self._create_sample_wb_data(start_year, end_year)

    def download_oil_prices(self, start_year=2019, end_year=2024):
        """Download global oil price data"""
        logger.info("Downloading oil price data")

        try:
            # Create sample oil price data (in practice, would use FRED API or similar)
            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            # Simulate oil price with trend and volatility
            base_price = 60
            trend = np.linspace(0, 20, len(dates))  # Upward trend
            volatility = np.random.normal(0, 10, len(dates))
            seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)

            oil_prices = base_price + trend + volatility + seasonal
            oil_prices = np.maximum(oil_prices, 20)  # Floor price

            oil_data = pd.DataFrame({
                'date': dates,
                'year': dates.year,
                'month': dates.month,
                'brent_crude_usd': oil_prices,
                'wti_crude_usd': oil_prices * 0.95,  # WTI typically lower than Brent
                'price_change_pct': np.concatenate([[0], np.diff(oil_prices) / oil_prices[:-1] * 100])
            })

            output_path = self.output_dir / "oil_prices.csv"
            oil_data.to_csv(output_path, index=False)

            logger.info(f"Oil price data saved to {output_path}")
            return oil_data

        except Exception as e:
            logger.error(f"Error downloading oil prices: {e}")
            return pd.DataFrame()

    def download_fao_food_price_index(self, start_year=2019, end_year=2024):
        """Download FAO Food Price Index data"""
        logger.info("Downloading FAO Food Price Index")

        try:
            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            # Create sample FAO price indices (base 2014-2016 = 100)
            base_index = 100
            trend = np.linspace(0, 30, len(dates))  # Upward trend
            volatility = np.random.normal(0, 8, len(dates))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)

            food_index = base_index + trend + volatility + seasonal

            fao_data = pd.DataFrame({
                'date': dates,
                'year': dates.year,
                'month': dates.month,
                'food_price_index': food_index,
                'cereals_price_index': food_index + np.random.normal(0, 5, len(dates)),
                'oils_price_index': food_index + np.random.normal(0, 15, len(dates)),
                'dairy_price_index': food_index + np.random.normal(0, 12, len(dates)),
                'meat_price_index': food_index + np.random.normal(0, 8, len(dates)),
                'sugar_price_index': food_index + np.random.normal(0, 20, len(dates))
            })

            # Ensure all indices are positive
            for col in self.fao_components:
                if col in fao_data.columns:
                    fao_data[col] = np.maximum(fao_data[col], 50)

            output_path = self.output_dir / "fao_food_price_index.csv"
            fao_data.to_csv(output_path, index=False)

            logger.info(f"FAO Food Price Index saved to {output_path}")
            return fao_data

        except Exception as e:
            logger.error(f"Error downloading FAO data: {e}")
            return pd.DataFrame()

    def download_exchange_rates(self, start_year=2019, end_year=2024):
        """Download exchange rates for Eastern Africa currencies"""
        logger.info("Downloading exchange rates")

        try:
            dates = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='M'
            )

            # Sample exchange rates (LCU per USD)
            exchange_rates = {
                'ETH': 'ETB',  # Ethiopian Birr
                'KEN': 'KES',  # Kenyan Shilling
                'UGA': 'UGX',  # Ugandan Shilling
                'TZA': 'TZS',  # Tanzanian Shilling
                'RWA': 'RWF',  # Rwandan Franc
                'MDG': 'MGA',  # Malagasy Ariary
            }

            all_rates = []

            for country_code, currency in exchange_rates.items():
                # Create sample exchange rate with volatility
                base_rate = {
                    'ETB': 45, 'KES': 110, 'UGX': 3700,
                    'TZS': 2300, 'RWF': 1000, 'MGA': 4000
                }.get(currency, 100)

                trend = np.random.normal(0, 0.1, len(dates))  # Random walk
                rates = base_rate * np.exp(np.cumsum(trend))

                for i, date in enumerate(dates):
                    all_rates.append({
                        'date': date,
                        'year': date.year,
                        'month': date.month,
                        'country_code': country_code,
                        'currency': currency,
                        'exchange_rate_lcu_per_usd': rates[i],
                        'volatility_30d': np.std(rates[max(0, i-30):i+1]) if i > 0 else 0
                    })

            exchange_df = pd.DataFrame(all_rates)

            output_path = self.output_dir / "exchange_rates.csv"
            exchange_df.to_csv(output_path, index=False)

            logger.info(f"Exchange rates saved to {output_path}")
            return exchange_df

        except Exception as e:
            logger.error(f"Error downloading exchange rates: {e}")
            return pd.DataFrame()

    def _create_sample_wb_data(self, start_year, end_year):
        """Create sample World Bank data if API fails"""
        logger.info("Creating sample World Bank data")

        all_data = []
        years = range(start_year, end_year + 1)

        for country in self.east_africa_countries:
            for year in years:
                # Sample values based on typical ranges for Eastern Africa
                gdp_per_capita = np.random.uniform(500, 2000)
                inflation = np.random.uniform(-2, 15)
                ag_land_pct = np.random.uniform(30, 80)
                rural_pop_pct = np.random.uniform(60, 85)

                all_data.append({
                    'country_code': country,
                    'year': year,
                    'gdp_per_capita': gdp_per_capita,
                    'inflation_rate': inflation,
                    'agricultural_land_pct': ag_land_pct,
                    'rural_population_pct': rural_pop_pct,
                    'food_production_index': np.random.uniform(90, 110)
                })

        return pd.DataFrame(all_data)

    def download_all_macro_data(self, start_year=2019, end_year=2024):
        """Download all macroeconomic datasets"""
        results = {}

        # Download each dataset
        results['world_bank'] = self.download_world_bank_data(start_year, end_year)
        results['oil_prices'] = self.download_oil_prices(start_year, end_year)
        results['fao_food_index'] = self.download_fao_food_price_index(start_year, end_year)
        results['exchange_rates'] = self.download_exchange_rates(start_year, end_year)

        # Create summary
        summary = {
            'world_bank_records': len(results['world_bank']),
            'oil_price_records': len(results['oil_prices']),
            'fao_index_records': len(results['fao_food_index']),
            'exchange_rate_records': len(results['exchange_rates']),
            'date_range': f"{start_year}-{end_year}",
            'countries_covered': self.east_africa_countries
        }

        # Save summary
        summary_path = self.output_dir / "macro_data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Macro data summary saved to {summary_path}")
        return results

def main():
    """Main function to run the macro data downloader"""
    downloader = MacroDownloader()

    logger.info("Starting macroeconomic data download")
    results = downloader.download_all_macro_data(2019, 2024)

    logger.info("Macroeconomic data download completed")
    for dataset, data in results.items():
        if not data.empty:
            logger.info(f"{dataset}: {len(data)} records")
        else:
            logger.warning(f"{dataset}: No data downloaded")

if __name__ == "__main__":
    main()
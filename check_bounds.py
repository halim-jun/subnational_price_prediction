
import pandas as pd

df = pd.read_csv('data/processed/spi/06_spi_csv/east_africa_spi_gamma_3_month.csv')
print("Lat range:", df['lat'].min(), df['lat'].max())
print("Lon range:", df['lon'].min(), df['lon'].max())

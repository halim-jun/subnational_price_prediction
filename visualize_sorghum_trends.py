import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set_theme(style="whitegrid")

# Load data
csv_path = "data/worldbank_imputed_price_data/WLD_RTFP_mkt_2026-02-03.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)

price = pd.read_csv(csv_path)

# Filter countries
target_countries = ['Kenya', 'Ethiopia', 'Somalia']
target_countries_df = price[price['country'].isin(target_countries)].copy()

# Select columns
# Note: 'ISO3', 'adm1_name', 'adm2_name', 'mkt_name', 'lat', 'lon' might vary in exact naming
# But user just wants to plot c_sorghum over time
final_price = target_countries_df[['year', 'month', 'country', 'c_sorghum']].copy()

# Create datetime column
final_price['date'] = pd.to_datetime(final_price[['year', 'month']].assign(day=1))

# Remove missing values for plotting
plot_data = final_price.dropna(subset=['c_sorghum'])

print(f"Plotting {len(plot_data)} data points...")

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=plot_data, x='date', y='c_sorghum', hue='country', marker='o')

plt.title('Sorghum Price Trends (c_sorghum) by Country')
plt.xlabel('Date')
plt.ylabel('Price (c_sorghum)')
plt.legend(title='Country')
plt.tight_layout()

# Save
output_path = "c_sorghum_trends.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

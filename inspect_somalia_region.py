import pandas as pd

file_path = 'data/population/somalia_population.xlsx'
df = pd.read_excel(file_path)

print("Unique values in Region column (Index 2):")
print(df.iloc[:, 2].unique())

print("\nSample rows:")
print(df.iloc[:10, [2, 3, 4]])

import pandas as pd

file_path = 'data/population/eth_admpop_2023.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("Head:\n", df.head(3))
except Exception as e:
    print(e)

import pandas as pd

def validate_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"\nValidating {file_path}")
        print("Available columns:", list(df.columns))
        print("Number of rows:", len(df))
        print("Missing values:\n", df.isnull().sum())
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

# Check all your data files
for year in ['2021', '2022', '2023']:
    validate_csv_file(f"/Users/Viku/GitHub/CO2-Emissions-Federated-Model/AI-Federated-Learning/Dataset/{year}.csv")

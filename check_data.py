import pandas as pd
from pathlib import Path

dataset_folder = Path("Data")
csv_files = list(dataset_folder.glob("*.csv"))

if not csv_files:
    print("No CSV file found inside the Data folder.")

else:
    file_path = csv_files[0]

    print("Dataset file:", file_path)

    df = pd.read_csv(file_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())
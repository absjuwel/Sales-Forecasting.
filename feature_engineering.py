import pandas as pd
from pathlib import Path

# Input and output file paths
input_file = Path("Data/daily_sales.csv")
output_file = Path("Data/model_data.csv")

# Load daily sales data
df = pd.read_csv(input_file)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data by date
df = df.sort_values("date").reset_index(drop=True)

# Create date-based features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Create previous sales features
df["lag_1"] = df["sales"].shift(1)
df["lag_7"] = df["sales"].shift(7)

# Remove rows with missing lag values
df = df.dropna().reset_index(drop=True)

# Save processed data
df.to_csv(output_file, index=False)

# Display results
print("Feature engineering completed successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nSaved file:", output_file)
import pandas as pd
from pathlib import Path

# File paths
input_file = Path("Data/sales.csv")
output_folder = Path("Data")
output_file = output_folder / "daily_sales.csv"

# Load dataset
df = pd.read_csv(input_file)

# Convert Date column into datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%m/%d/%Y",
    errors="coerce"
)

# Remove rows with invalid dates
df = df.dropna(subset=["Date"])

# Sort transactions by date
df = df.sort_values("Date")

# Calculate daily total sales
daily_sales = (
    df.groupby("Date", as_index=False)["Total"]
    .sum()
)

# Rename columns
daily_sales = daily_sales.rename(
    columns={
        "Date": "date",
        "Total": "sales"
    }
)

# Save the daily sales dataset
daily_sales.to_csv(output_file, index=False)

print("Data preprocessing completed successfully!")
print("\nDaily sales dataset:")
print(daily_sales.head())

print("\nDataset shape:", daily_sales.shape)
print("\nDate range:")
print(daily_sales["date"].min(), "to", daily_sales["date"].max())

print("\nSaved file:", output_file)
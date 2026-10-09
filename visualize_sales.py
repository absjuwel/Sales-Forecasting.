import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load daily sales data
df = pd.read_csv("Data/daily_sales.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Plot daily sales
plt.figure(figsize=(12, 6))

plt.plot(
    df["date"],
    df["sales"],
    marker="o",
    markersize=3
)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save graph
output_path = Path("Data/sales_trend.png")
plt.savefig(output_path, dpi=150)

plt.show()

print("Sales trend graph saved to:", output_path)
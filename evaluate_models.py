import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "Data" / "model_data.csv"
models_dir = BASE_DIR / "models"

df = pd.read_csv(data_path)

features = ["day_of_week", "month", "lag_1", "lag_7"]
target = "sales"

X = df[features]
y = df[target]

# Keep the same chronological 80/20 split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model_files = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regression": "decision_tree.pkl",
    "Random Forest Regression": "random_forest.pkl",
}

results = []
predictions = {}

for name, filename in model_files.items():
    bundle = joblib.load(models_dir / filename)
    model = bundle["model"]

    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2 Score": r2_score(y_test, y_pred),
    })

    predictions[name] = y_pred

# Save evaluation results
results_df = pd.DataFrame(results)
results_df.to_csv(BASE_DIR / "model_results.csv", index=False)

print("\nMODEL EVALUATION RESULTS")
print(results_df.round(3).to_string(index=False))

# Plot actual versus predicted sales
plt.figure(figsize=(12, 6))

plt.plot(
    range(len(y_test)),
    y_test.to_numpy(),
    marker="o",
    label="Actual Sales"
)

for name, y_pred in predictions.items():
    plt.plot(
        range(len(y_test)),
        y_pred,
        marker=".",
        label=name
    )

plt.title("Actual Sales vs Predicted Sales")
plt.xlabel("Test Observation")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()

graph_path = BASE_DIR / "Data" / "actual_vs_predicted.png"
plt.savefig(graph_path, dpi=150)
plt.show()

print(f"\nGraph saved to: {graph_path}")
print("Evaluation completed successfully!")

import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Data" / "model_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

# Features and target
features = ["day_of_week", "month", "lag_1", "lag_7"]

X = df[features]
y = df["sales"]


# Split data chronologically: 80% training, 20% testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Define three regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}


# Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        {
            "model": model,
            "features": features
        },
        MODEL_DIR / filename
    )


# Compare models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\nMODEL EVALUATION RESULTS")
print("=" * 65)
print(results_df.to_string(index=False))


# Select the model with the lowest MAE
best_model_name = results_df.iloc[0]["Model"]

best_filename = (
    best_model_name.lower().replace(" ", "_") + ".pkl"
)

best_model_path = MODEL_DIR / best_filename

joblib.dump(
    joblib.load(best_model_path),
    MODEL_DIR / "best_model.pkl"
)


# Save evaluation results
results_df.to_csv(
    BASE_DIR / "model_results.csv",
    index=False
)

print("\nBest Model:", best_model_name)
print("Models saved in:", MODEL_DIR)
print("Evaluation results saved to: model_results.csv")
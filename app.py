from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "linear_regression": "linear_regression.pkl",
    "decision_tree": "decision_tree.pkl",
    "random_forest": "random_forest.pkl",
}

MODEL_NAMES = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree Regression",
    "random_forest": "Random Forest Regression",
}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    selected_model = ""

    if request.method == "POST":
        try:
            model_name = request.form["model_name"]
            selected_model = MODEL_NAMES[model_name]

            model_path = MODEL_DIR / MODEL_FILES[model_name]
            bundle = joblib.load(model_path)

            model = bundle["model"]
            features = bundle["features"]

            input_data = {
                "day_of_week": int(request.form["day_of_week"]),
                "month": int(request.form["month"]),
                "lag_1": float(request.form["lag_1"]),
                "lag_7": float(request.form["lag_7"]),
            }

            input_df = pd.DataFrame([input_data])
            input_df = input_df[features]

            prediction = float(model.predict(input_df)[0])
            prediction = max(0, prediction)

        except (ValueError, KeyError, FileNotFoundError) as e:
            error = (
                "Prediction failed. Please check the inputs and ensure "
                "all three model files exist in the models folder."
            )
            print("Prediction error:", e)

        except Exception as e:
            error = "An unexpected error occurred. Check the VS Code terminal."
            print("Unexpected error:", e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    app.run(debug=True)
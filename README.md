# Sales Forecasting Using Machine Learning

## 1. Project Overview

This project predicts daily sales using Machine Learning regression models.

The system processes sales transaction data, creates features from historical sales, trains three regression models, evaluates their performance, and provides a web interface for sales prediction.

## 2. Project Objectives

- Analyze historical sales data.
- Prepare and preprocess the dataset.
- Apply feature engineering techniques.
- Train three Machine Learning regression models.
- Compare model performance.
- Predict daily sales through a web interface.

## 3. Dataset

The project uses supermarket sales transaction data.

The original dataset includes information such as:
- Date
- Product line
- Quantity
- Unit price
- Total sales
- Payment method
- Customer rating

The transaction data is aggregated by date to create a daily sales dataset.

## 4. Data Preprocessing

The preprocessing steps include:

- Loading the original CSV dataset.
- Converting dates into datetime format.
- Grouping transactions by date.
- Calculating total sales for each date.
- Saving the processed dataset.

## 5. Feature Engineering

The following features are used:

- `day_of_week`: Day of the week.
- `month`: Month of the year.
- `lag_1`: Sales from the previous observation.
- `lag_7`: Sales from seven observations earlier.

The target variable is `sales`.

## 6. Machine Learning Models

Three regression models are trained:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 7. Web Application

The project uses Flask to provide a web interface.

Users can enter:
- Day of the week
- Month
- Previous day's sales
- Sales from seven observations earlier
- Preferred Machine Learning model

The application returns a predicted sales value.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Flask
- Joblib
- HTML
- CSS

## 9. Project Structure

```text
Sales_Forecasting/
├── Data/
├── models/
├── src/
│   └── train_model.py
├── templates/
│   └── index.html
├── app.py
├── check_data.py
├── prepare_data.py
├── feature_engineering.py
├── visualize_sales.py
├── evaluate_models.py
├── model_results.csv
├── requirements.txt
└── README.md
```

## 10. How to Run the Project

### Step 1: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Train the models if needed

```bash
python src/train_model.py
```

### Step 3: Start the web application

```bash
python app.py
```

### Step 4: Open the application

Open this address in a web browser:

 https://sales-forecasting-qtqb.onrender.com

## 11. Limitations

The dataset contains a limited number of daily observations. Therefore, model performance may vary, and predictions should not be treated as guaranteed future sales.

## 12. Conclusion

This project demonstrates how Machine Learning regression models can be applied to sales forecasting. It combines data preprocessing, feature engineering, model evaluation, and a Flask-based web interface in one application.
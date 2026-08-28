# Next-Day Temperature Prediction Using Machine Learning and Feature Engineering

## Project Objective

This project implements a machine learning model to predict the next-day average temperature in Tehran using daily meteorological observations and engineered features.

## Data

- Location: Tehran, Iran (Latitude: 35.6892, Longitude: 51.3134)  
- Time period: 2020-01-01 to 2025-12-31  
- Number of records: 2192 days  
- Source: Open-Meteo Historical Weather API  
- Target variable: Next-day average temperature (`tavg` tomorrow)

### Data Acquisition

Raw data are downloaded from the Open-Meteo Archive API using the `src/download_data.py` script. This script fetches data directly from the Open-Meteo server and stores them in the project's `data` directory.

To download the data, run from the project root:

```bash
python src/download_data.py
```

After execution, the file `data/weather.csv` is created in the project directory.

Key request parameters:

- `latitude=35.6892`, `longitude=51.3134`: Tehran coordinates  
- `start_date=2020-01-01`, `end_date=2025-12-31`: Data time range  
- `daily`: Daily variables including mean, min and max temperature, precipitation, snowfall, wind direction and speed, pressure, humidity, and sunshine duration  
- `timezone=Asia/Tehran`: Time zone  
- `format=csv`: Output format

## Feature Engineering

A total of 46 new features are constructed from the raw data, including:

- Lagged temperature features (1, 2, 3, 7, and 14 days)  
- Rolling mean and standard deviation of temperature over 3-, 7-, 14-, and 30-day windows  
- Short-term changes in temperature, pressure, and wind speed  
- Temperature range, midpoint, and deviation from 30-day rolling mean  
- Seasonal and temporal features (month, day-of-year, sine/cosine of day-of-year, winter/summer/transition periods)  
- Interaction terms between temperature and pressure, wind, and precipitation  
- Variability, short-term trend, and flags for anomalous temperature

After feature engineering, the dataset contains 46 features plus the target variable.

## Models and Results

The following models are compared:

- Baseline: Tomorrow's temperature = today's temperature  
- Linear Regression  
- Ridge Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

The best model based on 2024 validation is Linear Regression.

### Final Test Results (Year 2025)

| Metric | Baseline | Linear Regression |
|---|---:|---:|
| MAE | 1.196 | 0.975 |
| RMSE | 1.579 | 1.248 |

MAE improvement over Baseline: approximately 18.5%.

## Project Structure

```text
project/
├── data/
│   └── weather.csv
├── models/
│   ├── best_temperature_model.joblib
│   └── model_features.joblib
├── outputs/
│   ├── model_results_2025.csv
│   ├── feature_importance_2025.csv
│   ├── high_error_days_model_comparison.csv
│   ├── tehran_2025_prediction_vs_actual.png
│   ├── tehran_2024_validation_comparison.png
│   └── tehran_feature_importance.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_temperature_model.py
│   ├── predict_for_date.py
│   ├── analyze_errors.py
│   └── download_data.py
├── README.md
├── README.fa.md
└── requirements.txt
```

## Usage Instructions

### 1) Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
```

### 2) Download Data

To automatically download data from Open-Meteo, run from the project root:

```bash
python src/download_data.py
```

This creates the file `data/weather.csv` in the project directory.

### 3) Train Model and Generate Outputs

```bash
python src/train_temperature_model.py
```

**Outputs:**

- `models/best_temperature_model.joblib`: Final Linear Regression model  
- `models/model_features.joblib`: List of features used  
- `outputs/model_results_2025.csv`: Predictions and errors for 2025  
- `outputs/feature_importance_2025.csv`: Feature importance scores  
- `outputs/tehran_2025_prediction_vs_actual.png`: Predicted vs. actual temperature plot  
- `outputs/tehran_2024_validation_comparison.png`: Model comparison on 2024 validation set  
- `outputs/tehran_feature_importance.png`: Feature importance plot  

### 4) Interactive Exploration of 2025 Predictions

```bash
python src/predict_for_date.py
```

**Output:**  
The script interactively asks for a date and displays for that date:

- Actual and predicted temperature  
- Absolute error  
- Values of important input features (e.g., `tavg`, `tavg_lag_1`, `tavg_roll_30_mean`, etc.)

### 5) Analyze High-Error Days

```bash
python src/analyze_errors.py
```

**Output:**

- Three days with the largest absolute errors in 2025  
- For each of these days:
  - Target date and input date  
  - Predicted and actual temperature  
  - Absolute error and direction (overestimated / underestimated)  
  - Meteorological conditions on the three days before and the target day  
- Output file: `outputs/high_error_days_model_comparison.csv` containing:
  - Dates of high-error days  
  - Actual and predicted temperatures for all models  
  - Absolute error for each model  
  - Best model for each day (lowest absolute error)

## Error Analysis (Summary)

Large errors typically occur during rapid temperature drops or fast-changing weather conditions. The model tends to underestimate the magnitude of such abrupt changes.

## Limitations

- The model is trained only for Tehran using daily data.  
- Predictions are based on same-day observations and do not replace operational numerical weather prediction models.  
- Errors increase during periods of rapid weather changes.

## Data Source

Open-Meteo. Historical Weather API.  
Data are retrieved via the Open-Meteo Archive API using the `src/download_data.py` script.

- Website: https://open-meteo.com/  
- API Documentation: https://open-meteo.com/en/docs/historical-weather-api

[English version](README.md) | [نسخه فارسی](README.fa.md)
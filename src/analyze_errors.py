import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    OUTPUTS_DIR,
    RAW_FEATURE_COLS,
    CONTINUOUS_COLS,
    FINAL_TRAIN_START,
    FINAL_TRAIN_END,
    TEST_START,
    TEST_END,
)
from data_loader import load_and_clean_data
from feature_engineering import create_features

warnings.filterwarnings("ignore")


# ==========================================================
# 1) Paths
# ==========================================================
OUTPUTS_DIR.mkdir(exist_ok=True)

HIGH_ERROR_DAYS_PATH = OUTPUTS_DIR / "high_error_days_model_comparison.csv"
LARGE_ERRORS_ANALYSIS_PATH = OUTPUTS_DIR / "large_errors_analysis.txt"


# ==========================================================
# 2) Load and prepare data
# ==========================================================
print("Loading and cleaning data...")
df = load_and_clean_data(DATA_PATH)

print("Creating features...")
df = create_features(df)

# Drop NaN rows from lag/rolling
model_df = df.dropna().reset_index(drop=True)

# Feature columns (exclude date, target, and redundant time features)
feature_cols = [
    col
    for col in model_df.columns
    if col not in ["date", "target_next_day_tavg", "year", "day_of_year"]
]


# ==========================================================
# 3) Final train and 2025 test split
# ==========================================================
train_df = model_df[
    (model_df["date"] >= FINAL_TRAIN_START)
    & (model_df["date"] <= FINAL_TRAIN_END)
].copy()

test_df = model_df[
    (model_df["date"] >= TEST_START) & (model_df["date"] <= TEST_END)
].copy()

X_train = train_df[feature_cols]
y_train = train_df["target_next_day_tavg"]

X_test = test_df[feature_cols]
y_test = test_df["target_next_day_tavg"]

print(f"\nTrain samples: {len(train_df)}")
print(f"Test samples:  {len(test_df)}")


# ==========================================================
# 4) Train all models on the same training data
# ==========================================================
print("\nTraining models...")

models = {
    "Baseline": None,
    "Linear Regression": Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    ),
    "Ridge": Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
    ),
}

try:
    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
except ImportError:
    print("XGBoost is not installed; it will be excluded.")


# ==========================================================
# 5) Build prediction table for all models
# ==========================================================
print("\nGenerating predictions for all models...")

comparison_df = pd.DataFrame(
    {
        "input_date": test_df["date"].values,
        "target_date": (test_df["date"] + pd.Timedelta(days=1)).values,
        "actual_tavg": y_test.values,
    }
)

# Baseline: tomorrow = today
comparison_df["Baseline_pred"] = test_df["tavg"].values
comparison_df["Baseline_abs_error"] = (
    comparison_df["Baseline_pred"] - comparison_df["actual_tavg"]
).abs()

for name, model in models.items():
    if name == "Baseline":
        continue

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    safe_name = name.replace(" ", "_")

    comparison_df[f"{safe_name}_pred"] = pred
    comparison_df[f"{safe_name}_abs_error"] = (
        pred - comparison_df["actual_tavg"]
    ).abs()


# ==========================================================
# 6) Days where Ridge error is above its average
# ==========================================================
ridge_error_col = "Ridge_abs_error"
ridge_mae = comparison_df[ridge_error_col].mean()

high_error_days = comparison_df[
    comparison_df[ridge_error_col] > ridge_mae
].copy()

error_cols = [
    col for col in comparison_df.columns if col.endswith("_abs_error")
]

# Best model for each day (lowest absolute error)
high_error_days["Best_model_for_day"] = (
    high_error_days[error_cols]
    .idxmin(axis=1)
    .str.replace("_abs_error", "", regex=False)
)

high_error_days["Best_model_error"] = high_error_days[error_cols].min(
    axis=1
)

high_error_days = high_error_days.sort_values(
    ridge_error_col, ascending=False
)

high_error_days.to_csv(HIGH_ERROR_DAYS_PATH, index=False)


# ==========================================================
# 7) Summary: model comparison on high-error days
# ==========================================================
print("\n" + "=" * 70)
print("=== Model Comparison on High-Error Ridge Days ===")
print(f"Ridge MAE over all 2025 test days: {ridge_mae:.3f} °C")
print(
    f"Days where Ridge error > Ridge MAE: "
    f"{len(high_error_days)} out of {len(comparison_df)}"
)

print("\nNumber of times each model was best on these days:")
print(high_error_days["Best_model_for_day"].value_counts().to_string())

print("\n=== Top 10 Largest Ridge Errors ===")

display_cols = [
    "target_date",
    "actual_tavg",
    "Ridge_pred",
    "Ridge_abs_error",
    "Best_model_for_day",
    "Best_model_error",
]

print(
    high_error_days[display_cols]
    .head(10)
    .to_string(
        index=False,
        formatters={
            "actual_tavg": "{:.2f}".format,
            "Ridge_pred": "{:.2f}".format,
            "Ridge_abs_error": "{:.2f}".format,
            "Best_model_error": "{:.2f}".format,
        },
    )
)

print(f"\nFull comparison saved to:\n{HIGH_ERROR_DAYS_PATH}")


# ==========================================================
# 8) Detailed review of three largest errors (Linear Regression)
# ==========================================================
print("\n" + "=" * 70)
print("=== Detailed Review of Three Largest Errors ===")

RESULTS_PATH = OUTPUTS_DIR / "model_results_2025.csv"

if not RESULTS_PATH.exists():
    print(f"\nPrediction results not found: {RESULTS_PATH}")
    print("Run src/train_temperature_model.py first.")
else:
    results = pd.read_csv(RESULTS_PATH)
    results["input_date"] = pd.to_datetime(results["input_date"])
    results["target_date"] = pd.to_datetime(results["target_date"])

    # Reload weather data for context
    weather = load_and_clean_data(DATA_PATH)

    largest_errors = results.nlargest(3, "absolute_error")

    for _, row in largest_errors.iterrows():
        target_date = row["target_date"]
        input_date = row["input_date"]

        start_date = input_date - pd.Timedelta(days=3)
        end_date = target_date

        window = weather[
            (weather["date"] >= start_date) & (weather["date"] <= end_date)
        ][["date", "tavg", "tmin", "tmax", "prcp", "wspd", "pres", "rhum"]]

        direction = (
            "overestimated"
            if row["predicted_tavg"] > row["actual_tavg"]
            else "underestimated"
        )

        print("\n" + "-" * 70)
        print(f"Target date: {target_date.date()}")
        print(f"Input date:  {input_date.date()}")
        print(f"Predicted:   {row['predicted_tavg']:.2f} °C")
        print(f"Actual:      {row['actual_tavg']:.2f} °C")
        print(f"Abs. error:  {row['absolute_error']:.2f} °C")
        print(f"Model {direction} the temperature.")

        print("\nWeather around this prediction:")
        print(window.to_string(index=False))

    print("\n" + "=" * 70)
    print("Analysis complete!")

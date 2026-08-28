import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import warnings


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


from config import (
    DATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    MODEL_PATH,
    FEATURES_PATH,
    RESULTS_PATH,
    IMPORTANCE_CSV_PATH,
    PREDICTION_PLOT,
    COMPARISON_PLOT,
    IMPORTANCE_PLOT,
    RAW_FEATURE_COLS,
    CONTINUOUS_COLS,
    TRAIN_START,
    TRAIN_END,
    VALIDATION_START,
    VALIDATION_END,
    FINAL_TRAIN_START,
    FINAL_TRAIN_END,
    TEST_START,
    TEST_END
)
from data_loader import load_and_clean_data
from feature_engineering import create_features


warnings.filterwarnings("ignore")


# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 1) Load data
# ==========================================================
print("Loading and cleaning data...")
df = load_and_clean_data(DATA_PATH)


# ==========================================================
# 2) Feature engineering
# ==========================================================
print("Creating features...")
df = create_features(df)


# Drop NaN rows from lag/rolling
model_df = df.dropna().reset_index(drop=True)


print(f"\nRows after feature engineering: {len(model_df)}")
print(f"Total features created: {len(model_df.columns) - 2}")  # exclude date and target


# ==========================================================
# 3) Time-based split
# ==========================================================
train_df = model_df[
    (model_df["date"] >= TRAIN_START) &
    (model_df["date"] <= TRAIN_END)
].copy()


validation_df = model_df[
    (model_df["date"] >= VALIDATION_START) &
    (model_df["date"] <= VALIDATION_END)
].copy()


final_train_df = model_df[
    (model_df["date"] >= FINAL_TRAIN_START) &
    (model_df["date"] <= FINAL_TRAIN_END)
].copy()


test_df = model_df[
    (model_df["date"] >= TEST_START) &
    (model_df["date"] <= TEST_END)
].copy()


# Feature columns (exclude date, target, and redundant time features)
feature_cols = [
    col for col in model_df.columns
    if col not in [
        "date",
        "target_next_day_tavg",
        "year",
        "day_of_year"
    ]
]


X_train = train_df[feature_cols]
y_train = train_df["target_next_day_tavg"]


X_val = validation_df[feature_cols]
y_val = validation_df["target_next_day_tavg"]


X_final_train = final_train_df[feature_cols]
y_final_train = final_train_df["target_next_day_tavg"]


X_test = test_df[feature_cols]
y_test = test_df["target_next_day_tavg"]


print("\n=== Time Split ===")
print(
    f"Train:       {len(train_df)} rows | "
    f"{train_df['date'].min().date()} to {train_df['date'].max().date()}"
)
print(
    f"Validation:  {len(validation_df)} rows | "
    f"{validation_df['date'].min().date()} to "
    f"{validation_df['date'].max().date()}"
)
print(
    f"Final train: {len(final_train_df)} rows | "
    f"{final_train_df['date'].min().date()} to "
    f"{final_train_df['date'].max().date()}"
)
print(
    f"Test inputs: {len(test_df)} rows | "
    f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
)
print("Test targets: 2025-01-01 to 2025-12-31")


# ==========================================================
# 4) Baseline
# ==========================================================
baseline_val_pred = validation_df["tavg"].values


baseline_val_mae = mean_absolute_error(y_val, baseline_val_pred)
baseline_val_rmse = np.sqrt(mean_squared_error(y_val, baseline_val_pred))


print("\n=== Validation Baseline ===")
print(f"MAE:  {baseline_val_mae:.3f} °C")
print(f"RMSE: {baseline_val_rmse:.3f} °C")


# ==========================================================
# 5) Train candidate models on 2020-2023
# ==========================================================
print("\nTraining models...")


models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        random_state=42
    )
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
        n_jobs=-1
    )
except ImportError:
    print("\nXGBoost is not installed. Run: pip install xgboost")


validation_results = [{
    "Model": "Baseline (today's tavg)",
    "MAE (°C)": baseline_val_mae,
    "RMSE (°C)": baseline_val_rmse
}]


for name, model in models.items():
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)


    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))


    validation_results.append({
        "Model": name,
        "MAE (°C)": mae,
        "RMSE (°C)": rmse
    })


validation_results_df = pd.DataFrame(validation_results)
validation_results_df = validation_results_df.sort_values(
    "MAE (°C)"
).reset_index(drop=True)


print("\n=== Validation Results: Year 2024 ===")
print(validation_results_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.3f}"
))


# ==========================================================
# 6) Select model and train final version
# ==========================================================
ml_validation_results = validation_results_df[
    validation_results_df["Model"] != "Baseline (today's tavg)"
].reset_index(drop=True)


best_model_name = ml_validation_results.iloc[0]["Model"]
best_model = models[best_model_name]


print(f"\nSelected best model from validation: {best_model_name}")


best_model.fit(X_final_train, y_final_train)


joblib.dump(best_model, MODEL_PATH)
joblib.dump(feature_cols, FEATURES_PATH)


# ==========================================================
# 7) Final test on 2025
# ==========================================================
baseline_test_pred = test_df["tavg"].values


baseline_test_mae = mean_absolute_error(y_test, baseline_test_pred)
baseline_test_rmse = np.sqrt(mean_squared_error(y_test, baseline_test_pred))


test_pred = best_model.predict(X_test)


test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))


test_improvement = (
    (baseline_test_mae - test_mae) / baseline_test_mae
) * 100


print("\n=== Final Test Results: Year 2025 ===")
print(f"Baseline MAE:  {baseline_test_mae:.3f} °C")
print(f"Baseline RMSE: {baseline_test_rmse:.3f} °C")
print(f"{best_model_name} MAE:  {test_mae:.3f} °C")
print(f"{best_model_name} RMSE: {test_rmse:.3f} °C")
print(f"MAE improvement vs baseline: {test_improvement:.2f}%")


# ==========================================================
# 8) Save 2025 predictions
# ==========================================================
prediction_df = pd.DataFrame({
    "input_date": test_df["date"].values,
    "target_date": (
        test_df["date"] + pd.Timedelta(days=1)
    ).values,
    "predicted_tavg": test_pred,
    "actual_tavg": y_test.values
})


prediction_df["absolute_error"] = (
    prediction_df["predicted_tavg"] -
    prediction_df["actual_tavg"]
).abs()


prediction_df.to_csv(RESULTS_PATH, index=False)


print(f"\nPredictions saved: {RESULTS_PATH}")


# ==========================================================
# 9) Plot predictions
# ==========================================================
fig, ax = plt.subplots(figsize=(14, 5))


ax.plot(
    prediction_df["target_date"],
    prediction_df["actual_tavg"],
    label="Actual temperature",
    color="steelblue",
    linewidth=1.4
)


ax.plot(
    prediction_df["target_date"],
    prediction_df["predicted_tavg"],
    label=f"Predicted ({best_model_name})",
    color="darkorange",
    linewidth=1.4
)


ax.set_title(
    "Next-Day Mean Temperature Prediction — "
    "Tehran Mehrabad Airport, 2025"
)
ax.set_xlabel("Target date")
ax.set_ylabel("Average temperature (°C)")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.35)


fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(PREDICTION_PLOT, dpi=160, bbox_inches="tight")
plt.close(fig)


# ==========================================================
# 10) Plot validation comparison
# ==========================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))


ax[0].barh(
    validation_results_df["Model"],
    validation_results_df["MAE (°C)"],
    color="steelblue"
)
ax[0].set_title("Validation MAE — Year 2024")
ax[0].set_xlabel("MAE (°C)")
ax[0].grid(axis="x", linestyle="--", alpha=0.3)


ax[1].barh(
    validation_results_df["Model"],
    validation_results_df["RMSE (°C)"],
    color="coral"
)
ax[1].set_title("Validation RMSE — Year 2024")
ax[1].set_xlabel("RMSE (°C)")
ax[1].grid(axis="x", linestyle="--", alpha=0.3)


fig.tight_layout()
fig.savefig(COMPARISON_PLOT, dpi=160, bbox_inches="tight")
plt.close(fig)


# ==========================================================
# 11) Permutation feature importance
# ==========================================================
permutation_result = permutation_importance(
    estimator=best_model,
    X=X_test,
    y=y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)


importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance Mean": permutation_result.importances_mean,
    "Importance Std": permutation_result.importances_std
})


importance_df = importance_df.sort_values(
    "Importance Mean",
    ascending=False
).head(15)


importance_df.to_csv(IMPORTANCE_CSV_PATH, index=False)


print(f"\nTop features according to {best_model_name}:")
print(importance_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))


fig, ax = plt.subplots(figsize=(9, 7))


ax.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance Mean"][::-1],
    xerr=importance_df["Importance Std"][::-1],
    color="seagreen",
    alpha=0.85
)


ax.set_title(
    f"Permutation Feature Importance — {best_model_name}"
)
ax.set_xlabel("Decrease in negative MAE after shuffling")
ax.grid(axis="x", linestyle="--", alpha=0.3)


fig.tight_layout()
fig.savefig(IMPORTANCE_PLOT, dpi=160, bbox_inches="tight")
plt.close(fig)


# ==========================================================
# 12) Show output files
# ==========================================================
print("\n=== Created Files ===")
print(f"Model:              {MODEL_PATH}")
print(f"Feature list:       {FEATURES_PATH}")
print(f"2025 predictions:   {RESULTS_PATH}")
print(f"Feature importance: {IMPORTANCE_CSV_PATH}")
print(f"Prediction plot:    {PREDICTION_PLOT}")
print(f"Validation plot:    {COMPARISON_PLOT}")
print(f"Importance plot:    {IMPORTANCE_PLOT}")

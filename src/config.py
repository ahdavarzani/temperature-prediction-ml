from pathlib import Path


# ==========================================================
# Project paths
# ==========================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


DATA_PATH = DATA_DIR / "weather.csv"


MODEL_PATH = MODELS_DIR / "best_temperature_model.joblib"
FEATURES_PATH = MODELS_DIR / "model_features.joblib"


RESULTS_PATH = OUTPUTS_DIR / "model_results_2025.csv"
IMPORTANCE_CSV_PATH = OUTPUTS_DIR / "feature_importance_2025.csv"
HIGH_ERROR_DAYS_PATH = OUTPUTS_DIR / "high_error_days_model_comparison.csv"


PREDICTION_PLOT = OUTPUTS_DIR / "tehran_2025_prediction_vs_actual.png"
COMPARISON_PLOT = OUTPUTS_DIR / "tehran_2024_validation_comparison.png"
IMPORTANCE_PLOT = OUTPUTS_DIR / "tehran_feature_importance.png"


# ==========================================================
# Feature columns
# ==========================================================
RAW_FEATURE_COLS = [
    "tavg",
    "tmin",
    "tmax",
    "prcp",
    "wspd",
    "pres",
    "rhum"
]


CONTINUOUS_COLS = ["tavg", "tmin", "tmax", "wspd", "pres", "rhum"]


# ==========================================================
# Time splits
# ==========================================================
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"


VALIDATION_START = "2024-01-01"
VALIDATION_END = "2024-12-30"


FINAL_TRAIN_START = "2020-01-01"
FINAL_TRAIN_END = "2024-12-30"


TEST_START = "2024-12-31"
TEST_END = "2025-12-30"

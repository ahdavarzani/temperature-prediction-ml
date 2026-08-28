import pandas as pd
from pathlib import Path
import joblib


from config import MODELS_DIR, OUTPUTS_DIR, MODEL_PATH, FEATURES_PATH, RESULTS_PATH


# ==========================================================
# 1) Check files and load prediction results
# ==========================================================
for file_path in [MODEL_PATH, FEATURES_PATH, RESULTS_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n\n"
            "Run src/train_temperature_model.py first."
        )


model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(FEATURES_PATH)


predictions = pd.read_csv(RESULTS_PATH)


predictions["input_date"] = pd.to_datetime(predictions["input_date"])
predictions["target_date"] = pd.to_datetime(predictions["target_date"])


# ==========================================================
# 2) Interactive demo
# ==========================================================
print("Tehran Mehrabad Airport — Next-Day Temperature Prediction")
print("Available target dates: 2025-01-01 to 2025-12-31")
print("Enter 'exit' to close the program.")


while True:
    user_input = input("\nEnter target date (YYYY-MM-DD): ").strip()


    if user_input.lower() == "exit":
        print("Program closed.")
        break


    try:
        target_date = pd.to_datetime(user_input)


        if target_date.year != 2025:
            print("Please enter a date in year 2025.")
            continue


        selected = predictions[
            predictions["target_date"] == target_date
        ]


        if selected.empty:
            print(
                "Date not found. Enter a date from "
                "2025-01-01 to 2025-12-31."
            )
            continue


        row = selected.iloc[0]


        predicted = row["predicted_tavg"]
        actual = row["actual_tavg"]
        absolute_error = row["absolute_error"]
        input_date = row["input_date"]


        print("\n=== Prediction Result ===")
        print(f"Target date:                 {target_date.date()}")
        print(f"Input data available until:  {input_date.date()}")
        print(f"Predicted mean temperature:  {predicted:.2f} °C")
        print(f"Actual recorded temperature: {actual:.2f} °C")
        print(f"Absolute error:              {absolute_error:.2f} °C")


    except (ValueError, TypeError):
        print(
            "Invalid date format. Use YYYY-MM-DD, "
            "for example: 2025-07-15"
        )

import pandas as pd
import numpy as np


def create_features(df):
    """
    Create all engineered features for temperature prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with columns: date, tavg, tmin, tmax, prcp, wspd, pres, rhum
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Target variable
    df["target_next_day_tavg"] = df["tavg"].shift(-1)
    
    # --- Lag features ---
    for lag in [1, 2, 3, 7, 14]:
        df[f"tavg_lag_{lag}"] = df["tavg"].shift(lag)
    
    # --- Rolling statistics ---
    for window in [3, 7, 14, 30]:
        df[f"tavg_roll_{window}_mean"] = (
            df["tavg"].shift(1).rolling(window).mean()
        )
        df[f"tavg_roll_{window}_std"] = (
            df["tavg"].shift(1).rolling(window).std()
        )
    
    # --- Change features ---
    df["tavg_change_1d"] = df["tavg"].diff(1)
    df["tavg_change_3d"] = df["tavg"].diff(3)
    df["pressure_change_1d"] = df["pres"].diff(1)
    df["wind_change_1d"] = df["wspd"].diff(1)
    
    # --- Derived features ---
    df["temp_range"] = df["tmax"] - df["tmin"]
    df["temp_midpoint"] = (df["tmax"] + df["tmin"]) / 2
    df["has_rain"] = (df["prcp"] > 0).astype(int)
    
    # --- Time features ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)
    
    # --- Interaction features ---
    df["temp_pressure_interaction"] = df["tavg"] * df["pres"]
    df["temp_wind_interaction"] = df["tavg"] * df["wspd"]
    df["temp_range_rain_interaction"] = df["temp_range"] * df["has_rain"]
    
    # --- Seasonal binary features ---
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_season_transition"] = df["month"].isin([3, 4, 9, 10]).astype(int)
    
    # --- Volatility features ---
    df["temp_volatility_7d"] = df["tavg"].shift(1).rolling(7).std()
    df["pressure_volatility_7d"] = df["pres"].shift(1).rolling(7).std()
    
    # --- Trend features (linear slope) ---
    df["temp_trend_3d"] = df["tavg"].shift(1).rolling(3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    df["temp_trend_7d"] = df["tavg"].shift(1).rolling(7).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # --- Deviation from long-term mean ---
    df["temp_deviation_from_30d_mean"] = (
        df["tavg"] - df["tavg"].shift(1).rolling(30).mean()
    )
    
    # --- Consecutive rain days ---
    df["consecutive_rain_days"] = df["has_rain"].rolling(7).sum()
    
    # --- Relative pressure (monthly anomaly) ---
    monthly_pressure_mean = df.groupby("month")["pres"].transform("mean")
    df["pressure_relative"] = df["pres"] - monthly_pressure_mean
    
    # --- Extreme temperature flag ---
    temp_30d_mean = df["tavg"].shift(1).rolling(30).mean()
    temp_30d_std = df["tavg"].shift(1).rolling(30).std()
    df["extreme_temp_flag"] = (
        (df["tavg"] - temp_30d_mean).abs() > 2 * temp_30d_std
    ).astype(int)
    
    return df

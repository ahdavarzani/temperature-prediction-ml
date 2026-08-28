import pandas as pd
from pathlib import Path


def load_and_clean_data(data_path):
    """
    Load weather data from CSV, rename columns, and clean missing values.
    
    Parameters
    ----------
    data_path : str or Path
        Path to weather.csv file
    
    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with columns: date, tavg, tmin, tmax, prcp, wspd, pres, rhum
    """
    # Load data (skip first 2 rows for new dataset format)
    df = pd.read_csv(data_path, skiprows=2)
    
    # Rename columns to match old naming convention
    df = df.rename(columns={
        "time": "date",
        "temperature_2m_mean (°C)": "tavg",
        "temperature_2m_max (°C)": "tmax",
        "temperature_2m_min (°C)": "tmin",
        "precipitation_sum (mm)": "prcp",
        "wind_speed_10m_mean (km/h)": "wspd",
        "pressure_msl_mean (hPa)": "pres",
        "relative_humidity_2m_mean (%)": "rhum"
    })
    
    # Parse date and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Select only required columns
    raw_feature_cols = [
        "tavg",
        "tmin",
        "tmax",
        "prcp",
        "wspd",
        "pres",
        "rhum"
    ]
    
    required_cols = ["date"] + raw_feature_cols
    df = df[required_cols].copy()
    
    # Interpolate missing values for continuous columns
    continuous_cols = ["tavg", "tmin", "tmax", "wspd", "pres", "rhum"]
    df[continuous_cols] = df[continuous_cols].interpolate(
        method="linear",
        limit_direction="both"
    )
    
    # Fill remaining missing values with median
    for col in continuous_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill precipitation with 0
    df["prcp"] = df["prcp"].fillna(0)
    
    return df

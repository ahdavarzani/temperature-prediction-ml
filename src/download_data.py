"""
Download Tehran weather data from the Open-Meteo Archive API
and save it to ../data/weather.csv relative to this script's location.
"""

import urllib.request
import urllib.error
from pathlib import Path


# Absolute path to the src directory (where this file is located)
SRC_DIR = Path(__file__).resolve().parent


# Project root directory (one level above src)
PROJECT_ROOT = SRC_DIR.parent


DATA_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=35.6892&longitude=51.3134&"
    "start_date=2020-01-01&end_date=2025-12-31&"
    "daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,snowfall_sum,wind_direction_10m_dominant,"
    "wind_speed_10m_mean,wind_gusts_10m_max,pressure_msl_mean,"
    "relative_humidity_2m_mean,sunshine_duration&"
    "timezone=Asia/Tehran&format=csv"
)


DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "weather.csv"


print(f"Downloading data from Open-Meteo...")
print(f"URL: {DATA_URL}")
print(f"Save path: {DATA_PATH.resolve()}")


try:
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    size = DATA_PATH.stat().st_size
    print(f"Data downloaded and saved successfully.")
    print(f"File size: {size:,} bytes")
except urllib.error.URLError as e:
    print(f"Download error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

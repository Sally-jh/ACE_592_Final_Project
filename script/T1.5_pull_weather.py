"""
T1.5 Pull daily weather from Open-Meteo for each EV Watts metro.

Run with:
    python pull_weather.py

Output: C:/Users/chenq6/OneDrive - University of Illinois - Urbana/Courses/4th semester of PhD/ACE 535 (592)/ACE 592 in-class project/final_project/data/processed/weather_daily.parquet
        (one row per metro_area x date, weather daily aggregates)
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Metro --> list of (airport_code, lat, lon).
# Single-airport metros: one tuple. Microclimate metros: 2-3 tuples averaged.
# Verify these names against `df["metro_area"].unique()` from the EV Watts
# data; rename keys here if EV Watts uses different labels.
# ---------------------------------------------------------------------------
METROS = {
    "Albany-Schenectady-Troy, NY Metro Area": [("ALB", 42.748, -73.802)],
    "Ann Arbor, MI Metro Area": [("ARB", 42.223, -83.745)],
    "Austin-Round Rock-Georgetown, TX Metro Area": [("AUS", 30.197, -97.666)],
    "Baltimore-Columbia-Towson, MD Metro Area": [("BWI", 39.177, -76.668)],
    "Boston-Cambridge-Newton, MA-NH Metro Area": [("BOS", 42.366, -71.020)],
    "Boulder, CO Metro Area": [("BDU", 40.039, -105.226)],
    "Burlington-South Burlington, VT Metro Area": [("BTV", 44.472, -73.153)],
    "Chicago-Naperville-Elgin, IL-IN-WI Metro Area": [("ORD", 41.978, -87.904)],
    "Dallas-Fort Worth-Arlington, TX Metro Area": [("DFW", 32.897, -97.038)],
    "Denver-Aurora-Lakewood, CO Metro Area": [("DEN", 39.857, -104.673)],
    "Des Moines-West Des Moines, IA Metro Area": [("DSM", 41.534, -93.660)],
    "Detroit-Warren-Dearborn, MI Metro Area": [("DTW", 42.213, -83.353)],
    "Grand Rapids-Kentwood, MI Metro Area": [("GRR", 42.880, -85.522)],
    "Kansas City, MO-KS Metro Area": [("MCI", 39.297, -94.714)],
    "Las Vegas-Henderson-Paradise, NV Metro Area": [("LAS", 36.080, -115.152)],
    "Los Angeles-Long Beach-Anaheim, CA Metro Area": [
        ("LAX", 33.943, -118.408),
        ("BUR", 34.201, -118.359),
        ("LGB", 33.818, -118.152)
    ],
    "Miami-Fort Lauderdale-Pompano Beach, FL Metro Area": [("MIA", 25.793, -80.291)],
    "New York-Newark-Jersey City, NY-NJ-PA Metro Area": [("LGA", 40.777, -73.872)],
    "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metro Area": [("PHL", 39.872, -75.241)],
    "Phoenix-Mesa-Chandler, AZ Metro Area": [("PHX", 33.434, -112.012)],
    "Pittsburgh, PA Metro Area": [("PIT", 40.491, -80.233)],
    "Portland-Vancouver-Hillsboro, OR-WA Metro Area": [("PDX", 45.589, -122.595)],
    "Providence-Warwick, RI-MA Metro Area": [("PVD", 41.724, -71.428)],
    "Reno, NV Metro Area": [("RNO", 39.499, -119.768)],
    "Rochester, NY Metro Area": [("ROC", 43.118, -77.672)],
    "Salem, OR Metro Area": [("SLE", 44.909, -123.002)],
    "Seattle-Tacoma-Bellevue, WA Metro Area": [
        ("SEA", 47.450, -122.309),
        ("BFI", 47.530, -122.302)
    ],
    "Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area": [("DCA", 38.852, -77.038)],
    "Worcester, MA-CT Metro Area": [("ORH", 42.267, -71.875)]
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
]

START_DATE = "2019-06-25"
END_DATE   = "2022-12-31"

# Updated paths using a raw string to handle Windows backslashes properly
BASE_DATA_DIR = Path(r"C:\Users\chenq6\OneDrive - University of Illinois - Urbana\Courses\4th semester of PhD\ACE 535 (592)\ACE 592 in-class project\final_project\data")
CACHE_DIR  = BASE_DATA_DIR / "raw" / "weather"
OUTPUT_DIR = BASE_DATA_DIR / "processed"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_point(code: str, lat: float, lon: float) -> dict:
    """Fetch daily weather for one airport coordinate, with on-disk caching."""
    cache_file = CACHE_DIR / f"{code}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "daily": ",".join(DAILY_VARS),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
            "timezone": "auto",
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()

    cache_file.write_text(json.dumps(payload))
    time.sleep(20)  # be polite to the free API
    return payload


def main() -> None:
    metro_frames = []
    for metro, points in METROS.items():
        print(f"  fetching {metro}: {len(points)} point(s)")
        point_dfs = []
        for code, lat, lon in points:
            payload = fetch_point(code, lat, lon)
            df = pd.DataFrame(payload["daily"])
            df = df.rename(columns={"time": "date"})
            point_dfs.append(df)

        # Average across points within the metro (numeric daily vars only).
        combined = (
            pd.concat(point_dfs)
              .groupby("date", as_index=False)
              .mean(numeric_only=True)
        )
        combined["metro_area"] = metro
        metro_frames.append(combined)

    weather_daily = pd.concat(metro_frames, ignore_index=True)
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])

    # Reorder columns: keys first, weather vars second.
    cols = ["metro_area", "date"] + DAILY_VARS
    weather_daily = weather_daily[cols].sort_values(["metro_area", "date"])

    out_path = OUTPUT_DIR / "weather_daily.parquet"
    weather_daily.to_parquet(out_path, index=False)

    print(f"\nWrote {len(weather_daily):,} rows to {out_path}")
    print(f"Metros: {weather_daily['metro_area'].nunique()}")
    print(f"Date range: {weather_daily['date'].min().date()} to "
          f"{weather_daily['date'].max().date()}")
    print("\nFirst rows:")
    print(weather_daily.head())


if __name__ == "__main__":
    main()
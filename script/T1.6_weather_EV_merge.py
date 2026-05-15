"""
Merge EV metro-day panel with daily weather panel and create charger-level splits.

Outputs: 
1. data/processed/merged_metro_day.parquet
2. data/processed/metro_day_charger_panel.parquet
"""

import pandas as pd
from pathlib import Path

# Define file paths
BASE_DATA_DIR = Path(r"C:\Users\chenq6\OneDrive - University of Illinois - Urbana\Courses\4th semester of PhD\ACE 535 (592)\ACE 592 in-class project\final_project\data")
PROCESSED_DIR = BASE_DATA_DIR / "processed"

EV_PATH = PROCESSED_DIR / "EV_merged.parquet"
WEATHER_PATH = PROCESSED_DIR / "weather_daily.parquet"
OUTPUT_PATH = PROCESSED_DIR / "merged_metro_day.parquet"
CHARGER_PANEL_PATH = PROCESSED_DIR / "metro_day_charger_panel.parquet"

def main():
    # 1. Load the datasets
    print("Loading datasets...")
    df_ev = pd.read_parquet(EV_PATH)
    df_weather = pd.read_parquet(WEATHER_PATH)

    # Ensure date columns are strictly datetime objects for a safe merge
    df_ev['date'] = pd.to_datetime(df_ev['date'])
    df_weather['date'] = pd.to_datetime(df_weather['date'])

    # 2. Left-join the EV data with the daily weather panel
    print("Merging datasets...")
    df_merged = pd.merge(
        df_ev, 
        df_weather, 
        on=['metro_area', 'date'], 
        how='left', 
        indicator=True
    )

    # 3. Verify the join hit rate
    hit_rate = (df_merged['_merge'] == 'both').mean() * 100
    print(f"Join hit rate: {hit_rate:.2f}%\n")
    
    # Drop the indicator column as it is no longer needed
    df_merged = df_merged.drop(columns=['_merge'])

    # 4. Save as the main analytical file
    df_merged.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved merged analytical file to: {OUTPUT_PATH}\n")

    # 5. Generate the 5-line printout
    # Calculate metrics directly from the session-level dataset
    
    num_metros = df_merged['metro_area'].nunique()
    
    # Number of unique metro-days is the count of unique (metro_area, date) pairs
    num_metro_days = df_merged[['metro_area', 'date']].drop_duplicates().shape[0]
    
    # Total sessions is simply the total number of rows since each row is a session
    total_sessions = len(df_merged)
    mean_sessions_per_day = total_sessions / num_metro_days if num_metro_days > 0 else 0
    
    # Mean kWh per session is the average of the 'energy_kwh' column
    mean_kwh_per_session = df_merged['energy_kwh'].mean()

    print("=== Deliverable 5-Line Printout ===")
    print(f"1. Number of metros: {num_metros}")
    print(f"2. Number of metro-days: {num_metro_days:,}")
    print(f"3. Mean sessions/day: {mean_sessions_per_day:.2f}")
    print(f"4. Mean kWh/session: {mean_kwh_per_session:.2f}")
    print("===================================\n")

    # 6. Create the metro-day-charger panel
    print("Aggregating data by metro, date, and charge_level...")
    
    weather_cols = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'apparent_temperature_max', 'apparent_temperature_min', 'precipitation_sum',
        'rain_sum', 'snowfall_sum', 'wind_speed_10m_max', 'relative_humidity_2m_mean'
    ]
    
    agg_dict = {
        'session_id': 'count',
        'energy_kwh': ['sum', 'mean'],
        'charge_duration': 'mean'
    }
    
    for col in weather_cols:
        if col in df_merged.columns:
            agg_dict[col] = 'first'

    grouped = df_merged.groupby(['metro_area', 'date', 'charge_level']).agg(agg_dict).reset_index()

    new_cols = []
    for col in grouped.columns:
        if col[1] == 'count':
            new_cols.append('sessions')
        elif col[1] in ['sum', 'mean']:
            new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append(col[0])
            
    grouped.columns = new_cols

    grouped.to_parquet(CHARGER_PANEL_PATH, index=False)
    print(f"Saved metro-day charger panel to: {CHARGER_PANEL_PATH}")

if __name__ == "__main__":
    main()
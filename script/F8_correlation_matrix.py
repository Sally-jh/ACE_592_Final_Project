#!/usr/bin/env python3
"""F8: metro-demeaned weather-charging correlation matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jinyangli/Downloads/evwatts.public 2")
INPUT_PATH = Path("/Users/jinyangli/Downloads/metro_day_charger_panel.parquet")
OUTPUT_DIR = ROOT / "outputs" / "visualizations"
FIGURE_PATH = OUTPUT_DIR / "f8_metro_demeaned_correlation_matrix.png"
CORR_CSV_PATH = OUTPUT_DIR / "f8_metro_demeaned_correlations.csv"

CHARGE_LEVEL_COL = "charge_level"
METRO_COL = "metro_area"
WEATHER_COLS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
]
CORR_OUTCOMES = [
    "sessions",
    "energy_kwh_sum",
    "energy_kwh_mean",
    "charge_duration_mean",
]


def prepare_panel() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df = df[df[CHARGE_LEVEL_COL].isin(["L2", "DCFC"])].copy()
    return df.dropna(subset=[METRO_COL, CHARGE_LEVEL_COL, *WEATHER_COLS, *CORR_OUTCOMES])


def write_metro_demeaned_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for charge_level, charge_df in df.groupby(CHARGE_LEVEL_COL):
        demeaned = charge_df.copy()
        cols = WEATHER_COLS + CORR_OUTCOMES
        for col in cols:
            demeaned[col] = demeaned[col] - demeaned.groupby(METRO_COL)[col].transform(
                "mean"
            )
        for weather_col in WEATHER_COLS:
            for outcome in CORR_OUTCOMES:
                valid = demeaned[[weather_col, outcome]].dropna()
                corr = valid[weather_col].corr(valid[outcome])
                rows.append(
                    {
                        "charge_level": charge_level,
                        "weather_variable": weather_col,
                        "outcome": outcome,
                        "pearson_corr_metro_demeaned": round(float(corr), 4)
                        if pd.notna(corr)
                        else "",
                        "n": len(valid),
                    }
                )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(CORR_CSV_PATH, index=False)
    return correlations


def plot_f8(correlations: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weather_labels = {
        "temperature_2m_mean": "Mean T",
        "temperature_2m_max": "Max T",
        "temperature_2m_min": "Min T",
        "precipitation_sum": "Precip",
        "relative_humidity_2m_mean": "Humidity",
        "wind_speed_10m_max": "Wind",
    }
    outcome_labels = {
        "sessions": "Sessions",
        "energy_kwh_sum": "Total kWh",
        "energy_kwh_mean": "Mean kWh",
        "charge_duration_mean": "Mean duration",
    }
    fig = plt.figure(figsize=(13.8, 6.2))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.045],
        left=0.08,
        right=0.92,
        bottom=0.2,
        top=0.82,
        wspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])
    image = None

    for ax, charge_level in zip(axes, ["L2", "DCFC"]):
        sub = correlations[correlations[CHARGE_LEVEL_COL] == charge_level]
        matrix = (
            sub.pivot(
                index="weather_variable",
                columns="outcome",
                values="pearson_corr_metro_demeaned",
            )
            .reindex(WEATHER_COLS)
            .reindex(columns=CORR_OUTCOMES)
            .astype(float)
        )
        image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.25, vmax=0.25)
        ax.set_title(charge_level, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(CORR_OUTCOMES)))
        ax.set_xticklabels(
            [outcome_labels[col] for col in CORR_OUTCOMES],
            rotation=30,
            ha="right",
            fontsize=10,
        )
        ax.set_yticks(range(len(WEATHER_COLS)))
        ax.set_yticklabels([weather_labels[col] for col in WEATHER_COLS], fontsize=10)
        for i in range(len(WEATHER_COLS)):
            for j in range(len(CORR_OUTCOMES)):
                value = matrix.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.12 else "#111827",
                )

    fig.suptitle(
        "F8: Metro-Demeaned Weather-Charging Correlations",
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Pearson correlation")
    fig.text(
        0.5,
        0.02,
        "Correlations are computed after subtracting each metro's mean within charger level.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    df = prepare_panel()
    correlations = write_metro_demeaned_correlations(df)
    plot_f8(correlations)
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {CORR_CSV_PATH}")
    print(f"Rows used: {len(df):,}")


if __name__ == "__main__":
    main()

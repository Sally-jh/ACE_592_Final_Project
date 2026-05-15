#!/usr/bin/env python3
"""F6: L2 vs. DCFC weather heterogeneity."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jinyangli/Downloads/evwatts.public 2")
INPUT_PATH = Path("/Users/jinyangli/Downloads/metro_day_charger_panel.parquet")
OUTPUT_DIR = ROOT / "outputs" / "visualizations"
FIGURE_PATH = OUTPUT_DIR / "f6_l2_vs_dcfc_weather_heterogeneity.png"
BINNED_CSV_PATH = OUTPUT_DIR / "f6_l2_vs_dcfc_binned_data.csv"

TEMP_COL = "temperature_2m_mean"
CHARGE_LEVEL_COL = "charge_level"
METRO_COL = "metro_area"
OUTCOMES = {
    "energy_kwh_mean": "Mean kWh per session",
    "sessions": "Session count",
}


def prepare_panel() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df = df[df[CHARGE_LEVEL_COL].isin(["L2", "DCFC"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df = df.dropna(subset=[METRO_COL, CHARGE_LEVEL_COL, TEMP_COL, *OUTCOMES])

    for outcome in OUTCOMES:
        metro_mean = df.groupby([METRO_COL, CHARGE_LEVEL_COL])[outcome].transform("mean")
        month_mean = df.groupby(["month", CHARGE_LEVEL_COL])[outcome].transform("mean")
        level_mean = df.groupby(CHARGE_LEVEL_COL)[outcome].transform("mean")
        df[f"{outcome}_demeaned"] = df[outcome] - metro_mean - month_mean + level_mean

    return df


def build_binned_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_bin"] = pd.cut(df[TEMP_COL], bins=np.arange(-26, 43, 2), include_lowest=True)
    rows = []

    for charge_level, charge_df in df.groupby(CHARGE_LEVEL_COL):
        for temp_bin, bin_df in charge_df.groupby("temp_bin", observed=True):
            if len(bin_df) < 25:
                continue
            row = {
                "charge_level": charge_level,
                "temp_bin": str(temp_bin),
                "temp_mid_c": round(float(temp_bin.mid), 2),
                "n_metro_days": int(len(bin_df)),
                "mean_temperature_c": round(float(bin_df[TEMP_COL].mean()), 3),
            }
            for outcome in OUTCOMES:
                demeaned = f"{outcome}_demeaned"
                mean_value = float(bin_df[demeaned].mean())
                standard_error = float(bin_df[demeaned].sem())
                row[f"{outcome}_demeaned_mean"] = round(mean_value, 4)
                row[f"{outcome}_demeaned_se"] = round(standard_error, 4)
                row[f"{outcome}_demeaned_ci95_low"] = round(
                    mean_value - 1.96 * standard_error, 4
                )
                row[f"{outcome}_demeaned_ci95_high"] = round(
                    mean_value + 1.96 * standard_error, 4
                )
                row[f"{outcome}_raw_mean"] = round(float(bin_df[outcome].mean()), 3)
            rows.append(row)

    binned = pd.DataFrame(rows).sort_values(["charge_level", "temp_mid_c"])
    binned.to_csv(BINNED_CSV_PATH, index=False)
    return binned


def plot_f6(binned: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8), sharex=True)
    colors = {"L2": "#2563eb", "DCFC": "#dc2626"}
    markers = {"L2": "o", "DCFC": "s"}

    for ax, (outcome, title) in zip(axes, OUTCOMES.items()):
        y_col = f"{outcome}_demeaned_mean"
        low_col = f"{outcome}_demeaned_ci95_low"
        high_col = f"{outcome}_demeaned_ci95_high"
        for charge_level in ["L2", "DCFC"]:
            sub = binned[binned[CHARGE_LEVEL_COL] == charge_level]
            ax.fill_between(
                sub["temp_mid_c"].to_numpy(),
                sub[low_col].to_numpy(),
                sub[high_col].to_numpy(),
                color=colors[charge_level],
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                sub["temp_mid_c"],
                sub[y_col],
                color=colors[charge_level],
                marker=markers[charge_level],
                linewidth=2.2,
                markersize=6,
                label=charge_level,
            )
            for _, row in sub.iterrows():
                ax.scatter(
                    row["temp_mid_c"],
                    row[y_col],
                    s=max(22, min(160, row["n_metro_days"] / 14)),
                    color=colors[charge_level],
                    alpha=0.22,
                    linewidth=0,
                )

        ax.axhline(0, color="#475569", linewidth=1, linestyle="--", alpha=0.75)
        ax.grid(True, color="#e2e8f0", linewidth=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily mean temperature bin (C)")
        ax.set_ylabel("Metro + month demeaned value")
        ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "F6: L2 vs. DCFC Weather Heterogeneity",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Binned metro-day-charger panel. Outcomes are demeaned by metro and month-of-year within charger level; shaded bands are 95% CI.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    df = prepare_panel()
    binned = build_binned_data(df)
    plot_f6(binned)
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {BINNED_CSV_PATH}")
    print(f"Rows used: {len(df):,}")


if __name__ == "__main__":
    main()

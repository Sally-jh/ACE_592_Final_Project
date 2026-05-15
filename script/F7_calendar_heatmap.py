#!/usr/bin/env python3
"""F7: calendar heatmap of charging sessions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/Users/jinyangli/Downloads/evwatts.public 2")
SESSION_INPUT_PATH = Path("/Users/jinyangli/Downloads/merged_metro_day.parquet")
OUTPUT_DIR = ROOT / "outputs" / "visualizations"
FIGURE_PATH = OUTPUT_DIR / "f7_calendar_heatmap.png"
DATA_CSV_PATH = OUTPUT_DIR / "f7_calendar_heatmap_data.csv"
CHARGE_LEVEL_COL = "charge_level"


def build_calendar_heatmap() -> pd.DataFrame:
    sessions = pd.read_parquet(
        SESSION_INPUT_PATH,
        columns=["session_id", "start_datetime", "charge_level"],
    )
    sessions = sessions[sessions[CHARGE_LEVEL_COL].isin(["L2", "DCFC"])].copy()
    sessions["start_datetime"] = pd.to_datetime(sessions["start_datetime"])
    sessions["date"] = sessions["start_datetime"].dt.date
    sessions["day_of_week"] = sessions["start_datetime"].dt.dayofweek
    sessions["hour"] = sessions["start_datetime"].dt.hour

    hourly = (
        sessions.groupby([CHARGE_LEVEL_COL, "date", "day_of_week", "hour"], observed=True)
        .size()
        .rename("session_count")
        .reset_index()
    )
    heatmap = (
        hourly.groupby([CHARGE_LEVEL_COL, "day_of_week", "hour"], observed=True)[
            "session_count"
        ]
        .mean()
        .rename("mean_session_count")
        .reset_index()
    )
    heatmap.to_csv(DATA_CSV_PATH, index=False)
    return heatmap


def plot_f7(heatmap: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    matrices = {}

    for charge_level in ["L2", "DCFC"]:
        matrix = np.zeros((7, 24))
        sub = heatmap[heatmap[CHARGE_LEVEL_COL] == charge_level]
        for _, row in sub.iterrows():
            matrix[int(row["day_of_week"]), int(row["hour"])] = row[
                "mean_session_count"
            ]
        matrices[charge_level] = matrix

    fig = plt.figure(figsize=(14.8, 6.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 0.055],
        left=0.07,
        right=0.96,
        bottom=0.18,
        top=0.82,
        wspace=0.16,
        hspace=0.42,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1], sharey=fig.axes[0])]
    caxes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    for ax, cax, charge_level in zip(axes, caxes, ["L2", "DCFC"]):
        matrix = matrices[charge_level]
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=float(matrix.max()),
        )
        ax.set_title(charge_level, fontsize=14, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.xaxis.labelpad = 8
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(7))
        ax.set_yticklabels(day_labels)
        ax.set_ylabel("Day of week")
        cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
        cbar.set_label(f"{charge_level} mean sessions")

    fig.suptitle("F7: Calendar Heatmap of Charging Sessions", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Session-level data from merged_metro_day.parquet. Each panel uses its own color scale to reveal within-charger calendar patterns.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.savefig(FIGURE_PATH, dpi=220)
    plt.close(fig)


def main() -> None:
    heatmap = build_calendar_heatmap()
    plot_f7(heatmap)
    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {DATA_CSV_PATH}")


if __name__ == "__main__":
    main()

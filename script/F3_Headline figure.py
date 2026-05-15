"""
F3 — Headline figure: Temperature vs. kWh per session (binned scatter).

Story: Energy delivered per session is U-shaped in temperature.
       Cold -> more kWh (battery range loss). Hot -> more kWh (AC drain).
       Mild -> less.

Approach:
  1. Aggregate session-level data to metro x date level (mean kWh per session).
  2. Demean by metro x month-of-year so that level differences across metros
     and seasonality are absorbed; only the within-metro weather signal remains.
  3. Bin metro-days by 2 C bins of mean daily temperature.
  4. Plot bin means with 95% CI ribbon, plus a quadratic fit to highlight U.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(
    r"C:\Users\chenq6\OneDrive - University of Illinois - Urbana"
    r"\Courses\4th semester of PhD\ACE 535 (592)"
    r"\ACE 592 in-class project\final_project"
)
DATA_PATH    = PROJECT_ROOT / "data" / "processed" / "merged_metro_day.parquet"
FIGURES_DIR  = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load session-level data (despite the file name, this is session-level).
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print(f"Loaded {len(df):,} sessions across "
      f"{df['metro_area'].nunique()} metros, "
      f"{df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------------------------------------
# 2. Aggregate to metro x date.
#    mean_kwh: average energy delivered per session that day in that metro.
#    Weather columns are constant within a metro-day, so .first() is fine.
# ---------------------------------------------------------------------------
metro_day = (
    df.groupby(["metro_area", "date"], as_index=False)
      .agg(mean_kwh   =("energy_kwh",          "mean"),
           n_sessions =("session_id",          "count"),
           temp_mean  =("temperature_2m_mean", "first"))
)

# Quality filters: drop metro-days with too few sessions to be stable,
# and drop missing weather/kWh.
metro_day = metro_day.dropna(subset=["temp_mean", "mean_kwh"])
metro_day = metro_day[metro_day["n_sessions"] >= 3]
metro_day = metro_day[metro_day["mean_kwh"]  > 0]

print(f"After filtering: {len(metro_day):,} metro-days.")

# ---------------------------------------------------------------------------
# 3. Demean by metro x month-of-year.
#    We subtract the (metro, month) cell mean and add back the grand mean,
#    so the y-axis stays in interpretable kWh units (not deviations from 0).
# ---------------------------------------------------------------------------
metro_day["month"] = metro_day["date"].dt.month
grand_mean = metro_day["mean_kwh"].mean()
metro_month_mean = (
    metro_day.groupby(["metro_area", "month"])["mean_kwh"].transform("mean")
)
metro_day["mean_kwh_adj"] = metro_day["mean_kwh"] - metro_month_mean + grand_mean

# ---------------------------------------------------------------------------
# 4. Bin temperature into 2 C bins.
# ---------------------------------------------------------------------------
bin_edges   = np.arange(-25, 41, 2)            # bins from -25 to +40 C
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

metro_day["temp_bin"] = pd.cut(
    metro_day["temp_mean"],
    bins=bin_edges,
    labels=bin_centers,
    include_lowest=True,
).astype(float)

binned = (
    metro_day.dropna(subset=["temp_bin"])
             .groupby("temp_bin", as_index=False)
             .agg(y   =("mean_kwh_adj", "mean"),
                  sd  =("mean_kwh_adj", "std"),
                  n   =("mean_kwh_adj", "count"))
)
binned["sem"]    = binned["sd"] / np.sqrt(binned["n"])
binned["ci_lo"]  = binned["y"] - 1.96 * binned["sem"]
binned["ci_hi"]  = binned["y"] + 1.96 * binned["sem"]

# Drop sparse bins (fewer than 30 metro-days) to avoid noisy tails.
binned = binned[binned["n"] >= 30].reset_index(drop=True)

print(f"\nBin summary:\n{binned[['temp_bin','y','n']].to_string(index=False)}")

# ---------------------------------------------------------------------------
# 5. Quadratic fit weighted by bin n (to find the U-shape minimum).
# ---------------------------------------------------------------------------
coef = np.polyfit(binned["temp_bin"], binned["y"], 2, w=binned["n"])
a, b, c = coef                                  # y = a*t^2 + b*t + c
t_min = -b / (2 * a) if a != 0 else np.nan      # vertex of the parabola
print(f"\nQuadratic fit:  y = {a:.4f}*T^2 + {b:.4f}*T + {c:.3f}")
print(f"U-shape minimum at T = {t_min:.1f} C")

xs       = np.linspace(binned["temp_bin"].min(), binned["temp_bin"].max(), 200)
fit_line = np.polyval(coef, xs)

# ---------------------------------------------------------------------------
# 6. Plot.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

BLUE = "#0072B2"   # Okabe-Ito blue
GREY = "#888888"

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)

# Cold / hot reference bands
ax.axvspan(-25,  0, color="#cfe2f3", alpha=0.25, zorder=0)
ax.axvspan( 30, 40, color="#f4cccc", alpha=0.25, zorder=0)

# Quadratic fit
ax.plot(xs, fit_line,
        color=GREY, linestyle="--", lw=1.5,
        label=f"Quadratic fit (min at {t_min:.1f}\u00B0C)", zorder=2)

# 95% CI ribbon
ax.fill_between(binned["temp_bin"], binned["ci_lo"], binned["ci_hi"],
                color=BLUE, alpha=0.18, zorder=3, label="95% CI")

# Bin means
ax.plot(binned["temp_bin"], binned["y"],
        color=BLUE, lw=1.2, zorder=4)
ax.scatter(binned["temp_bin"], binned["y"],
           s=44, color=BLUE, edgecolor="white", linewidth=0.9,
           zorder=5, label="Bin mean")

# Reference: grand mean
ax.axhline(grand_mean, color=GREY, lw=0.7, linestyle=":", alpha=0.7)

# Labels
ax.set_xlabel("Daily mean temperature (\u00B0C)")
ax.set_ylabel("Mean kWh per session\n(demeaned by metro \u00D7 month-of-year)")
ax.set_title("F3: Energy delivered per session is U-shaped in temperature",
             weight="bold", pad=12)

# Annotations for the cold / hot bands
ax.text(-12, ax.get_ylim()[1], "Cold:\nmore kWh\n(range loss)",
        ha="center", va="top", fontsize=9, color="#1f4e79")
ax.text( 35, ax.get_ylim()[1], "Hot:\nmore kWh\n(AC drain)",
        ha="center", va="top", fontsize=9, color="#9c0000")

# Spines and grid
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5)

# Legend
ax.legend(frameon=False, loc="lower left", fontsize=10)

plt.tight_layout()

out_png = FIGURES_DIR / "F3_temp_vs_kwh.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print(f"\nSaved: {out_png}")

plt.show()
"""
F4 — Precipitation vs. daily session count.

Story: Heavier precipitation reduces daily charging volume,
       with the strongest effect on extreme-rain days (>25 mm).

Approach:
  1. Aggregate session-level data to metro x date.
  2. Express each metro-day's session count as a ratio of the
     (metro x month-of-year) mean.
  3. Bin precipitation into 5 NOAA-aligned categories.
  4. Show within-category ratio distribution as notched boxplots
     on a linear y-axis with equally spaced ticks (0.25 intervals).
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
DATA_PATH   = PROJECT_ROOT / "data" / "processed" / "merged_metro_day.parquet"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------------------------
# 2. Aggregate to metro-day
# ---------------------------------------------------------------------------
metro_day = (
    df.groupby(["metro_area", "date"], as_index=False)
      .agg(n_sessions=("session_id",         "count"),
           precip    =("precipitation_sum",  "first"))
)
metro_day = metro_day.dropna(subset=["precip", "n_sessions"])
metro_day = metro_day[metro_day["n_sessions"] >= 3]

# ---------------------------------------------------------------------------
# 3. Ratio relative to (metro x month-of-year) mean
# ---------------------------------------------------------------------------
metro_day["month"] = metro_day["date"].dt.month
mm_mean = (
    metro_day.groupby(["metro_area", "month"])["n_sessions"].transform("mean")
)
metro_day["ratio"] = metro_day["n_sessions"] / mm_mean

# ---------------------------------------------------------------------------
# 4. NOAA-aligned 5-bin categorization
# ---------------------------------------------------------------------------
bins      = [-1.0, 0.0, 2.5, 10.0, 25.0, np.inf]
cat_codes = ["none", "light", "moderate", "heavy", "extreme"]
metro_day["precip_cat"] = pd.cut(
    metro_day["precip"], bins=bins, labels=cat_codes, include_lowest=True,
)

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
summary = (
    metro_day.groupby("precip_cat", observed=True)["ratio"]
             .agg(n="count",
                  median="median",
                  q25=lambda x: x.quantile(0.25),
                  q75=lambda x: x.quantile(0.75))
             .reindex(cat_codes)
             .reset_index()
)
baseline = summary.loc[summary["precip_cat"] == "none", "median"].iloc[0]
summary["pct_change_vs_dry"] = (summary["median"] / baseline - 1) * 100

print("\n----- F4 summary (NOAA 5-bin, linear axis) -----")
print(summary.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize":14,
    "axes.labelsize":12,
})

display_labels = [
    "None\n(0 mm)",
    "Light\n(\u22642.5 mm)",
    "Moderate\n(2.5\u201310 mm)",
    "Heavy\n(10\u201325 mm)",
    "Extreme\n(>25 mm)",
]
# Sequential blue palette: lighter for dry, darker for wet.
colors = ["#e6f0fa", "#bcd9ec", "#7fb6e6", "#3a89c2", "#0c4a7e"]

groups = [
    metro_day.loc[metro_day["precip_cat"] == c, "ratio"].values
    for c in cat_codes
]

fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=120)

bp = ax.boxplot(
    groups,
    labels=display_labels,
    patch_artist=True,
    showfliers=False,
    notch=True,
    widths=0.6,
    whis=1.0,                    # <-- shorter whiskers so caps stay in-frame
    boxprops=dict(linewidth=1.0, edgecolor="#333"),
    medianprops=dict(color="#222", linewidth=2),
    whiskerprops=dict(color="#333", linewidth=1),
    capprops=dict(color="#333", linewidth=1.2),
)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

# Reference line at 1.0 = "normal day for this metro and month"
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7,
           zorder=0)

# ---- LINEAR y-axis: pure numbers, no x symbol ----
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75",
                    "1.0", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5"])
ax.set_ylim(0, 2.7)

# Annotation per box (numeric only, no x symbol)
for i, row in summary.iterrows():
    cat = row["precip_cat"]
    pct = row["pct_change_vs_dry"]
    txt = (
        f"median: {row['median']:.2f}\n"
        + ("baseline" if cat == "none" else f"{pct:+.1f}%")
        + f"\n(n = {int(row['n']):,})"
    )
    ax.text(
    i + 1, 2.6, txt,        
    ha="center", va="center", fontsize=8.5, color="#333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", linewidth=0.6, alpha=0.92),
)

# Labels
ax.set_xlabel("Daily total precipitation")
ax.set_ylabel("Charging volume relative to normal")
ax.set_title("F4: Heavier precipitation reduces charging volume",
             weight="bold", pad=12)

# Style
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)

plt.tight_layout()

out_png = FIGURES_DIR / "F4_precip_vs_sessions.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved: {out_png}")

plt.show()
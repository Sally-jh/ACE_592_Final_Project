"""
F5 — Snowfall vs. daily session count.

Story: Snow disrupts charging much more than rain. The behavioral channel
       that was muted in F4 (rain) becomes sharply visible once the
       weather event is severe enough to genuinely impair driving.

Approach (mirrors F4 for visual comparability):
  1. Aggregate session-level data to metro x date.
  2. Restrict to metros that experience meaningful snowfall
     (>= 30 snow days in the panel) - drops Phoenix, Miami, LA, etc.
  3. Express each metro-day's session count as a ratio of the
     (metro x month-of-year) mean.
  4. Bin daily snowfall (cm) into 4 categories:
        None     : = 0 cm
        Light    : 0 < s <= 2.5 cm    (dusting, light snow)
        Moderate : 2.5 < s <= 10 cm   (a few inches up to ~4")
        Heavy    : s > 10 cm          (major snowstorm)
  5. Notched boxplot on a linear y-axis (matches F4 for side-by-side viewing).
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
      .agg(n_sessions=("session_id",   "count"),
           snow      =("snowfall_sum", "first"))
)
metro_day = metro_day.dropna(subset=["snow", "n_sessions"])
metro_day = metro_day[metro_day["n_sessions"] >= 3]

# ---------------------------------------------------------------------------
# 3. Restrict to snow-prone metros (>= 30 snow days in the panel).
# ---------------------------------------------------------------------------
snow_day_count = (
    metro_day.assign(has_snow=lambda d: d["snow"] > 0)
             .groupby("metro_area")["has_snow"].sum()
             .sort_values(ascending=False)
)
SNOW_DAY_THRESHOLD = 30
snow_metros = snow_day_count[snow_day_count >= SNOW_DAY_THRESHOLD].index.tolist()

print("\n----- Snow-prone metros (kept) -----")
for m in snow_metros:
    print(f"  {m:<25s}: {int(snow_day_count[m]):>4d} snow days")

dropped = [m for m in snow_day_count.index if m not in snow_metros]
print("\n----- Dropped (insufficient snow) -----")
for m in dropped:
    print(f"  {m:<25s}: {int(snow_day_count[m]):>4d} snow days")

metro_day = metro_day[metro_day["metro_area"].isin(snow_metros)].copy()
print(f"\nAnalytic sample: {len(metro_day):,} metro-days from "
      f"{len(snow_metros)} snow-prone metros.")

# ---------------------------------------------------------------------------
# 4. Ratio relative to (metro x month-of-year) mean
# ---------------------------------------------------------------------------
metro_day["month"] = metro_day["date"].dt.month
mm_mean = (
    metro_day.groupby(["metro_area", "month"])["n_sessions"].transform("mean")
)
metro_day["ratio"] = metro_day["n_sessions"] / mm_mean

# ---------------------------------------------------------------------------
# 5. Categorize snowfall (cm)
# ---------------------------------------------------------------------------
bins      = [-1.0, 0.0, 2.5, 10.0, np.inf]
cat_codes = ["none", "light", "moderate", "heavy"]
metro_day["snow_cat"] = pd.cut(
    metro_day["snow"], bins=bins, labels=cat_codes, include_lowest=True,
)

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
summary = (
    metro_day.groupby("snow_cat", observed=True)["ratio"]
             .agg(n="count",
                  median="median",
                  q25=lambda x: x.quantile(0.25),
                  q75=lambda x: x.quantile(0.75))
             .reindex(cat_codes)
             .reset_index()
)
baseline = summary.loc[summary["snow_cat"] == "none", "median"].iloc[0]
summary["pct_change_vs_dry"] = (summary["median"] / baseline - 1) * 100

print("\n----- F5 summary -----")
print(summary.to_string(index=False))
print()

# ---------------------------------------------------------------------------
# 7. Plot (same visual style as F4 for direct comparability)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     11,
    "axes.titlesize":14,
    "axes.labelsize":12,
})

display_labels = [
    "None\n(0 cm)",
    "Light\n(\u22642.5 cm)",
    "Moderate\n(2.5\u201310 cm)",
    "Heavy\n(>10 cm)",
]
# Cool palette (gray -> blue-gray -> deep blue) for snow.
colors = ["#ececec", "#a8b8d4", "#5874a4", "#2a3e6e"]

groups = [
    metro_day.loc[metro_day["snow_cat"] == c, "ratio"].values
    for c in cat_codes
]

fig, ax = plt.subplots(figsize=(8.5, 6.8), dpi=120)

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

# ---- LINEAR y-axis: pure numbers, no x symbol (matches F4) ----
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75",
                    "1.0", "1.25", "1.5", "1.75", "2.0", "2.25", "2.5"])
ax.set_ylim(0, 2.7)

# Annotation per box (numeric only, no x symbol)
for i, row in summary.iterrows():
    cat = row["snow_cat"]
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
ax.set_xlabel("Daily snowfall")
ax.set_ylabel("Charging volume relative to normal)")
ax.set_title("F5: Heavier snowfall reduces charging volume",
             weight="bold", pad=24)

# Subtitle: how the sample is restricted.
ax.text(0.5, 1.03,
        f"Restricted to {len(snow_metros)} snow-prone metros "
        f"(\u226530 snow days in 2019\u20132022)",
        transform=ax.transAxes, ha="center",
        fontsize=10, color="#666", style="italic")

# Style
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)

plt.tight_layout()

out_png = FIGURES_DIR / "F5_snow_vs_sessions.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Saved: {out_png}")

plt.show()
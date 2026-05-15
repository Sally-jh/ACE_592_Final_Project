#!/usr/bin/env python3
"""T2.1 light econometric modeling for the EV charging/weather project."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-evwatts")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/evwatts-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


ROOT = Path("/Users/jinyangli/Downloads/evwatts.public 2")
SESSION_PANEL = Path("/Users/jinyangli/Downloads/merged_metro_day.parquet")
OUTPUT_DIR = ROOT / "outputs" / "models"
PANEL_OUT = OUTPUT_DIR / "metro_day_model_panel.parquet"
TABLE_CSV = OUTPUT_DIR / "t2_1_regression_table.csv"
TABLE_MD = OUTPUT_DIR / "t2_1_regression_table.md"
TABLE_HTML = OUTPUT_DIR / "t2_1_regression_table.html"
TABLE_PNG = OUTPUT_DIR / "t2_1_regression_table.png"
WRITEUP_MD = OUTPUT_DIR / "t2_1_modeling_writeup.md"


def star(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def fmt_coef(result, name: str) -> str:
    if name not in result.params:
        return ""
    coef = result.params[name]
    se = result.bse[name]
    pvalue = result.pvalues[name]
    return f"{coef:.4f}{star(pvalue)}\n({se:.4f})"


def build_metro_day_panel() -> pd.DataFrame:
    cols = [
        "metro_area",
        "date",
        "session_id",
        "energy_kwh",
        "charge_duration",
        "temperature_2m_mean",
        "precipitation_sum",
    ]
    sessions = pd.read_parquet(SESSION_PANEL, columns=cols)
    sessions = sessions.dropna(
        subset=["metro_area", "date", "energy_kwh", "temperature_2m_mean", "precipitation_sum"]
    )
    sessions = sessions[(sessions["energy_kwh"] > 0) & (sessions["charge_duration"] > 0)]

    panel = (
        sessions.groupby(["metro_area", "date"], observed=True)
        .agg(
            n_sessions=("session_id", "count"),
            total_kwh=("energy_kwh", "sum"),
            mean_kwh=("energy_kwh", "mean"),
            mean_duration_h=("charge_duration", "mean"),
            temperature_2m_mean=("temperature_2m_mean", "first"),
            precipitation_sum=("precipitation_sum", "first"),
        )
        .reset_index()
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year_month"] = panel["date"].dt.to_period("M").astype(str)
    panel["day_of_week"] = panel["date"].dt.day_name()
    panel["temp_sq"] = panel["temperature_2m_mean"] ** 2
    panel["log_mean_kwh"] = np.log(panel["mean_kwh"])
    panel["log_n_sessions"] = np.log(panel["n_sessions"])
    panel["precip_bin"] = pd.cut(
        panel["precipitation_sum"],
        bins=[-0.001, 0, 2, 10, np.inf],
        labels=["No precip", "Light (<2mm)", "Moderate (2-10mm)", "Heavy (>10mm)"],
        include_lowest=True,
    )
    panel = panel.dropna(
        subset=["log_mean_kwh", "log_n_sessions", "precip_bin", "year_month", "day_of_week"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUT, index=False)
    return panel


def run_models(panel: pd.DataFrame):
    model_1 = smf.ols(
        "log_mean_kwh ~ temperature_2m_mean + temp_sq + C(metro_area) + C(year_month)",
        data=panel,
    ).fit(cov_type="HC1")
    model_2 = smf.ols(
        "log_n_sessions ~ C(precip_bin, Treatment(reference='No precip')) + C(metro_area) + C(day_of_week)",
        data=panel,
    ).fit(cov_type="HC1")
    model_3 = smf.ols(
        "log_n_sessions ~ temperature_2m_mean + temp_sq + C(precip_bin, Treatment(reference='No precip')) + C(metro_area) + C(year_month) + C(day_of_week)",
        data=panel,
    ).fit(cov_type="HC1")
    return model_1, model_2, model_3


def implied_min_temperature(result) -> float | None:
    beta_1 = result.params.get("temperature_2m_mean")
    beta_2 = result.params.get("temp_sq")
    if beta_1 is None or beta_2 is None or beta_2 == 0:
        return None
    return float(-beta_1 / (2 * beta_2))


def build_table(models) -> pd.DataFrame:
    m1, m2, m3 = models
    precip_light = "C(precip_bin, Treatment(reference='No precip'))[T.Light (<2mm)]"
    precip_moderate = "C(precip_bin, Treatment(reference='No precip'))[T.Moderate (2-10mm)]"
    precip_heavy = "C(precip_bin, Treatment(reference='No precip'))[T.Heavy (>10mm)]"

    rows = [
        ("Mean temperature (C)", "temperature_2m_mean"),
        ("Mean temperature squared", "temp_sq"),
        ("Light precipitation (<2mm)", precip_light),
        ("Moderate precipitation (2-10mm)", precip_moderate),
        ("Heavy precipitation (>10mm)", precip_heavy),
    ]
    table_rows = []
    for label, key in rows:
        table_rows.append(
            {
                "Variable": label,
                "(1) log(mean kWh)": fmt_coef(m1, key),
                "(2) log(sessions)": fmt_coef(m2, key),
                "(3) log(sessions)": fmt_coef(m3, key),
            }
        )

    min_temp = implied_min_temperature(m1)
    table_rows.extend(
        [
            {
                "Variable": "Metro fixed effects",
                "(1) log(mean kWh)": "Yes",
                "(2) log(sessions)": "Yes",
                "(3) log(sessions)": "Yes",
            },
            {
                "Variable": "Year-month fixed effects",
                "(1) log(mean kWh)": "Yes",
                "(2) log(sessions)": "No",
                "(3) log(sessions)": "Yes",
            },
            {
                "Variable": "Day-of-week fixed effects",
                "(1) log(mean kWh)": "No",
                "(2) log(sessions)": "Yes",
                "(3) log(sessions)": "Yes",
            },
            {
                "Variable": "Implied U-shape minimum (C)",
                "(1) log(mean kWh)": f"{min_temp:.2f}" if min_temp is not None else "",
                "(2) log(sessions)": "",
                "(3) log(sessions)": "",
            },
            {
                "Variable": "Observations",
                "(1) log(mean kWh)": f"{int(m1.nobs):,}",
                "(2) log(sessions)": f"{int(m2.nobs):,}",
                "(3) log(sessions)": f"{int(m3.nobs):,}",
            },
            {
                "Variable": "R-squared",
                "(1) log(mean kWh)": f"{m1.rsquared:.3f}",
                "(2) log(sessions)": f"{m2.rsquared:.3f}",
                "(3) log(sessions)": f"{m3.rsquared:.3f}",
            },
        ]
    )
    return pd.DataFrame(table_rows)


def write_markdown_table(table: pd.DataFrame, models) -> None:
    m1, m2, m3 = models
    min_temp = implied_min_temperature(m1)
    lines = [
        "# T2.1 Light Econometric Modeling",
        "",
        "Dependent variables are log-transformed. Heteroskedasticity-robust HC1 standard errors are in parentheses.",
        "Significance: *** p<0.01, ** p<0.05, * p<0.10.",
        "",
        table.to_markdown(index=False),
        "",
        "## Key numbers",
        "",
        f"- Implied temperature minimum for log(mean kWh): {min_temp:.2f} C.",
        f"- Model 1 observations: {int(m1.nobs):,}; R-squared: {m1.rsquared:.3f}.",
        f"- Model 2 observations: {int(m2.nobs):,}; R-squared: {m2.rsquared:.3f}.",
        f"- Model 3 observations: {int(m3.nobs):,}; R-squared: {m3.rsquared:.3f}.",
    ]
    TABLE_MD.write_text("\n".join(lines), encoding="utf-8")


def write_html_table(table: pd.DataFrame) -> None:
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #17202a; }}
h1 {{ font-size: 22px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px 10px; vertical-align: top; white-space: pre-line; }}
th {{ background: #eef2ff; text-align: left; }}
td:not(:first-child), th:not(:first-child) {{ text-align: center; }}
.note {{ margin-top: 14px; color: #475569; font-size: 13px; }}
</style>
</head>
<body>
<h1>T2.1 Light Econometric Modeling</h1>
{table.to_html(index=False, escape=False)}
<div class="note">Robust HC1 standard errors in parentheses. Significance: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.10.</div>
</body>
</html>"""
    TABLE_HTML.write_text(html, encoding="utf-8")


def write_png_table(table: pd.DataFrame) -> None:
    display = table.copy()
    display.columns = ["Variable", "(1)\nlog(mean kWh)", "(2)\nlog(sessions)", "(3)\nlog(sessions)"]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.axis("off")
    ax.text(
        0,
        1.04,
        "T2.1 Light Econometric Modeling",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0,
        0.995,
        "OLS estimates with metro/calendar fixed effects; robust HC1 standard errors in parentheses.",
        transform=ax.transAxes,
        fontsize=10,
        color="#475569",
        va="top",
    )

    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0.02, 1, 0.9],
        colWidths=[0.38, 0.21, 0.205, 0.205],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.4)
    tbl.scale(1, 1.35)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d8dee9")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e8eefc")
            cell.set_text_props(fontweight="bold", color="#17202a")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        if col == 0:
            cell.set_text_props(ha="left")

    ax.text(
        0,
        -0.025,
        "Significance: *** p<0.01, ** p<0.05, * p<0.10.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
        va="top",
    )
    fig.savefig(TABLE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_modeling_writeup(models) -> None:
    m1, m2, m3 = models
    min_temp = implied_min_temperature(m1)
    text = f"""# T2.1 Modeling Write-up

To add a light econometric layer to the visualization analysis, I estimate three pooled OLS models on a metro-day panel constructed from the session-level charging data. The first model regresses log mean kWh per session on daily mean temperature and its square, with metro and year-month fixed effects. This specification is intended to summarize the nonlinear temperature relationship shown visually in the binned temperature figures while controlling for persistent differences across metros and common monthly seasonality. The implied minimum of the fitted quadratic occurs at {min_temp:.2f} degrees Celsius, which is the temperature at which predicted kWh per session is lowest in this specification.

The second model regresses log session count on precipitation bins, with metro and day-of-week fixed effects. The omitted category is no precipitation. The precipitation bins separate light precipitation below 2 mm, moderate precipitation between 2 and 10 mm, and heavy precipitation above 10 mm. This specification asks whether rainy or snowy days are associated with fewer charging sessions after accounting for the metro and regular weekly charging pattern.

The third model is a combined robustness specification for log session count. It includes both the temperature quadratic and precipitation bins, along with metro, year-month, and day-of-week fixed effects. This column is useful for the slide deck because it shows whether the precipitation-bin estimates remain similar when temperature and seasonal controls are included in the same model.

These regressions should be interpreted as descriptive rather than causal. The fixed effects absorb important metro-level and calendar-level differences, but the estimates do not isolate random weather shocks. The table is therefore best used as numerical backup for the visual patterns rather than as the main evidence of the project.
"""
    WRITEUP_MD.write_text(text, encoding="utf-8")


def main() -> None:
    panel = build_metro_day_panel()
    models = run_models(panel)
    table = build_table(models)
    table.to_csv(TABLE_CSV, index=False)
    write_markdown_table(table, models)
    write_html_table(table)
    write_png_table(table)
    write_modeling_writeup(models)
    print(f"Wrote {PANEL_OUT}")
    print(f"Wrote {TABLE_CSV}")
    print(f"Wrote {TABLE_MD}")
    print(f"Wrote {TABLE_HTML}")
    print(f"Wrote {TABLE_PNG}")
    print(f"Wrote {WRITEUP_MD}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

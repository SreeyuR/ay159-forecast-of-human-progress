"""
Plot Global Energy Consumption (EJ) = A / B, where:
  A = sum of energy consumption (from random_forest_ec.py) over the 42 countries
  B = predicted ratio EC_42/Global EC (from random_forest_ec_global_ratio_ind_countries.py)

Historical (1970-2020): actual global EC from ARIMA input data (blue).
Forecast (2020-2100): A/B, scaled so that 2020 equals the historical value (lines match at 2020).
Forecast + AI energy (2027-2100): Global EC + AI energy from get_ai_energy_for_years (red dashed).
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path

from ai_energy import get_ai_energy_for_years, get_ai_energy_uncertainty_for_years

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paths to outputs from the two models
FUTURE_COUNTRY_PREDICTIONS_PATH = DATA_DIR / "future_country_predictions.csv"
TEST_RATIO_SUM_PATH = DATA_DIR / "test_ratio_sum_by_year_ind.csv"
HISTORICAL_GLOBAL_EC_PATH = DATA_DIR / "ARIMA input data - Global EC.csv"

HISTORICAL_YEAR_MIN = 1970
HISTORICAL_YEAR_MAX = 2020
FORECAST_YEAR_MIN = 2020
FORECAST_YEAR_MAX = 2100
AI_ENERGY_PLOT_YEAR_MIN = 2027

# Kardashev scale index K: power in watts -> K = (log10(P_W) - 6) / 10; P_W = y_EJ * 1e18 / (365*24*3600)
def _kardashev_k(y_ej):
    """Convert global energy consumption (EJ/year) to Kardashev scale index K."""
    y = y_ej #np.maximum(np.asarray(y_ej, dtype=float), 1e-30)
    power_watts = y * 1e18 / (365 * 24 * 60 * 60)
    return (np.log10(power_watts) - 6) / 10


def _format_k_growth_mpl(g: float) -> str:
    """Format relative K growth over 5 years as mathtext, e.g. $3.28\\times 10^{-3}$."""
    if g == 0 or not np.isfinite(g):
        return "0"
    exp = int(math.floor(math.log10(abs(g))))
    mant = g / (10**exp)
    return rf"${mant:.2f}\times 10^{{{exp}}}$"


def save_forecast_results_table_png(forecast_df: pd.DataFrame, out_path: Path) -> None:
    """
    Table 1 style: 5-year rows 2025–2100; energy (EJ), K, and relative K growth over the
    prior 5 years: (K_t - K_{t-5}) / K_{t-5} (for 2025, K_2020 from the same forecast series).
    """
    years = np.arange(2025, FORECAST_YEAR_MAX + 1, 5, dtype=int)
    fc = forecast_df.set_index("Year")["Global_EC_EJ"]
    missing = [y for y in years if y not in fc.index]
    if missing:
        raise ValueError(f"Forecast missing years required for table: {missing[:5]}...")

    header0 = ["", "Projected values", "", ""]
    header1 = ["Year", "Energy consumption (EJ)", "K", ""]
    header2 = ["", "", "Value", "Growth rate (past 5 years)"]

    rows = []
    for y in years:
        ec = float(fc.loc[y])
        k = float(_kardashev_k(ec))
        y0 = y - 5
        k_prev = float(_kardashev_k(float(fc.loc[y0])))
        growth = (k - k_prev) / k_prev if k_prev != 0 else float("nan")
        rows.append(
            [
                str(int(y)),
                f"{ec:.2f}",
                f"{k:.5f}",
                _format_k_growth_mpl(growth),
            ]
        )

    cell_text = [header0, header1, header2] + rows
    nrows = len(cell_text)

    fig_h = max(8.0, 0.35 * nrows + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    serif = "Times New Roman"
    if serif not in {f.name for f in fm.fontManager.ttflist}:
        serif = "DejaVu Serif"

    table = ax.table(
        cellText=cell_text,
        loc="center",
        cellLoc="center",
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.8)

    header_gray = "#d9d9d9"
    for (r, c), cell in table.get_celld().items():
        cell.set_text_props(family=serif)
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if r < 3:
            cell.set_facecolor(header_gray)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white")
            cell.get_text().set_weight("normal")

    cap = (
        "Table 1. Final forecasting results. The predicted values for energy consumption, "
        "civilization development index K, as well as its growth rate over each 5-year period."
    )
    fig.text(0.5, 0.02, cap, ha="center", va="bottom", fontsize=9, family=serif)
    plt.subplots_adjust(bottom=0.12, top=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_historical_global_ec() -> pd.DataFrame:
    """Load historical global EC (EJ) for 1970-2020 from ARIMA input data."""
    df = pd.read_csv(HISTORICAL_GLOBAL_EC_PATH)
    df = df.rename(columns={"global_EC_EJ": "Global_EC_EJ"})
    df = df.dropna(subset=["Year", "Global_EC_EJ"])
    df["Year"] = df["Year"].astype(int)
    df = df[(df["Year"] >= HISTORICAL_YEAR_MIN) & (df["Year"] <= HISTORICAL_YEAR_MAX)]
    df = df.sort_values("Year").drop_duplicates(subset=["Year"], keep="last")
    return df[["Year", "Global_EC_EJ"]]


def load_A_sum_by_year() -> pd.DataFrame:
    """A = sum of Predicted_EC over 42 countries by year (from random_forest_ec.py output)."""
    df = pd.read_csv(FUTURE_COUNTRY_PREDICTIONS_PATH)
    df = df[(df["Year"] >= FORECAST_YEAR_MIN) & (df["Year"] <= FORECAST_YEAR_MAX)]
    A = df.groupby("Year", as_index=False)["Predicted_EC"].sum()
    A = A.rename(columns={"Predicted_EC": "A_sum"})
    return A


def load_B_ratio_sum_by_year() -> pd.DataFrame:
    """B = predicted ratio sum EC_42/Global EC by year (from ratio model output)."""
    df = pd.read_csv(TEST_RATIO_SUM_PATH)
    df = df[(df["Year"] >= FORECAST_YEAR_MIN) & (df["Year"] <= FORECAST_YEAR_MAX)]
    df = df.rename(columns={"predicted_ratio_sum": "B"})
    return df[["Year", "B"]]


def compute_forecast_global_ec(
    historical_2020: float,
    A_by_year: pd.DataFrame,
    B_by_year: pd.DataFrame,
) -> pd.DataFrame:
    """Global EC (EJ) = A/B for forecast years; scale so 2020 = historical_2020."""
    merged = A_by_year.merge(B_by_year, on="Year", how="inner")
    merged["Global_EC_EJ_raw"] = merged["A_sum"] / merged["B"]
    raw_2020 = merged.loc[merged["Year"] == 2020, "Global_EC_EJ_raw"]
    if raw_2020.empty:
        raise ValueError("No forecast value for 2020; cannot scale.")
    scale = historical_2020 / raw_2020.iloc[0]
    merged["Global_EC_EJ"] = merged["Global_EC_EJ_raw"] * scale
    return merged[["Year", "Global_EC_EJ"]]


def main():
    print("Loading historical global EC (1970-2020) ...")
    historical = load_historical_global_ec()
    if historical.empty:
        raise FileNotFoundError(
            f"No historical global EC found in {HISTORICAL_GLOBAL_EC_PATH} for {HISTORICAL_YEAR_MIN}-{HISTORICAL_YEAR_MAX}."
        )
    hist_2020 = historical.loc[historical["Year"] == 2020, "Global_EC_EJ"]
    if hist_2020.empty:
        raise ValueError("Historical data has no year 2020; cannot align forecast.")
    historical_2020_value = hist_2020.iloc[0]

    print("Loading A (sum of EC over 42 countries, 2020-2100) ...")
    A_by_year = load_A_sum_by_year()
    if A_by_year.empty:
        raise FileNotFoundError(
            f"No data in {FUTURE_COUNTRY_PREDICTIONS_PATH} for years {FORECAST_YEAR_MIN}-{FORECAST_YEAR_MAX}."
        )

    print("Loading B (ratio sum, 2020-2100) ...")
    B_by_year = load_B_ratio_sum_by_year()
    if B_by_year.empty:
        raise FileNotFoundError(
            f"No data in {TEST_RATIO_SUM_PATH} for years {FORECAST_YEAR_MIN}-{FORECAST_YEAR_MAX}."
        )

    print("Computing Global EC = A/B and scaling so 2020 = historical value ...")
    forecast = compute_forecast_global_ec(historical_2020_value, A_by_year, B_by_year)

    # Uncertainty band for forecast: (A / B^2) * 0.02, added/subtracted from each point (same scale as forecast)
    forecast_ab = forecast.merge(A_by_year, on="Year").merge(B_by_year, on="Year")
    raw_2020 = (forecast_ab.loc[forecast_ab["Year"] == 2020, "A_sum"].iloc[0] /
                forecast_ab.loc[forecast_ab["Year"] == 2020, "B"].iloc[0])
    scale = historical_2020_value / raw_2020
    forecast_ab["delta"] = (forecast_ab["A_sum"] / forecast_ab["B"] ** 2) * 0.02 * scale
    forecast_lower = forecast_ab["Global_EC_EJ"] - forecast_ab["delta"]
    forecast_upper = forecast_ab["Global_EC_EJ"] + forecast_ab["delta"]

    # Forecast + AI energy: Global EC + AI energy per year; plot 2027-2100 only
    years_forecast = forecast["Year"].values
    ai_energy_ej = get_ai_energy_for_years(years_forecast)
    _, ai_lower, ai_upper = get_ai_energy_uncertainty_for_years(years_forecast)
    forecast_plus_ai = forecast.copy()
    forecast_plus_ai["Global_EC_EJ_plus_AI"] = forecast_plus_ai["Global_EC_EJ"].values + ai_energy_ej
    forecast_plus_ai_2027 = forecast_plus_ai[forecast_plus_ai["Year"] >= AI_ENERGY_PLOT_YEAR_MIN]
    # Uncertainty band for Forecast + AI: Global_EC + ai_lower/ai_upper (same years 2027-2100)
    idx_2027 = (forecast_plus_ai_2027["Year"].values - 2020).astype(int)
    forecast_plus_ai_2027_lower = forecast_plus_ai_2027["Global_EC_EJ"].values + ai_lower[idx_2027]
    forecast_plus_ai_2027_upper = forecast_plus_ai_2027["Global_EC_EJ"].values + ai_upper[idx_2027]

    # Print values at 2100 for reference
    year_2100_mask = forecast_ab["Year"] == 2100
    if year_2100_mask.any():
        ec_2100 = float(forecast_ab.loc[year_2100_mask, "Global_EC_EJ"].iloc[0])
        delta_2100 = float(forecast_ab.loc[year_2100_mask, "delta"].iloc[0])
        k_2100 = float(_kardashev_k(ec_2100))
        print(f"Forecast 2100 Global EC (no AI): {ec_2100:.3f} EJ/year")
        print(f"Forecast 2100 uncertainty (no AI): ±{delta_2100:.3f} EJ/year")
        print(f"Kardashev index K at 2100 (no AI): {k_2100:.4f}")

        # With AI energy included
        plus_ai_2100_row = forecast_plus_ai.loc[forecast_plus_ai["Year"] == 2100]
        if not plus_ai_2100_row.empty:
            ec_plus_ai_2100 = float(plus_ai_2100_row["Global_EC_EJ_plus_AI"].iloc[0])
            # Find corresponding uncertainty band with AI for 2100
            mask_2027_2100 = forecast_plus_ai_2027["Year"] == 2100
            if mask_2027_2100.any():
                idx = np.where(forecast_plus_ai_2027["Year"].values == 2100)[0][0]
                lower_2100_ai = float(forecast_plus_ai_2027_lower[idx])
                upper_2100_ai = float(forecast_plus_ai_2027_upper[idx])
                delta_2100_ai = (upper_2100_ai - lower_2100_ai) / 2.0
            else:
                delta_2100_ai = float("nan")
            k_plus_ai_2100 = float(_kardashev_k(ec_plus_ai_2100))
            print(f"Forecast 2100 Global EC + AI: {ec_plus_ai_2100:.3f} EJ/year")
            print(f"Forecast 2100 uncertainty (+ AI): ±{delta_2100_ai:.3f} EJ/year")
            print(f"Kardashev index K at 2100 (+ AI): {k_plus_ai_2100:.4f}")

    # Plot: Historical (1970-2020) in blue, Forecast (2020-2100), Forecast + AI (2027-2100) red dashed
    plt.figure(figsize=(10, 6))
    plt.plot(
        historical["Year"],
        historical["Global_EC_EJ"],
        color="blue",
        linewidth=2,
        label="Historical data",
    )
    plt.fill_between(
        forecast_ab["Year"],
        forecast_lower,
        forecast_upper,
        color="tab:green",
        alpha=0.3,
    )
    plt.plot(
        forecast["Year"],
        forecast["Global_EC_EJ"],
        color="tab:green",
        linewidth=2,
        label="Forecast",
    )
    plt.fill_between(
        forecast_plus_ai_2027["Year"],
        forecast_plus_ai_2027_lower,
        forecast_plus_ai_2027_upper,
        color="red",
        alpha=0.3,
    )
    plt.plot(
        forecast_plus_ai_2027["Year"],
        forecast_plus_ai_2027["Global_EC_EJ_plus_AI"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Forecast + AI energy",
    )
    plt.xlabel("Year")
    plt.ylabel("Global Energy Consumption (EJ)")
    plt.title("Global Energy Consumption (EJ): Historical and Forecast (A/B)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(1965, 2105)

    # Right y-axis: get primary axis y ticks (global EC in EJ), compute K for each, put on right
    ax1 = plt.gca()
    left_ticks_ej = ax1.get_yticks()
    k_values = [float(_kardashev_k(t)) for t in left_ticks_ej]
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim()[0], ax1.get_ylim()[1])
    ax2.set_yticks(left_ticks_ej)
    ax2.set_yticklabels([f"{k:.4f}" for k in k_values])
    ax2.set_ylabel("index K on the Kardashev Scale")

    out_path = PLOTS_DIR / "global_energy_consumption.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot to", out_path)

    # Optional: save forecast series for reference
    forecast.to_csv(DATA_DIR / "global_energy_consumption_forecast.csv", index=False)
    print("Saved forecast series to data/global_energy_consumption_forecast.csv")

    table_path = PLOTS_DIR / "forecast_final_results_table.png"
    save_forecast_results_table_png(forecast, table_path)
    print("Saved table to", table_path)


if __name__ == "__main__":
    main()

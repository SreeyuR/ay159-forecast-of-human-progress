"""
Plot Global Energy Consumption (EJ) = A / B, where:
  A = sum of energy consumption (from random_forest_ec.py) over the 42 countries
  B = predicted ratio EC_42/Global EC (from random_forest_ec_global_ratio_ind_countries.py)

Historical (1970-2020): actual global EC from ARIMA input data (blue).
Forecast (2020-2100): A/B, scaled so that 2020 equals the historical value (lines match at 2020).
Forecast + AI energy (2027-2100): Global EC + AI energy from get_ai_energy_for_years (red dashed).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    out_path = PLOTS_DIR / "global_energy_consumption.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot to", out_path)

    # Optional: save forecast series for reference
    forecast.to_csv(DATA_DIR / "global_energy_consumption_forecast.csv", index=False)
    print("Saved forecast series to data/global_energy_consumption_forecast.csv")


if __name__ == "__main__":
    main()

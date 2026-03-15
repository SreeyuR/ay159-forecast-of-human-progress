"""
Train a Random Forest to predict the ratio (per country):
  energy consumption of 1 country / global energy consumption.

Features (per country, per year):
  1. EC_country / GDP_country
  2. GDP_country / GDP_global
  3. Population_country / Population_global

Training (1970--2019): actual EC from finaldata0411 only; GDP and Population from ARIMA CSVs; no ec_model.
Test/predict (2020--2100): GDP/Pop from ARIMA; EC from ec_model per country. Sum ratios for plot.
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from ai_energy import get_ai_energy_for_years, get_ai_energy_uncertainty_for_years

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
EC_MODEL_PATH = DATA_DIR / "ec_model.joblib"
PLOTS_DIR = DATA_DIR / "plots"
OUTPUT_DATA_DIR = DATA_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_TO_CODE = {
    "Australia": 36, "Austria": 40, "Belgium": 56, "Canada": 124,
    "Czechia": 203, "Denmark": 208, "Finland": 246, "France": 250,
    "Germany": 276, "Greece": 300, "Hungary": 348, "Iceland": 352,
    "Ireland": 372, "Italy": 380, "Japan": 392, "Korea": 410,
    "Luxembourg": 442, "Mexico": 484, "Netherlands": 528, "New Zealand": 554,
    "Norway": 578, "Poland": 616, "Portugal": 620, "Slovakia": 703,
    "UK": 826, "USA": 840, "Argentina": 32, "Brazil": 76, "Chile": 152,
    "China": 156, "Colombia": 170, "India": 356, "Indonesia": 360,
    "Israel": 376, "Saudi Arabia": 682, "South Africa": 710,
    "Bulgaria": 100, "Romania": 642,
}

EC_FEATURES = ["Year", "GDP", "Population", "Country Code"]
TRAIN_YEAR_MIN = 1970
TRAIN_YEAR_MAX = 2019
TEST_YEAR_MIN = 2019  # Include 2019 so plot connects with training (1970-2019)
TEST_YEAR_MAX = 2100

# Map ARIMA CSV country column names to COUNTRY_TO_CODE keys
ARIMA_COUNTRY_TO_STANDARD = {
    "Korea, Republic of": "Korea",
    "United Kingdom of Great Britain and Northern Ireland": "UK",
    "United States of America": "USA",
}


# ---------------------------------------------------------------------------
# Load EC model
# ---------------------------------------------------------------------------

def load_ec_model():
    if not EC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"EC model not found at {EC_MODEL_PATH}. Run random_forest_ec.py first."
        )
    return joblib.load(EC_MODEL_PATH)


def predict_ec_per_country(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict EC for each (Year, Country) row; add column Predicted_EC."""
    X = df[EC_FEATURES].copy()
    X["Country Code"] = X["Country Code"].astype("category")
    out = df.copy()
    out["Predicted_EC"] = model.predict(X)
    return out


# ---------------------------------------------------------------------------
# Data loading (same as random_forest_ec_global_ratio.py)
# ---------------------------------------------------------------------------

def load_global_ec() -> pd.DataFrame:
    path = DATA_DIR / "ARIMA input data - Global EC.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"global_EC_EJ": "global_EC"})
    df = df.dropna(subset=["Year", "global_EC"])
    df["Year"] = df["Year"].astype(int)
    df = df.sort_values("Year").groupby("Year", as_index=False).last()
    return df


def load_global_gdp_historical() -> pd.DataFrame:
    path = DATA_DIR / "ARIMA input data - GDP.csv"
    df = pd.read_csv(path)
    if "Year" not in df.columns or "Global" not in df.columns:
        return pd.DataFrame(columns=["Year", "GDP_global"])
    out = df[["Year", "Global"]].rename(columns={"Global": "GDP_global"})
    out = out.dropna(subset=["Year", "GDP_global"])
    out["Year"] = out["Year"].astype(int)
    return out


def load_gdp_arima() -> pd.DataFrame:
    path = DATA_DIR / "ARIMA input data - GDP.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df


def load_pop_arima() -> pd.DataFrame:
    path = DATA_DIR / "ARIMA input data - Population.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["Year"] = df["Year"].astype(int)
    return df


def load_global_pop_historical() -> pd.DataFrame:
    path = DATA_DIR / "ARIMA input data - Population.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Year" not in df.columns or "Global" not in df.columns:
        return pd.DataFrame(columns=["Year", "Population_global"])
    out = df[["Year", "Global"]].rename(columns={"Global": "Population_global"})
    out = out.dropna(subset=["Year", "Population_global"])
    out["Year"] = out["Year"].astype(int)
    return out


def wide_to_long(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    return df_wide.melt(id_vars=["Year"], var_name="Country", value_name=value_name)


def _normalize_arima_country(name: str) -> str:
    """Map ARIMA CSV country name to COUNTRY_TO_CODE key."""
    s = str(name).strip()
    return ARIMA_COUNTRY_TO_STANDARD.get(s, s)


# Actual EC per country for 1970--2019: local file or fetch from URL.
HISTORICAL_EC_PATH = DATA_DIR / "finaldata0411.csv"
HISTORICAL_EC_URL = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"


def load_historical_ec_only_42(year_min: int, year_max: int) -> pd.DataFrame:
    """Load actual EC only for 42 countries from finaldata0411 (local or URL). One row per (Year, Country Code). GDP and Population for all years come from ARIMA CSVs."""
    if HISTORICAL_EC_PATH.exists():
        df = pd.read_csv(HISTORICAL_EC_PATH)
    else:
        df = pd.read_csv(HISTORICAL_EC_URL)
        df.to_csv(HISTORICAL_EC_PATH, index=False)  # cache for next time
    df = df.dropna(subset=["Year", "Country Code", "EC"])
    df["Year"] = df["Year"].astype(int)
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)]
    df = df[df["Country Code"].isin(COUNTRY_TO_CODE.values())]
    return df[["Year", "Country Code", "EC"]]


def get_arima_gdp_pop_42(year_min: int, year_max: int) -> pd.DataFrame:
    """Load GDP and Population from ARIMA CSVs for 42 countries; one row per (Year, Country Code)."""
    gdp_df = load_gdp_arima()
    pop_df = load_pop_arima()
    gdp_df.columns = [c.strip() if isinstance(c, str) else c for c in gdp_df.columns]
    pop_df.columns = [c.strip() if isinstance(c, str) else c for c in pop_df.columns]
    gdp_df = gdp_df[(gdp_df["Year"] >= year_min) & (gdp_df["Year"] <= year_max)]
    pop_df = pop_df[(pop_df["Year"] >= year_min) & (pop_df["Year"] <= year_max)]

    gdp_long = wide_to_long(gdp_df.drop(columns=["Global"], errors="ignore"), "GDP")
    gdp_long["Country_std"] = gdp_long["Country"].apply(_normalize_arima_country)
    gdp_long["Country Code"] = gdp_long["Country_std"].map(COUNTRY_TO_CODE)
    gdp_long = gdp_long.dropna(subset=["Country Code"])[["Year", "Country Code", "GDP"]]

    pop_long = wide_to_long(pop_df.drop(columns=["Global"], errors="ignore"), "Population")
    pop_long["Country_std"] = pop_long["Country"].apply(_normalize_arima_country)
    pop_long["Country Code"] = pop_long["Country_std"].map(COUNTRY_TO_CODE)
    pop_long = pop_long.dropna(subset=["Country Code"])[["Year", "Country Code", "Population"]]

    merged = gdp_long.merge(pop_long, on=["Year", "Country Code"], how="inner")
    return merged


# ---------------------------------------------------------------------------
# Build training data: one row per (Year, Country) for 1970--2019
# ---------------------------------------------------------------------------

def build_training_data(ec_model) -> pd.DataFrame:
    """Per-country rows; EC from finaldata0411 only; GDP and Population from ARIMA CSVs (no ec_model for 1970--2019)."""
    historical_ec = load_historical_ec_only_42(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    arima_42 = get_arima_gdp_pop_42(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    arima_42 = arima_42.dropna(subset=["Year", "GDP", "Population"])
    finaldata = arima_42.merge(historical_ec, on=["Year", "Country Code"], how="inner")

    global_ec = load_global_ec()
    global_ec = global_ec[(global_ec["Year"] >= TRAIN_YEAR_MIN) & (global_ec["Year"] <= TRAIN_YEAR_MAX)]
    finaldata = finaldata.merge(global_ec[["Year", "global_EC"]], on="Year", how="left")

    global_gdp = load_global_gdp_historical()
    global_gdp = global_gdp[
        (global_gdp["Year"] >= TRAIN_YEAR_MIN) & (global_gdp["Year"] <= TRAIN_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    finaldata = finaldata.merge(global_gdp[["Year", "GDP_global"]], on="Year", how="left")

    global_pop = load_global_pop_historical()
    global_pop = global_pop[
        (global_pop["Year"] >= TRAIN_YEAR_MIN) & (global_pop["Year"] <= TRAIN_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    finaldata = finaldata.merge(global_pop[["Year", "Population_global"]], on="Year", how="left")

    finaldata["target_ratio"] = finaldata["EC"] / finaldata["global_EC"]
    finaldata["feat_EC_over_GDP"] = finaldata["EC"] / finaldata["GDP"]
    finaldata["feat_GDP_over_GDPglobal"] = finaldata["GDP"] / finaldata["GDP_global"]
    finaldata["feat_Pop_over_Popglobal"] = finaldata["Population"] / finaldata["Population_global"]
    finaldata = finaldata.dropna(subset=["target_ratio", "feat_EC_over_GDP", "feat_GDP_over_GDPglobal", "feat_Pop_over_Popglobal"])
    return finaldata


# ---------------------------------------------------------------------------
# Build test data: one row per (Year, Country) for 2020--2100; EC from model
# ---------------------------------------------------------------------------

def build_test_inputs(ec_model) -> pd.DataFrame:
    """Per-country rows for 2020--2100. All data from ARIMA CSVs; EC from ec_model."""
    arima_42 = get_arima_gdp_pop_42(TEST_YEAR_MIN, TEST_YEAR_MAX)
    arima_42 = arima_42.dropna(subset=["Year", "GDP", "Population"])
    test_per_country = predict_ec_per_country(ec_model, arima_42[EC_FEATURES])
    test_per_country = test_per_country.rename(columns={"Predicted_EC": "EC"})

    global_ec = load_global_ec()
    global_ec = global_ec[(global_ec["Year"] >= TEST_YEAR_MIN) & (global_ec["Year"] <= TEST_YEAR_MAX)]
    test_per_country = test_per_country.merge(global_ec[["Year", "global_EC"]], on="Year", how="left")

    global_gdp = load_global_gdp_historical()
    global_gdp = global_gdp[
        (global_gdp["Year"] >= TEST_YEAR_MIN) & (global_gdp["Year"] <= TEST_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    test_per_country = test_per_country.merge(global_gdp[["Year", "GDP_global"]], on="Year", how="left")

    global_pop = load_global_pop_historical()
    global_pop = global_pop[
        (global_pop["Year"] >= TEST_YEAR_MIN) & (global_pop["Year"] <= TEST_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    test_per_country = test_per_country.merge(global_pop[["Year", "Population_global"]], on="Year", how="left")

    test_per_country["feat_EC_over_GDP"] = test_per_country["EC"] / test_per_country["GDP"]
    test_per_country["feat_GDP_over_GDPglobal"] = test_per_country["GDP"] / test_per_country["GDP_global"]
    test_per_country["feat_Pop_over_Popglobal"] = test_per_country["Population"] / test_per_country["Population_global"]
    test_per_country = test_per_country.dropna(subset=[
        "feat_EC_over_GDP", "feat_GDP_over_GDPglobal", "feat_Pop_over_Popglobal"
    ])
    return test_per_country


# ---------------------------------------------------------------------------
# Train ratio model, predict per country, sum by year, plot
# ---------------------------------------------------------------------------

def main():
    print("Loading EC model ...")
    ec_model = load_ec_model()

    print("Building training data (1970--2019, per country) ...")
    train_df = build_training_data(ec_model)
    if train_df.empty or len(train_df) < 50:
        raise ValueError("Insufficient training data.")

    feature_cols = ["feat_EC_over_GDP", "feat_GDP_over_GDPglobal", "feat_Pop_over_Popglobal"]
    X = train_df[feature_cols]
    y = train_df["target_ratio"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ratio_model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    ratio_model.fit(X_train, y_train)

    y_val_pred = ratio_model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    print("Validation R² (ratio model, per country):", round(r2, 4))

    # SHAP summary plot: feature importance and impact on model output
    print("Computing SHAP values for feature importance ...")
    explainer = shap.TreeExplainer(ratio_model, X_train)
    shap_values = explainer.shap_values(X_train)
    feature_display_names = [
        "EC / GDP",
        "GDP / GDP_global",
        "Population / Pop_global",
    ]
    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        shap_values,
        X_train,
        feature_names=feature_display_names,
        show=False,
    )
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="grey", alpha=0.7)
    ax.xaxis.grid(True, linestyle="--", color="grey", alpha=0.7)
    ax.set_xlabel("SHAP value (impact on model output)")
    plt.tight_layout()
    shap_plot_path = PLOTS_DIR / "shap_ec_global_ratio_ind_countries.png"
    plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved SHAP plot to", shap_plot_path)

    print("Building test inputs (2020--2100, per country) ...")
    test_per_country = build_test_inputs(ec_model)
    X_test = test_per_country[feature_cols]
    test_per_country["predicted_ratio"] = ratio_model.predict(X_test)

    # Sum predicted ratio over 42 countries per year -> one value per year (like EC_42/global_EC)
    test_by_year = (
        test_per_country.groupby("Year", as_index=False)["predicted_ratio"]
        .sum()
        .rename(columns={"predicted_ratio": "predicted_ratio_sum"})
    )

    # Training actual ratio summed by year (EC_42/global_EC)
    train_by_year = (
        train_df.groupby("Year", as_index=False)["target_ratio"]
        .sum()
        .rename(columns={"target_ratio": "actual_ratio_sum"})
    )

    # Save outputs
    test_per_country.to_csv(OUTPUT_DATA_DIR / "test_per_country_ratio_ind.csv", index=False)
    test_by_year.to_csv(OUTPUT_DATA_DIR / "test_ratio_sum_by_year_ind.csv", index=False)
    train_by_year.to_csv(OUTPUT_DATA_DIR / "train_ratio_sum_by_year_ind.csv", index=False)
    joblib.dump(ratio_model, DATA_DIR / "ec_global_ratio_ind_model.joblib")
    print("Saved outputs to", OUTPUT_DATA_DIR)

    # Plot: year vs ratio (blue 1970–2019 actual sum, orange 2019–2100 predicted sum); dashed = + AI
    test_plot = test_by_year.copy()
    train_2019 = train_by_year.loc[train_by_year["Year"] == 2019, "actual_ratio_sum"]
    if not train_2019.empty:
        test_plot.loc[test_plot["Year"] == 2019, "predicted_ratio_sum"] = train_2019.values[0]
    # AI effect: ratio = (total EC 42 countries + AI energy) / global_EC for 2027–2100
    AI_DASHED_YEAR_MIN = 2027
    AI_DASHED_YEAR_MAX = 2100
    test_agg = (
        test_per_country.groupby("Year")
        .agg(EC_42=("EC", "sum"), global_EC=("global_EC", "first"))
        .reset_index()
    )
    test_agg = test_agg.merge(
        test_by_year[["Year", "predicted_ratio_sum"]], on="Year", how="left"
    )
    # ARIMA Global EC ends at 2024; for 2025+ use implied global_EC = EC_42 / predicted_ratio_sum
    test_agg["global_EC_filled"] = test_agg["global_EC"].fillna(
        test_agg["EC_42"] / test_agg["predicted_ratio_sum"]
    )
    ai_energy = get_ai_energy_for_years(test_agg["Year"].values)
    _, ai_lower, ai_upper = get_ai_energy_uncertainty_for_years(test_agg["Year"].values)
    ratio_with_ai = (test_agg["EC_42"].values + ai_energy) / test_agg["global_EC_filled"].values
    ratio_with_ai_lower = (test_agg["EC_42"].values + ai_lower) / test_agg["global_EC_filled"].values
    ratio_with_ai_upper = (test_agg["EC_42"].values + ai_upper) / test_agg["global_EC_filled"].values
    mask_ai = (test_agg["Year"] >= AI_DASHED_YEAR_MIN) & (test_agg["Year"] <= AI_DASHED_YEAR_MAX)
    years_ai = test_agg.loc[mask_ai, "Year"].values
    ratio_ai_plot = ratio_with_ai[mask_ai.to_numpy()]
    ratio_ai_lower = ratio_with_ai_lower[mask_ai.to_numpy()]
    ratio_ai_upper = ratio_with_ai_upper[mask_ai.to_numpy()]
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_by_year["Year"],
        train_by_year["actual_ratio_sum"],
        color="blue",
        linewidth=2,
        label="Training (1970–2019)",
    )
    pred_color = "tab:green"
    plt.plot(
        test_plot["Year"],
        test_plot["predicted_ratio_sum"],
        color=pred_color,
        linewidth=2,
        label="Predictions (2019–2100)",
    )
    plt.fill_between(
        test_plot["Year"],
        test_plot["predicted_ratio_sum"] - 0.02,
        test_plot["predicted_ratio_sum"] + 0.02,
        color=pred_color,
        alpha=0.3,
    )
    plt.fill_between(
        years_ai,
        ratio_ai_lower,
        ratio_ai_upper,
        color="tab:purple",
        alpha=0.3,
    )
    plt.plot(
        years_ai,
        ratio_ai_plot,
        color="tab:purple",
        linewidth=2,
        linestyle="--",
        label="Predictions + AI energy (2027–2100)",
    )
    plt.xlabel("Year")
    plt.ylabel("Ratio (EC_42 / Global EC)")
    plt.title("Random Forest EC_42/Global_EC Predictions (per-country model, summed)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(1965, 2105)
    plot_path = PLOTS_DIR / "ec_global_ratio_predictions_ind_countries.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot to", plot_path)


if __name__ == "__main__":
    main()

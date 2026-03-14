"""
Train a Random Forest to predict the ratio (per country):
  energy consumption of 1 country / global energy consumption.

Features (per country, per year):
  1. EC_country / GDP_country  (actual 1970--2019, EC model 2020--2100)
  2. GDP_country / GDP_global
  3. Population_country / Population_global

- Years 1970--2019: EC from finaldata0411.csv (actual). Train on per-country rows.
- Years 2020--2100: EC from ec_model per country. Predict ratio per country; sum
  ratios over 42 countries per year for the final plot (same as EC_42/global_EC).
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent
EC_MODEL_PATH = DATA_DIR / "ec_model.joblib"
PLOTS_DIR = DATA_DIR / "plots"
OUTPUT_DATA_DIR = DATA_DIR / "data"
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
TEST_YEAR_MIN = 2020
TEST_YEAR_MAX = 2100


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


# ---------------------------------------------------------------------------
# Build training data: one row per (Year, Country) for 1970--2019
# ---------------------------------------------------------------------------

def build_training_data() -> pd.DataFrame:
    """Per-country rows; EC from actual data (finaldata0411.csv). No EC model."""
    url_final = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"
    finaldata = pd.read_csv(url_final)
    finaldata = finaldata[
        (finaldata["Year"] >= TRAIN_YEAR_MIN) & (finaldata["Year"] <= TRAIN_YEAR_MAX)
    ].copy()
    finaldata = finaldata.dropna(subset=["Year", "EC", "GDP", "Population"])
    country_codes_42 = set(COUNTRY_TO_CODE.values())
    finaldata = finaldata[finaldata["Country Code"].isin(country_codes_42)]

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
    """Per-country rows for 2020--2100. EC from ec_model. No target."""
    url_ssp = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs126.csv"
    gdp_arima = load_gdp_arima()
    pop_arima = load_pop_arima()
    pop_arima = pop_arima.rename(columns={c: c.strip() for c in pop_arima.columns})

    # 2020--2060: inputs126
    ssp = pd.read_csv(url_ssp)
    ssp = ssp[(ssp["Year"] >= TEST_YEAR_MIN) & (ssp["Year"] <= 2060)].copy()
    ssp = ssp.dropna(subset=["Year", "GDP", "Population"])
    ssp = ssp[ssp["Country Code"].isin(COUNTRY_TO_CODE.values())]
    test_2020_2060 = predict_ec_per_country(ec_model, ssp)

    # 2061--2100: ARIMA wide -> long
    gdp_future = gdp_arima[gdp_arima["Year"] >= 2061].copy()
    pop_future = pop_arima[pop_arima["Year"] >= 2061].copy()
    country_cols = [c for c in gdp_future.columns if c not in ("Year", "Global")]
    gdp_long = wide_to_long(gdp_future[["Year"] + country_cols], "GDP")
    pop_cols = [c for c in pop_future.columns if c not in ("Year", "Global")]
    pop_long = wide_to_long(pop_future[["Year"] + pop_cols], "Population")
    future = gdp_long.merge(pop_long, on=["Year", "Country"])
    future["Country Code"] = future["Country"].map(COUNTRY_TO_CODE)
    future = future.dropna(subset=["Country Code"])
    future = future[EC_FEATURES]
    test_2061_2100 = predict_ec_per_country(ec_model, future)

    # Rename for consistent columns (EC or Predicted_EC)
    test_2020_2060 = test_2020_2060.rename(columns={"Predicted_EC": "EC"})
    test_2061_2100 = test_2061_2100.rename(columns={"Predicted_EC": "EC"})
    # 2020-2060 from ssp, 2061-2100 from ARIMA (no overlap)
    test_per_country = pd.concat([test_2020_2060, test_2061_2100], ignore_index=True)

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
    train_df = build_training_data()
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

    # Plot: year vs ratio (blue 1970--2019 actual sum, orange 2020--2100 predicted sum)
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_by_year["Year"],
        train_by_year["actual_ratio_sum"],
        color="blue",
        linewidth=2,
        label="Training (1970–2019)",
    )
    plt.plot(
        test_by_year["Year"],
        test_by_year["predicted_ratio_sum"],
        color="orange",
        linewidth=2,
        label="Predictions (2020–2100)",
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

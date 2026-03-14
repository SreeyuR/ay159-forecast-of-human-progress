"""
Train a Random Forest to predict the ratio:
  (total energy consumption of 42 countries) / (global energy consumption).

Features:
  1. EC_42 / GDP_42  (EC_42: actual data 1970--2019, EC model 2020--2100)
  2. GDP_42 / GDP_global
  3. Population_42 / Population_global

- Years 1970--2019: EC per country from finaldata0411.csv (actual data); sum for
  total EC of 42 countries. No EC model used.
- Years 2020--2100: EC per country from ec_model prediction; then sum for EC_42.

Training: 1970--2019 (target = actual ratio from data).
Test/predict: 2020--2100 (target unknown; we predict the ratio).
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
OUTPUT_DATA_DIR = DATA_DIR / "data"  # CSVs for training/test inputs, features, and ratios
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 42 countries: same mapping as in random_forest_ec.py
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
# Load EC model (from random_forest_ec.py)
# ---------------------------------------------------------------------------

def load_ec_model():
    if not EC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"EC model not found at {EC_MODEL_PATH}. "
            "Run random_forest_ec.py first to train and save the model."
        )
    return joblib.load(EC_MODEL_PATH)


def predict_ec_42(model, df: pd.DataFrame) -> pd.Series:
    """Predict per-country EC and return yearly sum for the 42 countries."""
    X = df[EC_FEATURES].copy()
    X["Country Code"] = X["Country Code"].astype("category")
    df = df.copy()
    df["Predicted_EC"] = model.predict(X)
    return df.groupby("Year", as_index=False)["Predicted_EC"].sum()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_global_ec() -> pd.DataFrame:
    """Global EC (EJ) from ARIMA file. One value per year (drop duplicates)."""
    path = DATA_DIR / "ARIMA input data - Global EC.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"global_EC_EJ": "global_EC"})
    df = df.dropna(subset=["Year", "global_EC"])
    df["Year"] = df["Year"].astype(int)
    # Multiple rows per year for early years; take the main series (last occurrence per year)
    df = df.sort_values("Year").groupby("Year", as_index=False).last()
    return df


def load_global_gdp_historical() -> pd.DataFrame:
    """
    Global GDP (current US$) for 1970--2100 from ARIMA input data - GDP.csv
    using the "Global" column. No other file or API used.
    """
    path = DATA_DIR / "ARIMA input data - GDP.csv"
    df = pd.read_csv(path)
    if "Year" not in df.columns or "Global" not in df.columns:
        return pd.DataFrame(columns=["Year", "GDP_global"])
    out = df[["Year", "Global"]].rename(columns={"Global": "GDP_global"})
    out = out.dropna(subset=["Year", "GDP_global"])
    out["Year"] = out["Year"].astype(int)
    return out


def load_gdp_arima() -> pd.DataFrame:
    """GDP by country and Global for 2060--2100 (wide then long for 42 + Global)."""
    path = DATA_DIR / "ARIMA input data - GDP.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)
    return df


def load_pop_arima() -> pd.DataFrame:
    """Population by country and Global for 2060--2100."""
    path = DATA_DIR / "ARIMA input data - Population.csv"
    df = pd.read_csv(path)
    # Fix column name if it has a space
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df["Year"] = df["Year"].astype(int)
    return df


def load_global_pop_historical() -> pd.DataFrame:
    """
    Global population for 1970--2100 from ARIMA input data - Population.csv
    using the "Global" column. No other file used.
    """
    path = DATA_DIR / "ARIMA input data - Population.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "Year" not in df.columns or "Global" not in df.columns:
        return pd.DataFrame(columns=["Year", "Population_global"])
    out = df[["Year", "Global"]].rename(columns={"Global": "Population_global"})
    out = out.dropna(subset=["Year", "Population_global"])
    out["Year"] = out["Year"].astype(int)
    return out


def wide_to_long_gdp(df_wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Melt GDP CSV: Year + country columns -> Year, Country, value."""
    id_vars = ["Year"]
    var_name = "Country"
    return df_wide.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)


# ---------------------------------------------------------------------------
# Build training data (1970--2019)
# ---------------------------------------------------------------------------

def build_training_data(ec_model) -> pd.DataFrame:
    # 1970--2019: use actual EC from finaldata0411.csv (do not use ec_model)
    url_final = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"
    finaldata = pd.read_csv(url_final)
    finaldata = finaldata[
        (finaldata["Year"] >= TRAIN_YEAR_MIN) & (finaldata["Year"] <= TRAIN_YEAR_MAX)
    ].copy()
    finaldata = finaldata.dropna(subset=["Year", "EC", "GDP", "Population"])
    # Restrict to the 42 countries; sum actual EC (not model) for total EC_42
    country_codes_42 = set(COUNTRY_TO_CODE.values())
    finaldata = finaldata[finaldata["Country Code"].isin(country_codes_42)]

    # Yearly aggregates: EC_42 = sum of actual energy consumption from CSV
    by_year = finaldata.groupby("Year").agg(
        EC_42=("EC", "sum"),
        GDP_42=("GDP", "sum"),
        Population_42=("Population", "sum"),
    ).reset_index()

    # Global EC
    global_ec = load_global_ec()
    global_ec = global_ec[(global_ec["Year"] >= TRAIN_YEAR_MIN) & (global_ec["Year"] <= TRAIN_YEAR_MAX)]
    by_year = by_year.merge(global_ec[["Year", "global_EC"]], on="Year", how="left")

    # Global GDP from ARIMA input data - GDP.csv (Global column)
    global_gdp = load_global_gdp_historical()
    global_gdp = global_gdp[
        (global_gdp["Year"] >= TRAIN_YEAR_MIN) & (global_gdp["Year"] <= TRAIN_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    by_year = by_year.merge(global_gdp[["Year", "GDP_global"]], on="Year", how="left")

    # Global population from ARIMA input data - Population.csv (Global column)
    global_pop = load_global_pop_historical()
    global_pop = global_pop[
        (global_pop["Year"] >= TRAIN_YEAR_MIN) & (global_pop["Year"] <= TRAIN_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    by_year = by_year.merge(global_pop[["Year", "Population_global"]], on="Year", how="left")

    # Target: EC_42 / global_EC
    by_year["target_ratio"] = by_year["EC_42"] / by_year["global_EC"]

    # Features
    by_year["feat_EC42_over_GDP42"] = by_year["EC_42"] / by_year["GDP_42"]
    by_year["feat_GDP42_over_GDPglobal"] = by_year["GDP_42"] / by_year["GDP_global"]
    by_year["feat_Pop42_over_Popglobal"] = by_year["Population_42"] / by_year["Population_global"]

    # Drop rows with NaN (missing global EC, global GDP, or global population)
    by_year = by_year.dropna()
    return by_year


# ---------------------------------------------------------------------------
# Build test input data (2020--2100): features only
# ---------------------------------------------------------------------------

def build_test_inputs(ec_model) -> pd.DataFrame:
    """Build feature rows for 2020--2100. EC_42 from ec_model (no actual data). No target (we predict it)."""
    url_ssp = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs126.csv"
    gdp_arima = load_gdp_arima()
    pop_arima = load_pop_arima()
    pop_arima = pop_arima.rename(columns={c: c.strip() for c in pop_arima.columns})

    # ----- 2020--2060: from inputs126 -----
    ssp = pd.read_csv(url_ssp)
    ssp = ssp[(ssp["Year"] >= TEST_YEAR_MIN) & (ssp["Year"] <= 2060)].copy()
    ssp = ssp.dropna(subset=["Year", "GDP", "Population"])

    ec_sum_ssp = predict_ec_42(ec_model, ssp)
    gdp_42_ssp = ssp.groupby("Year", as_index=False)["GDP"].sum().rename(columns={"GDP": "GDP_42"})
    pop_42_ssp = ssp.groupby("Year", as_index=False)["Population"].sum().rename(columns={"Population": "Population_42"})
    merge_2020_2060 = ec_sum_ssp.merge(gdp_42_ssp, on="Year").merge(pop_42_ssp, on="Year")

    # ----- 2061--2100: from ARIMA GDP/Pop (wide) -----
    gdp_future = gdp_arima[gdp_arima["Year"] >= 2061].copy()
    pop_future = pop_arima[pop_arima["Year"] >= 2061].copy()

    # Exclude "Global" when building per-country table for EC model
    country_cols = [c for c in gdp_future.columns if c != "Year" and c != "Global"]
    gdp_long = wide_to_long_gdp(gdp_future[["Year"] + country_cols], "GDP")
    pop_country_cols = [c for c in pop_future.columns if c != "Year" and c != "Global"]
    pop_long = wide_to_long_gdp(pop_future[["Year"] + pop_country_cols], "Population")
    future = gdp_long.merge(pop_long, on=["Year", "Country"])
    future["Country Code"] = future["Country"].map(COUNTRY_TO_CODE)
    future = future.dropna(subset=["Country Code"])
    future = future[EC_FEATURES]

    ec_sum_future = predict_ec_42(ec_model, future)
    gdp_42_future = gdp_long.groupby("Year", as_index=False)["GDP"].sum().rename(columns={"GDP": "GDP_42"})
    pop_42_future = pop_long.groupby("Year", as_index=False)["Population"].sum().rename(columns={"Population": "Population_42"})
    merge_2061_2100 = (
        ec_sum_future.merge(gdp_42_future, on="Year").merge(pop_42_future, on="Year")
    )

    # Combine 2020--2060 and 2061--2100
    test_years = (
        pd.concat([merge_2020_2060, merge_2061_2100])
        .drop_duplicates(subset=["Year"])
        .sort_values("Year")
    )

    # Global GDP from ARIMA input data - GDP.csv (Global column) for all test years
    global_gdp = load_global_gdp_historical()
    gdp_global_test = global_gdp[
        (global_gdp["Year"] >= TEST_YEAR_MIN) & (global_gdp["Year"] <= TEST_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    test_years = test_years.merge(gdp_global_test, on="Year", how="left")

    # Global population from ARIMA input data - Population.csv (Global column) for all test years
    global_pop = load_global_pop_historical()
    pop_global_test = global_pop[
        (global_pop["Year"] >= TEST_YEAR_MIN) & (global_pop["Year"] <= TEST_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    test_years = test_years.merge(pop_global_test, on="Year", how="left")

    test_years["feat_EC42_over_GDP42"] = test_years["Predicted_EC"] / test_years["GDP_42"]
    test_years["feat_GDP42_over_GDPglobal"] = test_years["GDP_42"] / test_years["GDP_global"]
    test_years["feat_Pop42_over_Popglobal"] = test_years["Population_42"] / test_years["Population_global"]
    test_years = test_years.dropna(subset=[
        "feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal"
    ])
    return test_years


# ---------------------------------------------------------------------------
# Train ratio model and predict
# ---------------------------------------------------------------------------

def main():
    print("Loading EC model from random_forest_ec.py ...")
    ec_model = load_ec_model()

    print("Building training data (1970--2019) ...")
    train_df = build_training_data(ec_model)
    if train_df.empty or len(train_df) < 10:
        raise ValueError("Insufficient training data. Check global EC and global GDP sources.")

    # Save training data to data/
    train_input_cols = ["Year", "EC_42", "GDP_42", "Population_42", "global_EC", "GDP_global", "Population_global"]
    train_df[train_input_cols].to_csv(OUTPUT_DATA_DIR / "train_inputs.csv", index=False)
    train_feature_cols = ["Year", "feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal"]
    train_df[train_feature_cols].to_csv(OUTPUT_DATA_DIR / "train_input_features.csv", index=False)
    train_df[["Year", "target_ratio"]].to_csv(OUTPUT_DATA_DIR / "train_actual_ratio.csv", index=False)
    train_df.to_csv(OUTPUT_DATA_DIR / "train_full.csv", index=False)
    print("Saved training CSVs to", OUTPUT_DATA_DIR)

    feature_cols = ["feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal"]
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
    print("Validation R² (ratio model):", round(r2, 4))

    # Predict for test years 2020--2100
    print("Building test inputs (2020--2100) ...")
    test_df = build_test_inputs(ec_model)
    X_test = test_df[feature_cols]
    test_df["predicted_ratio"] = ratio_model.predict(X_test)

    # Save test data to data/
    test_input_cols = ["Year", "Predicted_EC", "GDP_42", "Population_42", "GDP_global", "Population_global"]
    test_df[[c for c in test_input_cols if c in test_df.columns]].to_csv(
        OUTPUT_DATA_DIR / "test_inputs.csv", index=False
    )
    test_feature_cols = ["Year", "feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal"]
    test_df[test_feature_cols].to_csv(OUTPUT_DATA_DIR / "test_input_features.csv", index=False)
    test_df[["Year", "predicted_ratio"]].to_csv(OUTPUT_DATA_DIR / "test_predicted_ratio.csv", index=False)
    test_full_cols = ["Year", "Predicted_EC", "GDP_42", "Population_42", "GDP_global", "Population_global",
                      "feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal",
                      "predicted_ratio"]
    test_df[[c for c in test_full_cols if c in test_df.columns]].to_csv(
        OUTPUT_DATA_DIR / "test_full.csv", index=False
    )
    print("Saved test CSVs to", OUTPUT_DATA_DIR)

    # Also save legacy single predictions file in project root
    out_cols = ["Year", "predicted_ratio", "Predicted_EC", "GDP_42", "GDP_global",
                "Population_42", "Population_global",
                "feat_EC42_over_GDP42", "feat_GDP42_over_GDPglobal", "feat_Pop42_over_Popglobal"]
    out_cols = [c for c in out_cols if c in test_df.columns]
    test_df[out_cols].to_csv(DATA_DIR / "ec_global_ratio_predictions.csv", index=False)
    print("Saved ec_global_ratio_predictions.csv to project root")

    # Optional: save ratio model
    joblib.dump(ratio_model, DATA_DIR / "ec_global_ratio_model.joblib")
    print("Saved ratio model to ec_global_ratio_model.joblib")

    # Plot: year vs ratio — blue 1970–2020 (training), orange 2020–2100 (predictions)
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_df["Year"],
        train_df["target_ratio"],
        color="blue",
        linewidth=2,
        label="Training (1970–2020)",
    )
    plt.plot(
        test_df["Year"],
        test_df["predicted_ratio"],
        color="orange",
        linewidth=2,
        label="Predictions (2020–2100)",
    )
    plt.xlabel("Year")
    plt.ylabel("Ratio (EC_42 / Global EC)")
    plt.title("Random Forest EC_42/Global_EC Predictions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(1965, 2105)
    plot_path = PLOTS_DIR / "ec_global_ratio_predictions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot to", plot_path)


if __name__ == "__main__":
    main()

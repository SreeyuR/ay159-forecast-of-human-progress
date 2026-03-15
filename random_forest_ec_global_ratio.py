"""
Train a Random Forest to predict the ratio:
  (total energy consumption of 42 countries) / (global energy consumption).

Features:
  1. EC_42 / GDP_42
  2. GDP_42 / GDP_global
  3. Population_42 / Population_global

Training (1970--2019): actual EC from finaldata0411 only; GDP and Population from ARIMA CSVs; no ec_model.
Test/predict (2020--2100): GDP/Pop from ARIMA; EC_42 from ec_model.

Data: finaldata0411 = EC per country 1970--2019 only. All GDP and Population from ARIMA; global_EC from ARIMA Global EC.
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
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
OUTPUT_DATA_DIR = DATA_DIR # CSVs for training/test inputs, features, and ratios
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
TEST_YEAR_MIN = 2019  # Include 2019 so plot connects with training (1970-2019)
TEST_YEAR_MAX = 2100

# Map ARIMA CSV country column names to COUNTRY_TO_CODE keys (GDP uses long names)
ARIMA_COUNTRY_TO_STANDARD = {
    "Korea, Republic of": "Korea",
    "United Kingdom of Great Britain and Northern Ireland": "UK",
    "United States of America": "USA",
}


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
    country_codes_42 = set(COUNTRY_TO_CODE.values())
    df = df[df["Country Code"].isin(country_codes_42)]
    return df[["Year", "Country Code", "EC"]]


def get_arima_gdp_pop_42(year_min: int, year_max: int) -> pd.DataFrame:
    """Load GDP and Population from ARIMA CSVs for 42 countries in [year_min, year_max]; one row per (Year, Country Code)."""
    gdp_df = load_gdp_arima()
    pop_df = load_pop_arima()
    gdp_df.columns = [c.strip() if isinstance(c, str) else c for c in gdp_df.columns]
    pop_df.columns = [c.strip() if isinstance(c, str) else c for c in pop_df.columns]
    gdp_df = gdp_df[(gdp_df["Year"] >= year_min) & (gdp_df["Year"] <= year_max)]
    pop_df = pop_df[(pop_df["Year"] >= year_min) & (pop_df["Year"] <= year_max)]
    # Melt and restrict to 42 countries with consistent naming
    gdp_long = wide_to_long_gdp(gdp_df.drop(columns=["Global"], errors="ignore"), "GDP")
    gdp_long["Country_std"] = gdp_long["Country"].apply(_normalize_arima_country)
    gdp_long["Country Code"] = gdp_long["Country_std"].map(COUNTRY_TO_CODE)
    gdp_long = gdp_long.dropna(subset=["Country Code"])[["Year", "Country Code", "GDP"]]
    pop_long = wide_to_long_gdp(pop_df.drop(columns=["Global"], errors="ignore"), "Population")
    pop_long["Country_std"] = pop_long["Country"].apply(_normalize_arima_country)
    pop_long["Country Code"] = pop_long["Country_std"].map(COUNTRY_TO_CODE)
    pop_long = pop_long.dropna(subset=["Country Code"])[["Year", "Country Code", "Population"]]
    merged = gdp_long.merge(pop_long, on=["Year", "Country Code"], how="inner")
    return merged


# ---------------------------------------------------------------------------
# Build training data (1970--2019)
# ---------------------------------------------------------------------------

def build_training_data(ec_model) -> pd.DataFrame:
    # 1970--2019: EC from finaldata0411 only; GDP and Population from ARIMA CSVs (no model for training).
    historical_ec = load_historical_ec_only_42(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    arima_42 = get_arima_gdp_pop_42(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
    arima_42 = arima_42.dropna(subset=["Year", "GDP", "Population"])
    merged = arima_42.merge(historical_ec, on=["Year", "Country Code"], how="inner")
    by_year = merged.groupby("Year", as_index=False).agg(
        EC_42=("EC", "sum"),
        GDP_42=("GDP", "sum"),
        Population_42=("Population", "sum"),
    )

    # Global EC from ARIMA input data - Global EC.csv
    global_ec = load_global_ec()
    global_ec = global_ec[(global_ec["Year"] >= TRAIN_YEAR_MIN) & (global_ec["Year"] <= TRAIN_YEAR_MAX)]
    by_year = by_year.merge(global_ec[["Year", "global_EC"]], on="Year", how="left")

    # Global GDP and Population from ARIMA input data
    global_gdp = load_global_gdp_historical()
    global_gdp = global_gdp[
        (global_gdp["Year"] >= TRAIN_YEAR_MIN) & (global_gdp["Year"] <= TRAIN_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    by_year = by_year.merge(global_gdp[["Year", "GDP_global"]], on="Year", how="left")

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

    by_year = by_year.dropna()
    return by_year


# ---------------------------------------------------------------------------
# Build test input data (2020--2100): features only
# ---------------------------------------------------------------------------

def build_test_inputs(ec_model) -> pd.DataFrame:
    """Build feature rows for 2020--2100. All data from ARIMA CSVs; EC_42 from ec_model."""
    # All test years from ARIMA GDP and Population (all countries, all years in CSVs)
    arima_42 = get_arima_gdp_pop_42(TEST_YEAR_MIN, TEST_YEAR_MAX)
    arima_42 = arima_42.dropna(subset=["Year", "GDP", "Population"])

    ec_sum = predict_ec_42(ec_model, arima_42[EC_FEATURES])
    by_year = arima_42.groupby("Year", as_index=False).agg(
        GDP_42=("GDP", "sum"),
        Population_42=("Population", "sum"),
    )
    test_years = by_year.merge(ec_sum, on="Year", how="left")

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

    # Plot: year vs ratio — blue 1970–2019 (training), orange 2019–2100 (predictions); dashed = + AI
    test_plot = test_df.copy()
    train_2019 = train_df.loc[train_df["Year"] == 2019, "target_ratio"]
    if not train_2019.empty:
        test_plot.loc[test_plot["Year"] == 2019, "predicted_ratio"] = train_2019.values[0]
    # AI effect: ratio = (total EC 42 countries + AI energy) / global_EC for 2027–2100
    AI_DASHED_YEAR_MIN = 2027
    AI_DASHED_YEAR_MAX = 2100
    global_ec = load_global_ec()
    global_ec_test = global_ec[
        (global_ec["Year"] >= TEST_YEAR_MIN) & (global_ec["Year"] <= TEST_YEAR_MAX)
    ].drop_duplicates(subset=["Year"])
    test_with_global = test_df.merge(global_ec_test[["Year", "global_EC"]], on="Year", how="left")
    # ARIMA Global EC CSV ends at 2024; for 2025+ use implied global_EC = Predicted_EC / predicted_ratio
    test_with_global["global_EC_filled"] = test_with_global["global_EC"].fillna(
        test_with_global["Predicted_EC"] / test_with_global["predicted_ratio"]
    )
    ai_energy = get_ai_energy_for_years(test_with_global["Year"].values)
    _, ai_lower, ai_upper = get_ai_energy_uncertainty_for_years(test_with_global["Year"].values)
    # ratio = (total EC 42 countries + AI energy) / global_EC for 2020–2100
    ratio_with_ai = (
        (test_with_global["Predicted_EC"].values + ai_energy)
        / test_with_global["global_EC_filled"].values
    )
    ratio_with_ai_lower = (
        (test_with_global["Predicted_EC"].values + ai_lower)
        / test_with_global["global_EC_filled"].values
    )
    ratio_with_ai_upper = (
        (test_with_global["Predicted_EC"].values + ai_upper)
        / test_with_global["global_EC_filled"].values
    )
    mask_ai = (test_with_global["Year"] >= AI_DASHED_YEAR_MIN) & (test_with_global["Year"] <= AI_DASHED_YEAR_MAX)
    years_ai = test_with_global.loc[mask_ai, "Year"].values
    ratio_ai_plot = ratio_with_ai[mask_ai.to_numpy()]
    ratio_ai_lower = ratio_with_ai_lower[mask_ai.to_numpy()]
    ratio_ai_upper = ratio_with_ai_upper[mask_ai.to_numpy()]
    plt.figure(figsize=(10, 6))
    plt.plot(
        train_df["Year"],
        train_df["target_ratio"],
        color="blue",
        linewidth=2,
        label="Training (1970–2019)",
    )
    pred_color = "tab:green"
    plt.plot(
        test_plot["Year"],
        test_plot["predicted_ratio"],
        color=pred_color,
        linewidth=2,
        label="Predictions (2019–2100)",
    )
    plt.fill_between(
        test_plot["Year"],
        test_plot["predicted_ratio"] - 0.02,
        test_plot["predicted_ratio"] + 0.02,
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

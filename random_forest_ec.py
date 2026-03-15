import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path

from ai_energy import get_ai_energy_for_years, get_ai_energy_uncertainty_for_years

# ==========================================================
# 1. OUTPUT FOLDER
# ==========================================================

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# ==========================================================
# 2. LOAD DATA: EC FROM URL, GDP/POPULATION FROM ARIMA FILES
# ==========================================================

# Energy consumption for training (1970-2020, 42 countries)
url_final = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"
finaldata = pd.read_csv(url_final)

# GDP and Population for ALL years from local ARIMA files only
gdp_arima = pd.read_csv(f"{DATA_DIR}/ARIMA input data - GDP.csv")
pop_arima = pd.read_csv(f"{DATA_DIR}/ARIMA input data - Population.csv")

# ==========================================================
# 3. FEATURES
# ==========================================================

features = [
    "Year",
    "GDP",
    "Population",
    "Country Code"
]

target = "EC"

# ==========================================================
# 3b. COUNTRY CODE MAPPING + ARIMA GDP/POPULATION (ALL YEARS)
# ==========================================================

country_to_code = {
    "Australia": 36,
    "Austria": 40,
    "Belgium": 56,
    "Canada": 124,
    "Czechia": 203,
    "Denmark": 208,
    "Finland": 246,
    "France": 250,
    "Germany": 276,
    "Greece": 300,
    "Hungary": 348,
    "Iceland": 352,
    "Ireland": 372,
    "Italy": 380,
    "Japan": 392,
    "Korea": 410,
    "Luxembourg": 442,
    "Mexico": 484,
    "Netherlands": 528,
    "New Zealand": 554,
    "Norway": 578,
    "Poland": 616,
    "Portugal": 620,
    "Slovakia": 703,
    "UK": 826,
    "USA": 840,
    "Argentina": 32,
    "Brazil": 76,
    "Chile": 152,
    "China": 156,
    "Colombia": 170,
    "India": 356,
    "Indonesia": 360,
    "Israel": 376,
    "Saudi Arabia": 682,
    "South Africa": 710,
    "Bulgaria": 100,
    "Romania": 642
}

# Map ARIMA file column names to country_to_code keys (strip spaces, long names)
def _normalize_arima_columns(df, drop_global=True):
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    rename = {
        "Korea, Republic of": "Korea",
        "United Kingdom of Great Britain and Northern Ireland": "UK",
        "United States of America": "USA",
    }
    df = df.rename(columns=rename)
    if drop_global and "Global" in df.columns:
        df = df.drop(columns=["Global"])
    # Pandas may read duplicate "Israel" as "Israel.1"; map to "Israel" then keep one column
    if "Israel.1" in df.columns:
        df = df.drop(columns=["Israel.1"])
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df

gdp_arima = _normalize_arima_columns(gdp_arima)
pop_arima = _normalize_arima_columns(pop_arima)

def _wide_to_long(df, value_name):
    return df.melt(id_vars="Year", var_name="Country", value_name=value_name)

gdp_long = _wide_to_long(gdp_arima, "GDP")
pop_long = _wide_to_long(pop_arima, "Population")

arima_long = gdp_long.merge(pop_long, on=["Year", "Country"])
arima_long["Country Code"] = arima_long["Country"].map(country_to_code)
arima_long = arima_long.dropna(subset=["Country Code"])

# ==========================================================
# 4. TRAIN MODEL
# ==========================================================
# Use EC from finaldata, GDP and Population from ARIMA for all training years.
ec_from_final = finaldata[finaldata["Year"] < 2020][["Year", "Country Code", "EC"]].copy()
train_arima = arima_long[arima_long["Year"] < 2020][["Year", "Country Code", "GDP", "Population"]].copy()
train_data = ec_from_final.merge(train_arima, on=["Year", "Country Code"], how="inner")

X = train_data[features].copy()
y = train_data[target]

X["Country Code"] = X["Country Code"].astype("category")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = lgb.LGBMRegressor(
    boosting_type="gbdt",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Save model for use in random_forest_ec_global_ratio.py
MODEL_PATH = Path("ec_model.joblib")
joblib.dump(model, MODEL_PATH)
print("Saved EC model to", MODEL_PATH)

# ==========================================================
# 5. VALIDATION
# ==========================================================

y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)

print("Validation R²:", r2)

# ==========================================================
# 6. HISTORICAL GLOBAL ENERGY
# ==========================================================

historical_global = (
    finaldata.groupby("Year", as_index=False)["EC"]
    .sum()
)

# ==========================================================
# 7. FORECAST 2020–2100 USING ARIMA GDP/POPULATION ONLY
# ==========================================================

future_arima = arima_long[arima_long["Year"] >= 2020].copy()
X_future = future_arima[features].copy()
X_future["Country Code"] = X_future["Country Code"].astype("category")
future_arima["Predicted_EC"] = model.predict(X_future)

future_yearly = (
    future_arima.groupby("Year", as_index=False)["Predicted_EC"]
    .sum()
)

# Total EC + AI energy (2020–2100) for dashed line showing impact of AI
years_2020_2100 = future_yearly["Year"].values
ai_energy_ej = get_ai_energy_for_years(years_2020_2100)
future_yearly_plus_ai = future_yearly.copy()
future_yearly_plus_ai["EC_plus_AI"] = future_yearly_plus_ai["Predicted_EC"] + ai_energy_ej
_, ai_lower, ai_upper = get_ai_energy_uncertainty_for_years(years_2020_2100)
future_yearly_plus_ai["EC_plus_AI_lower"] = future_yearly_plus_ai["Predicted_EC"] + ai_lower
future_yearly_plus_ai["EC_plus_AI_upper"] = future_yearly_plus_ai["Predicted_EC"] + ai_upper
# Plot "Predictions + AI energy" only from 2027 onward
future_yearly_plus_ai_from_2027 = future_yearly_plus_ai[future_yearly_plus_ai["Year"] >= 2027].copy()

# Country-level output 2020–2100 for CSV; 2061–2100 subset for zoom plot and global CSV
future_proj = future_arima.copy()  # 2020–2100
future_2100 = (
    future_proj[future_proj["Year"] >= 2061]
    .groupby("Year", as_index=False)["Predicted_EC"]
    .sum()
)

# ==========================================================
# 12. FULL RANGE PLOT (1960–2100)
# ==========================================================
# Blue line ends at 2020 with the forecast value at 2020 so it connects to green
historical_global_plot = historical_global.copy()
pred_2020 = future_yearly[future_yearly["Year"] == 2020]
if not pred_2020.empty:
    predicted_total_2020 = pred_2020["Predicted_EC"].iloc[0]
    mask_2020 = historical_global_plot["Year"] == 2020
    if mask_2020.any():
        historical_global_plot.loc[mask_2020, "EC"] = predicted_total_2020
    else:
        historical_global_plot = pd.concat([
            historical_global_plot,
            pd.DataFrame({"Year": [2020], "EC": [predicted_total_2020]})
        ], ignore_index=True).sort_values("Year").reset_index(drop=True)
# Forecast plot: 2020–2100 only
future_yearly_plot = future_yearly

plt.figure(figsize=(12,6))

plt.plot(
    historical_global_plot["Year"],
    historical_global_plot["EC"],
    label="Historical data",
    linewidth=2,
    color="blue"
)

plt.plot(
    future_yearly_plot["Year"],
    future_yearly_plot["Predicted_EC"],
    label="Forecast (ARIMA GDP/Population)",
    color="tab:green",
    linewidth=2,
)
y_err = 0.02 * future_yearly_plot["Predicted_EC"].values
plt.fill_between(
    future_yearly_plot["Year"],
    future_yearly_plot["Predicted_EC"] - y_err,
    future_yearly_plot["Predicted_EC"] + y_err,
    color="tab:green",
    alpha=0.3,
)
plt.fill_between(
    future_yearly_plus_ai_from_2027["Year"],
    future_yearly_plus_ai_from_2027["EC_plus_AI_lower"],
    future_yearly_plus_ai_from_2027["EC_plus_AI_upper"],
    color="tab:orange",
    alpha=0.3,
)
plt.plot(
    future_yearly_plus_ai_from_2027["Year"],
    future_yearly_plus_ai_from_2027["EC_plus_AI"],
    label="Forecast + AI energy",
    color="tab:orange",
    linewidth=2,
    linestyle="--",
)

plt.xlim(1965, 2100)
plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (EJ)")
plt.title("1960 - 2100")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plot_path = PLOTS_DIR / "ssp_full_2100.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 13. ZOOM PLOT (2021–2060)
# ==========================================================

plt.figure(figsize=(10,6))

zoom_2060 = future_yearly[future_yearly["Year"].between(2020, 2060)]
zoom_2060_plus_ai = future_yearly_plus_ai_from_2027[future_yearly_plus_ai_from_2027["Year"].between(2027, 2060)]
plt.plot(
    zoom_2060["Year"],
    zoom_2060["Predicted_EC"],
    label="Forecast (ARIMA GDP/Population)",
    color="tab:green",
    linewidth=2,
)
y_err = 0.02 * zoom_2060["Predicted_EC"].values
plt.fill_between(
    zoom_2060["Year"],
    zoom_2060["Predicted_EC"] - y_err,
    zoom_2060["Predicted_EC"] + y_err,
    color="tab:green",
    alpha=0.3,
)
plt.fill_between(
    zoom_2060_plus_ai["Year"],
    zoom_2060_plus_ai["EC_plus_AI_lower"],
    zoom_2060_plus_ai["EC_plus_AI_upper"],
    color="tab:orange",
    alpha=0.3,
)
plt.plot(
    zoom_2060_plus_ai["Year"],
    zoom_2060_plus_ai["EC_plus_AI"],
    label="Forecast + AI energy",
    color="tab:orange",
    linewidth=2,
    linestyle="--",
)

plt.xlim(2020, 2060)

plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (EJ)")
plt.title("2021 - 2060")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

zoom_path = PLOTS_DIR / "ssp_zoom_2060.png"
plt.savefig(zoom_path, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 14. ZOOM PLOT (2021–2100)
# ==========================================================

plt.figure(figsize=(10, 6))

plt.plot(
    future_yearly["Year"],
    future_yearly["Predicted_EC"],
    label="Forecast (ARIMA GDP/Population)",
    color="tab:green",
    linewidth=2,
)
y_err = 0.02 * future_yearly["Predicted_EC"].values
plt.fill_between(
    future_yearly["Year"],
    future_yearly["Predicted_EC"] - y_err,
    future_yearly["Predicted_EC"] + y_err,
    color="tab:green",
    alpha=0.3,
)
plt.fill_between(
    future_yearly_plus_ai_from_2027["Year"],
    future_yearly_plus_ai_from_2027["EC_plus_AI_lower"],
    future_yearly_plus_ai_from_2027["EC_plus_AI_upper"],
    color="tab:orange",
    alpha=0.3,
)
plt.plot(
    future_yearly_plus_ai_from_2027["Year"],
    future_yearly_plus_ai_from_2027["EC_plus_AI"],
    label="Forecast + AI energy",
    color="tab:orange",
    linewidth=2,
    linestyle="--",
)

plt.xlim(2020, 2100)

plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (EJ)")
plt.title("2021 - 2100")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

zoom_path = PLOTS_DIR / "ssp_zoom_2100.png"
plt.savefig(zoom_path, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 15. ZOOM PLOT (2061–2100)
# ==========================================================

plt.figure(figsize=(10, 6))

future_2100_plus_ai = future_yearly_plus_ai_from_2027[future_yearly_plus_ai_from_2027["Year"] >= 2061]
plt.plot(
    future_2100["Year"],
    future_2100["Predicted_EC"],
    label="Forecast (ARIMA GDP/Population)",
    color="tab:green",
    linewidth=2,
)
y_err = 0.02 * future_2100["Predicted_EC"].values
plt.fill_between(
    future_2100["Year"],
    future_2100["Predicted_EC"] - y_err,
    future_2100["Predicted_EC"] + y_err,
    color="tab:green",
    alpha=0.3,
)
plt.fill_between(
    future_2100_plus_ai["Year"],
    future_2100_plus_ai["EC_plus_AI_lower"],
    future_2100_plus_ai["EC_plus_AI_upper"],
    color="tab:orange",
    alpha=0.3,
)
plt.plot(
    future_2100_plus_ai["Year"],
    future_2100_plus_ai["EC_plus_AI"],
    label="Forecast + AI energy",
    color="tab:orange",
    linewidth=2,
    linestyle="--",
)

plt.xlim(2061, 2100)

plt.xlabel("Year")
plt.ylabel("Total Energy Consumption (EJ)")
plt.title("2061 - 2100")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

zoom_path = PLOTS_DIR / "ssp_zoom_2060_2100.png"
plt.savefig(zoom_path, dpi=300, bbox_inches="tight")
plt.close()

# ==========================================================
# 16. SAVE OUTPUTS
# ==========================================================

future_proj.to_csv(DATA_DIR / "future_country_predictions.csv", index=False)
future_2100.to_csv(DATA_DIR / "future_global_predictions.csv", index=False)

print("Saved output files.")
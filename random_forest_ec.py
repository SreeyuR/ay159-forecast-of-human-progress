import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path

# ==========================================================
# 1. OUTPUT FOLDER
# ==========================================================

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# 2. LOAD HISTORICAL + SSP DATA
# ==========================================================

# 1960-2020
url_final = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"
# 2020-2060
url_ssp126 = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs126.csv"
url_ssp245 = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs245.csv"
url_ssp370 = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs370.csv"
url_ssp585 = "https://raw.githubusercontent.com/AntongZ1/Data/main/inputs585.csv"

# 1960-2020
finaldata = pd.read_csv(url_final)

ssp126 = pd.read_csv(url_ssp126)
ssp245 = pd.read_csv(url_ssp245)
ssp370 = pd.read_csv(url_ssp370)
ssp585 = pd.read_csv(url_ssp585)

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
# 4. TRAIN MODEL
# ==========================================================

train_data = finaldata[finaldata["Year"] < 2020].copy()

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
# 7. SSP SCENARIO PREDICTION (2020–2060)
# ==========================================================

def predict_scenario(df):

    X_future = df[features].copy()
    X_future["Country Code"] = X_future["Country Code"].astype("category")

    df = df.copy()
    df["Predicted_EC"] = model.predict(X_future)

    yearly = (
        df.groupby("Year", as_index=False)["Predicted_EC"]
        .sum()
    )

    return yearly

ssp126_year = predict_scenario(ssp126)
ssp245_year = predict_scenario(ssp245)
ssp370_year = predict_scenario(ssp370)
ssp585_year = predict_scenario(ssp585)

# ==========================================================
# 8. COUNTRY NAME TO COUNTRY CODE
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

# ==========================================================
# 9. LOAD FUTURE GDP + POPULATION (2061–2100)
# ==========================================================

gdp_future = pd.read_csv("ARIMA input data - GDP.csv")
pop_future = pd.read_csv("ARIMA input data - Population.csv")

gdp_future = gdp_future[gdp_future["Year"] >= 2061]
pop_future = pop_future[pop_future["Year"] >= 2061]

def wide_to_long(df, value_name):
    return df.melt(
        id_vars="Year",
        var_name="Country",
        value_name=value_name
    )

gdp_long = wide_to_long(gdp_future, "GDP")
pop_long = wide_to_long(pop_future, "Population")

future_proj = gdp_long.merge(
    pop_long,
    on=["Year", "Country"]
)

future_proj["Country Code"] = future_proj["Country"].map(country_to_code)
future_proj = future_proj.dropna()

# ==========================================================
# 10. FUTURE PREDICTION (2061–2100)
# ==========================================================

X_proj = future_proj[features].copy()
X_proj["Country Code"] = X_proj["Country Code"].astype("category")

future_proj["Predicted_EC"] = model.predict(X_proj)

future_2100 = (
    future_proj.groupby("Year", as_index=False)["Predicted_EC"]
    .sum()
)

# ==========================================================
# 11. EXTEND ALL SSPs TO 2100
# ==========================================================

ssp126_extended = pd.concat([ssp126_year, future_2100])
ssp245_extended = pd.concat([ssp245_year, future_2100])
ssp370_extended = pd.concat([ssp370_year, future_2100])
ssp585_extended = pd.concat([ssp585_year, future_2100])

# ==========================================================
# 12. FULL RANGE PLOT (1960–2100)
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    historical_global["Year"],
    historical_global["EC"],
    label="Historical data",
    linewidth=2,
    color="blue"
)

plt.plot(
    ssp126_extended["Year"],
    ssp126_extended["Predicted_EC"],
    label="Forecast (ssp126)",
    color="tab:blue",
    linewidth=2,
)
plt.plot(
    ssp245_extended["Year"],
    ssp245_extended["Predicted_EC"],
    label="Forecast (ssp245)",
    color="tab:green",
    linewidth=2,
)
plt.plot(
    ssp370_extended["Year"],
    ssp370_extended["Predicted_EC"],
    label="Forecast (ssp370)",
    color="tab:orange",
    linewidth=2,
)
plt.plot(
    ssp585_extended["Year"],
    ssp585_extended["Predicted_EC"],
    label="Forecast (ssp585)",
    color="tab:red",
    linewidth=2,
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

plt.plot(
    ssp126_year["Year"],
    ssp126_year["Predicted_EC"],
    label="Forecast (ssp126)",
    color="tab:blue",
    linewidth=2,
)
plt.plot(
    ssp245_year["Year"],
    ssp245_year["Predicted_EC"],
    label="Forecast (ssp245)",
    color="tab:green",
    linewidth=2,
)
plt.plot(
    ssp370_year["Year"],
    ssp370_year["Predicted_EC"],
    label="Forecast (ssp370)",
    color="tab:orange",
    linewidth=2,
)
plt.plot(
    ssp585_year["Year"],
    ssp585_year["Predicted_EC"],
    label="Forecast (ssp585)",
    color="tab:red",
    linewidth=2,
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
    ssp126_extended["Year"],
    ssp126_extended["Predicted_EC"],
    label="Forecast (ssp126)",
    color="tab:blue",
    linewidth=2,
)
plt.plot(
    ssp245_extended["Year"],
    ssp245_extended["Predicted_EC"],
    label="Forecast (ssp245)",
    color="tab:green",
    linewidth=2,
)
plt.plot(
    ssp370_extended["Year"],
    ssp370_extended["Predicted_EC"],
    label="Forecast (ssp370)",
    color="tab:orange",
    linewidth=2,
)
plt.plot(
    ssp585_extended["Year"],
    ssp585_extended["Predicted_EC"],
    label="Forecast (ssp585)",
    color="tab:red",
    linewidth=2,
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

plt.plot(
    ssp126_extended["Year"],
    ssp126_extended["Predicted_EC"],
    label="Forecast (ssp126)",
    color="tab:blue",
    linewidth=2,
)
plt.plot(
    ssp245_extended["Year"],
    ssp245_extended["Predicted_EC"],
    label="Forecast (ssp245)",
    color="tab:green",
    linewidth=2,
)
plt.plot(
    ssp370_extended["Year"],
    ssp370_extended["Predicted_EC"],
    label="Forecast (ssp370)",
    color="tab:orange",
    linewidth=2,
)
plt.plot(
    ssp585_extended["Year"],
    ssp585_extended["Predicted_EC"],
    label="Forecast (ssp585)",
    color="tab:red",
    linewidth=2,
)

plt.xlim(2061, 2100)

# Tighten y-axis to highlight trends in 2061–2100 window
zoom_window = lambda df: df[df["Year"].between(2061, 2100)]["Predicted_EC"]
zoom_min = min(
    zoom_window(ssp126_extended).min(),
    zoom_window(ssp245_extended).min(),
    zoom_window(ssp370_extended).min(),
    zoom_window(ssp585_extended).min(),
)
zoom_max = max(
    zoom_window(ssp126_extended).max(),
    zoom_window(ssp245_extended).max(),
    zoom_window(ssp370_extended).max(),
    zoom_window(ssp585_extended).max(),
)
pad = (zoom_max - zoom_min) * 0.08 if zoom_max > zoom_min else 10
plt.ylim(zoom_min - pad, zoom_max + pad)

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

future_proj.to_csv("future_country_predictions.csv", index=False)
future_2100.to_csv("future_global_predictions.csv", index=False)

print("Saved output files.")
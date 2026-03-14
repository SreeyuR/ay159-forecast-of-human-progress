import pandas as pd
import lightgbm as lgb

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score


# ==========================================================
# 1. LOAD DATA (same source as random_forest_ec.py)
# ==========================================================

url_final = "https://raw.githubusercontent.com/AntongZ1/Data/main/finaldata0411.csv"
finaldata = pd.read_csv(url_final)


# ==========================================================
# 2. FEATURES AND TARGET
# ==========================================================

features = [
    "Year",
    "GDP",
    "Population",
    "Country Code",
]

target = "EC"


# ==========================================================
# 3. TRAIN ONLY ON HISTORICAL PRE-2020
# ==========================================================

train_data = finaldata[finaldata["Year"] < 2020].copy()

X = train_data[features].copy()
y = train_data[target]

# LightGBM can use pandas "category" dtype for categorical features
X["Country Code"] = X["Country Code"].astype("category")


# ==========================================================
# 4. TRAIN / VALIDATION SPLIT
# ==========================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ==========================================================
# 5. HYPERPARAMETER GRID FOR LGBMRegressor
# ==========================================================
#
# This grid is intentionally moderate in size so that it finishes
# in a reasonable amount of time, but still explores:
# - tree count / learning rate tradeoff
# - depth and leaves (model complexity)
# - min_child_samples (regularization)
# - subsampling of rows/columns (stability)

base_model = lgb.LGBMRegressor(
    boosting_type="gbdt",
    objective="regression",
    random_state=42,
    n_jobs=-1,
)

param_grid = {
    "n_estimators": [200, 500, 1000],
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [-1, 8, 12],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 30, 60],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ==========================================================
# 6. GRID SEARCH WITH 3-FOLD CV (R²)
# ==========================================================

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

print("Starting hyperparameter search...")
grid_search.fit(X_train, y_train)

print("\nBest CV R²:", grid_search.best_score_)
print("Best hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")


# ==========================================================
# 7. EVALUATE BEST MODEL ON HOLD-OUT VALIDATION SET
# ==========================================================

best_model: lgb.LGBMRegressor = grid_search.best_estimator_

y_val_pred = best_model.predict(X_val)
r2_val = r2_score(y_val, y_val_pred)

print("\nHold-out validation R² with best params:", r2_val)


# ==========================================================
# 8. OPTIONAL: SAVE BEST PARAMS TO A TEXT FILE
# ==========================================================

OUTPUTS_DIR = Path("tuning_outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

params_path = OUTPUTS_DIR / "best_lgbm_params.txt"

with params_path.open("w") as f:
    f.write(f"Best CV R²: {grid_search.best_score_}\n")
    f.write(f"Validation R²: {r2_val}\n\n")
    f.write("Best hyperparameters:\n")
    for k, v in grid_search.best_params_.items():
        f.write(f"{k}: {v}\n")

print(f"\nSaved best hyperparameters to: {params_path}")


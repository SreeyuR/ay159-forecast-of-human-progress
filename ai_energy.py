"""
Fit and compare growth models (quadratic, exponential, logistic, Gompertz) to AI-related
energy consumption (EJ) for 2020–2030. Provides central forecast and uncertainty band
for use in random_forest_ec and ratio scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# IEA AI energy data (2020–2030), EJ
AI_ENERGY_YEARS = np.arange(2020, 2031)
AI_ENERGY_DATA = np.array([
    1.68, 1.83, 1.98, 2.19, 2.49, 2.94, 3.42, 4.08, 4.80, 5.28, 5.67
])


def quadratic(t, a, b, c):
    """a + b*t + c*t^2."""
    return a + b * t + c * t**2


def exponential(t, A, k):
    """A * exp(k*t)."""
    return A * np.exp(k * t)


def logistic(t, L, k, t0):
    """L / (1 + exp(-k*(t - t0)))."""
    return L / (1 + np.exp(-k * (t - t0)))


def gompertz(t, L, k, t0):
    """L * exp(-exp(-k*(t - t0)))."""
    return L * np.exp(-np.exp(-k * (t - t0)))


# Model configs for AIC-based best fit: func, initial params (p0), bounds for curve_fit
MODELS = {
    "Quadratic": {"func": quadratic, "p0": [1.5, 0.2, 0.02], "bounds": (-np.inf, np.inf)},
    "Exponential": {"func": exponential, "p0": [1.5, 0.12], "bounds": (0, np.inf)},
    "Logistic": {"func": logistic, "p0": [8.0, 0.4, 6.0], "bounds": ([0, 0, -50], [1000, 10, 50])},
    "Gompertz": {"func": gompertz, "p0": [8.0, 0.4, 6.0], "bounds": ([0, 0, -50], [1000, 10, 50])},
}

# Params for uncertainty band (quadratic = middle, logistic/gompertz = envelope)
UNCERTAINTY_FIT = {
    "Quadratic": {"func": quadratic, "p0": [1.5, 0.2, 0.02], "bounds": (-np.inf, np.inf)},
    "Logistic": {"func": logistic, "p0": [24, 0.16, 17], "bounds": ([0, 0, -100], [1000, 10, 100])},
    "Gompertz": {"func": gompertz, "p0": [20, 0.2, 15], "bounds": ([0, 0, -100], [1000, 10, 100])},
}


def _compute_metrics(y_true, y_pred, n_params):
    """Return RSS, RMSE, R², and AIC for a fit."""
    resid = y_true - y_pred
    rss = np.sum(resid**2)
    rmse = np.sqrt(np.mean(resid**2))
    tss = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - rss / tss
    n = len(y_true)
    aic = n * np.log(rss / n) + 2 * n_params
    return rss, rmse, r2, aic


def _fit_ai_models():
    """Fit all models to 2020–2030 AI energy data; return results dict and best model name."""
    t_data = AI_ENERGY_YEARS - 2020
    results = {}
    for name, info in MODELS.items():
        func, p0, bounds = info["func"], info["p0"], info["bounds"]
        try:
            popt, _ = curve_fit(func, t_data, AI_ENERGY_DATA, p0=p0, bounds=bounds, maxfev=50000)
            y_fit = func(t_data, *popt)
            rss, rmse, r2, aic = _compute_metrics(AI_ENERGY_DATA, y_fit, len(popt))
            results[name] = {"params": popt, "y_fit": y_fit, "rss": rss, "rmse": rmse, "r2": r2, "aic": aic}
        except Exception as e:
            results[name] = {"error": str(e)}
    valid = {k: v for k, v in results.items() if "error" not in v}
    best_name = min(valid, key=lambda k: valid[k]["aic"]) if valid else None
    return results, best_name


def _fit_uncertainty_models():
    """Fit quadratic, logistic, Gompertz for uncertainty band; return (popt_quad, popt_log, popt_gom)."""
    t_data = AI_ENERGY_YEARS - 2020
    popt_quad, _ = curve_fit(
        quadratic, t_data, AI_ENERGY_DATA,
        p0=UNCERTAINTY_FIT["Quadratic"]["p0"],
        bounds=UNCERTAINTY_FIT["Quadratic"]["bounds"],
    )
    popt_log, _ = curve_fit(
        logistic, t_data, AI_ENERGY_DATA,
        p0=UNCERTAINTY_FIT["Logistic"]["p0"],
        bounds=UNCERTAINTY_FIT["Logistic"]["bounds"],
        maxfev=50000,
    )
    popt_gom, _ = curve_fit(
        gompertz, t_data, AI_ENERGY_DATA,
        p0=UNCERTAINTY_FIT["Gompertz"]["p0"],
        bounds=UNCERTAINTY_FIT["Gompertz"]["bounds"],
        maxfev=50000,
    )
    return popt_quad, popt_log, popt_gom


def get_ai_energy_for_years(years):
    """
    Return AI energy (EJ) for the given years using the best-fit model (by AIC).
    Years before 2020 are set to 0; 2020+ use the fitted curve (extrapolated beyond 2030).
    """
    years = np.asarray(years)
    out = np.zeros_like(years, dtype=float)
    results, best_name = _fit_ai_models()
    if best_name is None:
        return out
    func = MODELS[best_name]["func"]
    popt = results[best_name]["params"]
    mask = years >= 2020
    t = years[mask] - 2020
    out[mask] = func(t, *popt)
    out = np.maximum(out, 0.0)
    return out


def get_ai_energy_uncertainty_for_years(years):
    """
    Return (middle, lower, upper) AI energy (EJ) for the given years.
    Middle = quadratic fit; lower/upper = envelope of logistic, Gompertz.
    Fitted to 2020–2030 IEA data. For years < 2020, returns zeros.
    """
    years = np.asarray(years, dtype=float)
    middle = np.zeros_like(years, dtype=float)
    lower = np.zeros_like(years, dtype=float)
    upper = np.zeros_like(years, dtype=float)

    popt_quad, popt_log, popt_gom = _fit_uncertainty_models()

    mask = years >= 2020
    t = years[mask] - 2020
    mid_vals = quadratic(t, *popt_quad)
    log_vals = np.maximum(logistic(t, *popt_log), 0.0)
    gom_vals = np.maximum(gompertz(t, *popt_gom), 0.0)
    lower_final = np.minimum.reduce([log_vals, mid_vals, gom_vals])
    upper_final = np.maximum.reduce([log_vals, mid_vals, gom_vals])

    middle[mask] = mid_vals
    lower[mask] = lower_final
    upper[mask] = upper_final
    return middle, lower, upper


if __name__ == "__main__":
    t_data = AI_ENERGY_YEARS - 2020
    results, best_name = _fit_ai_models()

    print("\nFit results (AIC comparison):\n")
    for name, res in results.items():
        if "error" in res:
            print(f"{name}: failed -> {res['error']}")
        else:
            print(f"{name}:")
            print(f"  parameters = {res['params']}")
            print(f"  RMSE       = {res['rmse']:.6f}")
            print(f"  R^2        = {res['r2']:.6f}")
            print(f"  AIC        = {res['aic']:.6f}")
            print()
    print(f"Best fit by AIC: {best_name}")

    # Plot: data + all model fits (2020–2030)
    t_fine = np.linspace(t_data.min(), t_data.max(), 400)
    years_fine = 2020 + t_fine
    plt.figure(figsize=(9, 6))
    plt.plot(AI_ENERGY_YEARS, AI_ENERGY_DATA, "o", label="Data", markersize=7)
    valid = {k: v for k, v in results.items() if "error" not in v}
    for name, res in valid.items():
        y_fine = MODELS[name]["func"](t_fine, *res["params"])
        plt.plot(years_fine, y_fine, label=name)
    plt.xlabel("Year")
    plt.ylabel("Energy (EJ)")
    plt.title("Fits to 2020–2030 AI Energy Data")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Build and save uncertainty CSV (2020–2099)
    years_export = np.arange(2020, 2100)
    mid, low, high = get_ai_energy_uncertainty_for_years(years_export)
    pd.DataFrame({
        "Year": years_export,
        "Lower_EJ": low,
        "Middle_EJ": mid,
        "Upper_EJ": high,
    }).to_csv("data/ai_energy_uncertainty_2020_2100.csv", index=False)
    print("Saved data/ai_energy_uncertainty_2020_2100.csv")

    # Plot: middle curve + uncertainty band
    yerr_lo = mid - low
    yerr_hi = high - mid
    yerr = np.vstack([yerr_lo, yerr_hi])
    plt.figure(figsize=(10, 6))
    plt.plot(AI_ENERGY_YEARS, AI_ENERGY_DATA, "o", label="Data (IEA projections)")
    plt.plot(years_export, mid, label="Center fit (Quadratic)")
    plt.errorbar(
        years_export, mid, yerr=yerr,
        fmt="none", capsize=3, elinewidth=1, alpha=0.6,
        label="Uncertainty (Gompertz and Logistic Fits)",
    )
    plt.xlabel("Year")
    plt.ylabel("Energy (EJ)")
    plt.title("Global AI Energy Demand with Uncertainty Error Bars")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================
# 03_models.py
# Toronto Shelter System Flow - Fit and Compare Forecasting Models
# =============================================================

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/actively_homeless_by_group.csv"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8985"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})

# -----------------------------------------------------------------------------
# STEP 1: LOAD THE SERIES AND BUILD THE COVID INDICATOR
# The indicator is 1 from Mar 2020 to Jun 2021 (the drop and its aftermath)
# and 0 otherwise. It lets one model absorb the break instead of treating it
# as ordinary noise.
# -----------------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

wide = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
y = wide["All Population"].astype(float)
y.index.freq = "MS"
covid = pd.DataFrame(
    {"covid": ((y.index >= "2020-03-01") & (y.index <= "2021-06-01")).astype(float)},
    index=y.index,
)
print(f"{len(y)} months, {y.index.min():%b %Y} to {y.index.max():%b %Y}")
print(f"COVID indicator months: {int(covid['covid'].sum())}")

# -----------------------------------------------------------------------------
# STEP 2: ETS (HOLT-WINTERS)
# Compare additive trend, damped trend, and with / without seasonality.
# AIC rewards fit and penalizes complexity; lower is better.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: ETS / Holt-Winters candidates")
print("=" * 60)

ets_rows, ets_fits = [], {}
for trend, damped, seasonal in itertools.product(["add"], [False, True], [None, "add"]):
    name = f"ETS trend={'damped' if damped else 'add'}, seasonal={seasonal or 'none'}"
    try:
        fit = ExponentialSmoothing(
            y, trend=trend, damped_trend=damped, seasonal=seasonal,
            seasonal_periods=12 if seasonal else None,
            initialization_method="estimated",
        ).fit()
        ets_rows.append({"model": name, "aic": round(fit.aic, 1)})
        ets_fits[name] = (fit, damped, seasonal)
    except Exception as e:
        print(f"  {name} failed: {e}")
ets_table = pd.DataFrame(ets_rows).sort_values("aic").reset_index(drop=True)
print(ets_table.to_string(index=False))
best_ets_name = ets_table.loc[0, "model"]
best_ets, ets_damped, ets_seasonal = ets_fits[best_ets_name]
print(f"\nBest ETS by AIC: {best_ets_name}")

# -----------------------------------------------------------------------------
# STEP 3: SARIMA ORDER SEARCH
# Small grid, guided by the diagnostics: the series needs one difference,
# ACF/PACF of the differences suggest low AR/MA orders, and the seasonal
# pattern is weak so seasonal orders stay small.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: SARIMA order search (lowest AIC)")
print("=" * 60)


def search(exog=None):
    rows = []
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
        try:
            res = SARIMAX(y, exog=exog, order=(p, 1, q), seasonal_order=(P, 0, Q, 12),
                          enforce_stationarity=False, enforce_invertibility=False
                          ).fit(disp=False, maxiter=200)
            rows.append({"p": p, "d": 1, "q": q, "P": P, "D": 0, "Q": Q,
                         "aic": res.aic})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


sarima_grid = search()
print("Top 5 SARIMA (no COVID indicator):")
print(sarima_grid.head(5).round(1).to_string(index=False))
sarimax_grid = search(exog=covid)
print("\nTop 5 SARIMA with COVID indicator:")
print(sarimax_grid.head(5).round(1).to_string(index=False))


def fit_best(grid, exog=None):
    r = grid.iloc[0]
    order, sorder = (int(r.p), 1, int(r.q)), (int(r.P), 0, int(r.Q), 12)
    res = SARIMAX(y, exog=exog, order=order, seasonal_order=sorder,
                  enforce_stationarity=False, enforce_invertibility=False
                  ).fit(disp=False, maxiter=200)
    return res, order, sorder


sarima, s_order, s_seas = fit_best(sarima_grid)
sarimax, x_order, x_seas = fit_best(sarimax_grid, exog=covid)
print(f"\nChosen SARIMA:              {s_order} x {s_seas}")
print(f"Chosen SARIMA + COVID:      {x_order} x {x_seas}")
print(f"COVID coefficient:          {sarimax.params['covid']:,.0f} people "
      f"(p = {sarimax.pvalues['covid']:.3f})")

# -----------------------------------------------------------------------------
# STEP 4: COMPARE FIT AND CHECK RESIDUALS
# Ljung-Box tests whether leftover errors still contain autocorrelation.
# p >= 0.05 means no evidence the model missed structure.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Model comparison and residual checks")
print("=" * 60)

models = {
    "ETS": (best_ets.fittedvalues, best_ets.resid, best_ets.aic),
    "SARIMA": (sarima.fittedvalues, sarima.resid, sarima.aic),
    "SARIMA + COVID": (sarimax.fittedvalues, sarimax.resid, sarimax.aic),
}
rows = []
for name, (fitted, resid, aic) in models.items():
    resid = resid.iloc[13:]   # skip start-up residuals of the state space models
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
    rows.append({
        "model": name, "aic": round(aic, 1),
        "in_sample_mae": round(resid.abs().mean(), 0),
        "residual_std": round(resid.std(), 0),
        "ljung_box_p": round(lb, 3),
        "residuals_clean": "yes" if lb >= 0.05 else "no",
    })
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
print("\nAIC is only comparable between the two SARIMA models (same data scale and")
print("differencing); ETS is judged on in-sample error and, later, the backtest.")
summary.to_csv("outputs/model_fit_summary.csv", index=False)

orders = pd.DataFrame([
    {"model": "SARIMA", "order": str(s_order), "seasonal_order": str(s_seas)},
    {"model": "SARIMA + COVID", "order": str(x_order), "seasonal_order": str(x_seas)},
    {"model": "ETS", "order": f"damped={ets_damped}", "seasonal_order": f"seasonal={ets_seasonal}"},
])
orders.to_csv("outputs/model_orders.csv", index=False)

# -----------------------------------------------------------------------------
# STEP 5: FIGURE - FITTED VALUES VS ACTUAL
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(y.index, y.values, color=INK_2, linewidth=2, label="Actual")
ax.plot(y.index[13:], models["ETS"][0].iloc[13:], color=GRAY, linewidth=1.4, label="ETS")
ax.plot(y.index[13:], models["SARIMA"][0].iloc[13:], color=BLUE, linewidth=1.4, label="SARIMA")
ax.plot(y.index[13:], models["SARIMA + COVID"][0].iloc[13:], color=ORANGE, linewidth=1.4,
        label="SARIMA + COVID")
ax.set_ylabel("People actively homeless")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("In-sample fit: all three models track the series",
             loc="left", fontsize=13, fontweight="bold")
ax.legend(frameon=False, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/12_model_fits.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved: outputs/model_fit_summary.csv, outputs/model_orders.csv, 12_model_fits.png")
print("\nModel fitting complete. Ready for backtesting.")
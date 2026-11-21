# =============================================================
# 04_backtest.py
# Toronto Shelter System Flow - Rolling-Origin Backtest
# =============================================================

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
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

# Same colour for the same model in every chart.
COLORS = {"Naive": INK_2, "Seasonal naive": GRAY, "Drift": "#b8b6ae",
          "ETS": "#1baa8f", "SARIMA": BLUE, "SARIMA + COVID": ORANGE}

MAX_H = 6          # longest horizon scored (months)
N_ORIGINS = 36     # number of forecast origins (months we pretend to be "today")

# -----------------------------------------------------------------------------
# STEP 1: LOAD AND SET UP THE BACKTEST
# At each origin the models see only data up to that month, forecast the next
# 6 months, and are scored against what really happened. Every model is scored
# on exactly the same origins, so the comparison is fair.
# -----------------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Setting up the rolling-origin backtest")
print("=" * 60)

wide = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
y = wide["All Population"].astype(float)
y.index.freq = "MS"
covid = pd.Series(((y.index >= "2020-03-01") & (y.index <= "2021-06-01")).astype(float),
                  index=y.index)

orders = pd.read_csv("outputs/model_orders.csv").set_index("model")
s_order = eval(orders.loc["SARIMA", "order"])
s_seas = eval(orders.loc["SARIMA", "seasonal_order"])
x_order = eval(orders.loc["SARIMA + COVID", "order"])
x_seas = eval(orders.loc["SARIMA + COVID", "seasonal_order"])
ets_damped = orders.loc["ETS", "order"] == "damped=True"
ets_seasonal = None if orders.loc["ETS", "seasonal_order"] == "seasonal=None" else "add"

n = len(y)
origins = list(range(n - MAX_H - N_ORIGINS, n - MAX_H))
print(f"{N_ORIGINS} origins: {y.index[origins[0]]:%b %Y} to {y.index[origins[-1]]:%b %Y}")
print(f"Each origin forecasts 1 to {MAX_H} months ahead")
print(f"Models re-fit at every origin: ETS, SARIMA {s_order}x{s_seas}, SARIMA + COVID")

# -----------------------------------------------------------------------------
# STEP 2: RUN THE BACKTEST
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: Running forecasts (this takes a minute)")
print("=" * 60)

records = []
for i, o in enumerate(origins, 1):
    train = y.iloc[: o + 1]
    last = train.iloc[-1]
    drift_slope = (last - train.iloc[0]) / (len(train) - 1)
    preds = {}
    for h in range(1, MAX_H + 1):
        preds.setdefault("Naive", []).append(last)
        preds.setdefault("Seasonal naive", []).append(y.iloc[o + h - 12])
        preds.setdefault("Drift", []).append(last + h * drift_slope)

    try:
        ets = ExponentialSmoothing(train, trend="add", damped_trend=ets_damped,
                                   seasonal=ets_seasonal,
                                   seasonal_periods=12 if ets_seasonal else None,
                                   initialization_method="estimated").fit()
        preds["ETS"] = list(ets.forecast(MAX_H))
    except Exception:
        preds["ETS"] = [np.nan] * MAX_H
    try:
        sar = SARIMAX(train, order=s_order, seasonal_order=s_seas,
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False, maxiter=200)
        preds["SARIMA"] = list(sar.forecast(MAX_H))
    except Exception:
        preds["SARIMA"] = [np.nan] * MAX_H
    try:
        sarx = SARIMAX(train, exog=covid.iloc[: o + 1], order=x_order, seasonal_order=x_seas,
                       enforce_stationarity=False, enforce_invertibility=False
                       ).fit(disp=False, maxiter=200)
        preds["SARIMA + COVID"] = list(sarx.forecast(MAX_H, exog=covid.iloc[o + 1: o + 1 + MAX_H]))
    except Exception:
        preds["SARIMA + COVID"] = [np.nan] * MAX_H

    for model, vals in preds.items():
        for h, p in enumerate(vals, 1):
            records.append({"origin": y.index[o], "target": y.index[o + h], "horizon": h,
                            "model": model, "forecast": p, "actual": y.iloc[o + h]})
    if i % 12 == 0:
        print(f"  {i} of {len(origins)} origins done")

bt = pd.DataFrame(records)
bt["error"] = bt["actual"] - bt["forecast"]
print(f"\nForecast rows: {len(bt):,}   Failed fits: {bt['forecast'].isna().sum()}")
bt.to_csv("outputs/backtest_forecasts.csv", index=False)

# -----------------------------------------------------------------------------
# STEP 3: SCORE
# MAE = average miss in people. RMSE punishes large misses more.
# MAPE = average miss as a percent of the true value.
# "vs naive" below 1.00 means the model beats just repeating the last value.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: Scores by model and horizon")
print("=" * 60)

order = ["Naive", "Seasonal naive", "Drift", "ETS", "SARIMA", "SARIMA + COVID"]
g = bt.groupby(["model", "horizon"])
scores = g.apply(lambda d: pd.Series({
    "mae": d["error"].abs().mean(),
    "rmse": np.sqrt((d["error"] ** 2).mean()),
    "mape_pct": (d["error"].abs() / d["actual"]).mean() * 100,
})).reset_index()
naive_mae = scores[scores["model"] == "Naive"].set_index("horizon")["mae"]
scores["mae_vs_naive"] = scores.apply(lambda r: r["mae"] / naive_mae[r["horizon"]], axis=1)
scores["model"] = pd.Categorical(scores["model"], order, ordered=True)
scores = scores.sort_values(["horizon", "model"]).round(2)
scores.to_csv("outputs/backtest_scores.csv", index=False)

for h in (1, 3, 6):
    print(f"\nHorizon {h} month{'s' if h > 1 else ''}:")
    print(scores[scores["horizon"] == h].drop(columns="horizon").to_string(index=False))

# -----------------------------------------------------------------------------
# STEP 4: WHO WINS?
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Best model at each horizon (lowest MAE)")
print("=" * 60)

best = scores.loc[scores.groupby("horizon")["mae"].idxmin(), ["horizon", "model", "mae", "mae_vs_naive"]]
print(best.to_string(index=False))
avg = scores.groupby("model", observed=True)["mae"].mean().round(0).sort_values()
print("\nAverage MAE across horizons 1-6:")
print(avg.to_string())

# -----------------------------------------------------------------------------
# STEP 5: FIGURES
# -----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5.5))
for m in order:
    d = scores[scores["model"] == m]
    lw = 2.6 if m in ("Naive", "SARIMA", "ETS") else 1.6
    ax.plot(d["horizon"], d["mae"], marker="o", color=COLORS[m], linewidth=lw, label=m)
ax.set_xlabel("Months ahead")
ax.set_ylabel("Mean absolute error (people)")
ax.set_xticks(range(1, MAX_H + 1))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("Forecast error grows with the horizon", loc="left", fontsize=13, fontweight="bold")
ax.legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/13_backtest_mae.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5.5))
actual = y.iloc[origins[0]:]
ax.plot(actual.index, actual.values, color=INK, linewidth=2.4, label="Actual")
d3 = bt[bt["horizon"] == 3]
for m in ["Naive", "ETS", "SARIMA"]:
    s = d3[d3["model"] == m].set_index("target")["forecast"]
    ax.plot(s.index, s.values, color=COLORS[m], linewidth=1.6, label=f"{m} (3 months ahead)")
ax.set_ylabel("People actively homeless")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("Three-month-ahead forecasts against what happened",
             loc="left", fontsize=13, fontweight="bold")
ax.legend(frameon=False, ncol=4, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/14_backtest_forecasts.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved: outputs/backtest_forecasts.csv, outputs/backtest_scores.csv")
print("Saved: 13_backtest_mae.png, 14_backtest_forecasts.png")
print("\nBacktest complete. Ready for the final forecast.")
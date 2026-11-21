# =============================================================
# 02_diagnostics.py
# Toronto Shelter System Flow - Time Series Diagnostics
# =============================================================

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

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
# STEP 1: LOAD THE MAIN SERIES
# -----------------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Loading the All Population series")
print("=" * 60)

wide = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
y = wide["All Population"].astype(float)
y.index.freq = "MS"

print(f"Months: {len(y)}   Range: {y.index.min():%b %Y} to {y.index.max():%b %Y}")
print(f"Min {y.min():,.0f} ({y.idxmin():%b %Y})   Max {y.max():,.0f} ({y.idxmax():%b %Y})")
print(f"Last value: {y.iloc[-1]:,.0f} ({y.index[-1]:%b %Y})")

# -----------------------------------------------------------------------------
# STEP 2: TREND AND SEASONALITY (STL DECOMPOSITION)
# Strength runs from 0 (none) to 1 (the component explains everything).
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: Trend and seasonality (STL decomposition)")
print("=" * 60)

stl = STL(y, period=12, robust=True).fit()
trend_strength = max(0, 1 - stl.resid.var() / (stl.resid + stl.trend).var())
seasonal_strength = max(0, 1 - stl.resid.var() / (stl.resid + stl.seasonal).var())
print(f"Trend strength:    {trend_strength:.2f}")
print(f"Seasonal strength: {seasonal_strength:.2f}")

profile = (stl.seasonal.groupby(stl.seasonal.index.month).mean()).round(0)
profile.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in profile.index]
print("\nTypical seasonal effect by month (people above or below the trend):")
print(profile.astype(int).to_string())

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
parts = [("Observed", y, BLUE), ("Trend", stl.trend, INK_2),
         ("Seasonal", stl.seasonal, ORANGE), ("Remainder", stl.resid, GRAY)]
for ax, (name, series, color) in zip(axes, parts):
    ax.plot(series.index, series.values, color=color, linewidth=1.8)
    ax.set_ylabel(name)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
axes[0].set_title("STL decomposition of the actively homeless count",
                  loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/09_stl_decomposition.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: 09_stl_decomposition.png")

# -----------------------------------------------------------------------------
# STEP 3: STATIONARITY TESTS
# ADF null hypothesis: the series has a unit root (not stationary).
# KPSS null hypothesis: the series is stationary.
# Reading them together is more reliable than either alone.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: Stationarity tests")
print("=" * 60)


def adf_p(x):
    r = adfuller(x, autolag="AIC")
    return r[1]


def kpss_p(x):
    return kpss(x, regression="c", nlags="auto")[1]


candidates = {
    "Level": y,
    "First difference": y.diff().dropna(),
    "Seasonal difference (12)": y.diff(12).dropna(),
    "First + seasonal difference": y.diff().diff(12).dropna(),
}

rows = []
for name, x in candidates.items():
    p_adf, p_kpss = adf_p(x), kpss_p(x)
    if p_adf < 0.05 and p_kpss >= 0.05:
        verdict = "stationary"
    elif p_adf >= 0.05 and p_kpss < 0.05:
        verdict = "not stationary"
    else:
        verdict = "inconclusive"
    rows.append({"series": name, "observations": len(x),
                 "adf_p": round(p_adf, 3), "kpss_p": round(p_kpss, 3), "verdict": verdict})

stationarity = pd.DataFrame(rows)
print(stationarity.to_string(index=False))
print("\nADF p < 0.05 and KPSS p >= 0.05 together mean stationary.")
stationarity.to_csv("outputs/stationarity_tests.csv", index=False)

# -----------------------------------------------------------------------------
# STEP 4: AUTOCORRELATION
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Autocorrelation (first difference)")
print("=" * 60)

d1 = y.diff().dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_acf(d1, lags=24, ax=axes[0], color=BLUE, vlines_kwargs={"colors": BLUE})
plot_pacf(d1, lags=24, ax=axes[1], color=BLUE, vlines_kwargs={"colors": BLUE}, method="ywm")
axes[0].set_title("Autocorrelation (ACF)", loc="left", fontsize=12, fontweight="bold")
axes[1].set_title("Partial autocorrelation (PACF)", loc="left", fontsize=12, fontweight="bold")
for ax in axes:
    ax.set_xlabel("Lag (months)")
fig.suptitle("Month-to-month changes in the actively homeless count",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/10_acf_pacf.png", dpi=150, bbox_inches="tight")
plt.close(fig)

from statsmodels.tsa.stattools import acf  # noqa: E402

acf_vals = acf(d1, nlags=24)
print("ACF of the first difference at selected lags:")
for lag in [1, 2, 3, 6, 12, 24]:
    print(f"  lag {lag:>2}: {acf_vals[lag]:>6.2f}")
print(f"  95% band: +/- {1.96 / np.sqrt(len(d1)):.2f}")
print("\nSaved: 10_acf_pacf.png")

# -----------------------------------------------------------------------------
# STEP 5: THE COVID BREAK
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: The COVID-19 break")
print("=" * 60)

pre_peak = y["2019-01-01":"2020-02-01"]
covid = y["2020-03-01":"2021-06-01"]
pre_level = y["2020-02-01"]
trough_date, trough = covid.idxmin(), covid.min()
recovered = y[trough_date:][y[trough_date:] >= pre_level]
recovery_date = recovered.index[0] if len(recovered) else None

print(f"Level in Feb 2020 (before the drop): {pre_level:,.0f}")
print(f"Lowest point:                        {trough:,.0f} ({trough_date:%b %Y})")
print(f"Fall from Feb 2020 to the low:       {trough - pre_level:,.0f} "
      f"({(trough / pre_level - 1) * 100:.1f}%)")
if recovery_date is not None:
    months = (recovery_date.year - trough_date.year) * 12 + recovery_date.month - trough_date.month
    print(f"Back above the Feb 2020 level:       {recovery_date:%b %Y} ({months} months after the low)")
else:
    print("Back above the Feb 2020 level:       not yet")

print("\nAverage level by period:")
periods = {
    "Before COVID (Jan 2018 - Feb 2020)": y[:"2020-02-01"],
    "COVID period (Mar 2020 - Jun 2021)": covid,
    "After COVID (Jul 2021 - Jun 2026)": y["2021-07-01":],
}
for label, s in periods.items():
    print(f"  {label:<38} {s.mean():>8,.0f}  ({len(s)} months)")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(y.index, y.values, color=BLUE, linewidth=2)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-01"), color=ORANGE, alpha=0.12)
ax.axhline(pre_level, color=GRAY, linestyle="--", linewidth=1)
ax.annotate(f"Feb 2020: {pre_level:,.0f}", xy=(pd.Timestamp("2018-02-01"), pre_level),
            xytext=(0, 6), textcoords="offset points", color=INK_2, fontsize=9)
ax.annotate(f"Low: {trough:,.0f} ({trough_date:%b %Y})", xy=(trough_date, trough),
            xytext=(12, -14), textcoords="offset points", color=INK, fontsize=9)
ax.set_ylabel("People actively homeless")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("The 2020 drop is a level shift, not noise", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/11_covid_break.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved: 11_covid_break.png")

# -----------------------------------------------------------------------------
# STEP 6: SIMPLE FORECAST BENCHMARKS
# For every month in the test window, forecast it using only earlier data.
# These are the scores any real model has to beat.
#   naive:           repeat the last observed value
#   seasonal naive:  repeat the value from 12 months before the target month
#   drift:           extend the average change seen so far
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6: Simple forecast benchmarks (last 24 months)")
print("=" * 60)

values = y.values
n = len(values)
TEST_MONTHS = 24


def benchmark(horizon):
    errors = {"Naive (last value)": [], "Seasonal naive": [], "Drift": []}
    actuals = []
    for target in range(n - TEST_MONTHS, n):
        origin = target - horizon          # last month we are allowed to see
        if origin < 12:
            continue
        history = values[: origin + 1]
        actual = values[target]
        actuals.append(actual)
        errors["Naive (last value)"].append(actual - history[-1])
        errors["Seasonal naive"].append(actual - values[target - 12])
        drift = history[-1] + horizon * (history[-1] - history[0]) / (len(history) - 1)
        errors["Drift"].append(actual - drift)
    out = []
    for name, e in errors.items():
        e = np.array(e)
        out.append({
            "horizon_months": horizon, "model": name,
            "mae": round(np.mean(np.abs(e)), 0),
            "rmse": round(np.sqrt(np.mean(e ** 2)), 0),
            "mape_pct": round(np.mean(np.abs(e) / np.array(actuals)) * 100, 2),
        })
    return out


bench = pd.DataFrame(sum([benchmark(h) for h in (1, 3, 6)], []))
for h in (1, 3, 6):
    print(f"\nHorizon: {h} month{'s' if h > 1 else ''} ahead")
    print(bench[bench["horizon_months"] == h].drop(columns="horizon_months").to_string(index=False))
bench.to_csv("outputs/benchmark_baselines.csv", index=False)

print("\nSaved: outputs/stationarity_tests.csv, outputs/benchmark_baselines.csv")
print("\nDiagnostics complete. Ready for modeling.")
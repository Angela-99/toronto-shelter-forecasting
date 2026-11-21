# =============================================================
# 05_forecast.py
# Toronto Shelter System Flow - 12-Month Forecast
# =============================================================

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/actively_homeless_by_group.csv"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8985"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ETS_GREEN = "#1baa8f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})

HORIZON = 12
N_SIMS = 2000

# -----------------------------------------------------------------------------
# STEP 1: FIT THE FINAL MODEL ON ALL DATA
# ETS (damped trend, additive seasonality) was at or near the top of the
# backtest at every horizon, so it is the main forecast.
# -----------------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Fitting ETS on the full series")
print("=" * 60)

wide = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
y = wide["All Population"].astype(float)
y.index.freq = "MS"

model = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                             seasonal_periods=12, initialization_method="estimated").fit()
print(f"Data: {y.index.min():%b %Y} to {y.index.max():%b %Y} ({len(y)} months)")
print(f"Damping parameter (phi): {model.params['damping_trend']:.2f}")
print(f"Last observed value: {y.iloc[-1]:,.0f} ({y.index[-1]:%b %Y})")

# -----------------------------------------------------------------------------
# STEP 2: FORECAST WITH PREDICTION INTERVALS
# Holt-Winters has no built-in intervals, so we simulate 2,000 possible futures
# from the fitted model and take the middle 80% and 95% of them each month.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: 12-month forecast with prediction intervals")
print("=" * 60)

point = model.forecast(HORIZON)
sims = model.simulate(HORIZON, repetitions=N_SIMS, error="add", random_state=42)
fc = pd.DataFrame({
    "forecast": point,
    "lo80": sims.quantile(0.10, axis=1), "hi80": sims.quantile(0.90, axis=1),
    "lo95": sims.quantile(0.025, axis=1), "hi95": sims.quantile(0.975, axis=1),
})
# Seasonal naive for comparison: value from the same month one year earlier.
fc["seasonal_naive"] = y.iloc[-12:].values
fc.index.name = "date"
fc = fc.round(0)
print(fc.assign(date_label=fc.index.strftime("%b %Y")).set_index("date_label").to_string())
fc.to_csv("outputs/forecast_12m.csv")

# -----------------------------------------------------------------------------
# STEP 3: HOW TRUSTWORTHY ARE THE INTERVALS?
# Compare the simulated 80% band with the real ETS errors seen in the backtest.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: Interval check against the backtest")
print("=" * 60)

bt = pd.read_csv("outputs/backtest_forecasts.csv")
ets_bt = bt[bt["model"] == "ETS"].copy()
ets_bt["error"] = ets_bt["actual"] - ets_bt["forecast"]
rows = []
for h in range(1, 7):
    e = ets_bt.loc[ets_bt["horizon"] == h, "error"]
    sim_half = (fc["hi80"].iloc[h - 1] - fc["lo80"].iloc[h - 1]) / 2
    emp_half = (e.quantile(0.90) - e.quantile(0.10)) / 2
    rows.append({"months_ahead": h, "simulated_80_half_width": round(sim_half),
                 "backtest_80_half_width": round(emp_half)})
check = pd.DataFrame(rows)
print(check.to_string(index=False))
ratio = (check["backtest_80_half_width"] / check["simulated_80_half_width"]).mean()
print(f"\nBacktest errors were on average {ratio:.1f}x the simulated 80% half-width.")
if ratio > 1.15:
    print("The simulated bands are narrower than real past errors, so treat them as optimistic.")
elif ratio < 0.85:
    print("The simulated bands are wider than real past errors, so they are cautious.")
else:
    print("The simulated bands match real past errors well.")
check.to_csv("outputs/interval_check.csv", index=False)

# -----------------------------------------------------------------------------
# STEP 4: SUMMARY OF THE FORECAST
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Summary")
print("=" * 60)

last = y.iloc[-1]
f6, f12 = fc["forecast"].iloc[5], fc["forecast"].iloc[11]
print(f"Latest actual ({y.index[-1]:%b %Y}):  {last:,.0f}")
print(f"Forecast {fc.index[5]:%b %Y}:        {f6:,.0f}  (80% range {fc['lo80'].iloc[5]:,.0f} to {fc['hi80'].iloc[5]:,.0f})")
print(f"Forecast {fc.index[11]:%b %Y}:        {f12:,.0f}  (80% range {fc['lo80'].iloc[11]:,.0f} to {fc['hi80'].iloc[11]:,.0f})")
print(f"Peak forecast month:       {fc['forecast'].idxmax():%b %Y} ({fc['forecast'].max():,.0f})")
print(f"Lowest forecast month:     {fc['forecast'].idxmin():%b %Y} ({fc['forecast'].min():,.0f})")
print(f"Average forecast vs seasonal naive: {fc['forecast'].mean():,.0f} vs {fc['seasonal_naive'].mean():,.0f}")

# -----------------------------------------------------------------------------
# STEP 5: FIGURE
# -----------------------------------------------------------------------------

hist = y["2022-01-01":]
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(hist.index, hist.values, color=INK, linewidth=2.2, label="Actual")
ax.fill_between(fc.index, fc["lo95"], fc["hi95"], color=ETS_GREEN, alpha=0.12, label="95% range")
ax.fill_between(fc.index, fc["lo80"], fc["hi80"], color=ETS_GREEN, alpha=0.25, label="80% range")
ax.plot(fc.index, fc["forecast"], color=ETS_GREEN, linewidth=2.2, label="ETS forecast")
ax.plot(fc.index, fc["seasonal_naive"], color=GRAY, linewidth=1.6, linestyle="--",
        label="Seasonal naive")
ax.plot([y.index[-1], fc.index[0]], [y.iloc[-1], fc["forecast"].iloc[0]], color=ETS_GREEN, linewidth=2.2)
ax.set_ylabel("People actively homeless")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_title("12-month forecast of the actively homeless count",
             loc="left", fontsize=13, fontweight="bold")
ax.legend(frameon=False, ncol=5, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/15_forecast.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nSaved: outputs/forecast_12m.csv, outputs/interval_check.csv, 15_forecast.png")
print("\nForecasting complete.")
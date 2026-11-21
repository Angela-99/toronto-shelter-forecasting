# =============================================================
# 01_prepare_series.py
# Toronto Shelter System Flow - Time Series Preparation
# =============================================================

import os

import pandas as pd

RAW_PATH = "data/raw/toronto-shelter-system-flow.csv"
LONG_PATH = "data/processed/shelter_monthly.csv"
WIDE_PATH = "data/processed/actively_homeless_by_group.csv"

AGE_COLS = ["age_under16", "age_16_24", "age_25_34", "age_35_44",
            "age_45_54", "age_55_64", "age_65over"]
GENDER_COLS = ["gender_male", "gender_female", "gender_other"]
INFLOWS = ["newly_identified", "returned_to_shelter", "returned_from_housing"]
OUTFLOWS = ["moved_to_housing", "became_inactive"]

# -----------------------------------------------------------------------------
# STEP 1: LOAD AND STANDARDIZE
# -----------------------------------------------------------------------------

print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df = pd.read_csv(RAW_PATH)
print(f"Shape: {df.shape}")

df = df.drop(columns=["_id"]).rename(columns={
    "date(mmm-yy)":                                "date",
    "ageunder16":                                  "age_under16",
    "age16-24":                                    "age_16_24",
    "age25-34":                                    "age_25_34",
    "age35-44":                                    "age_35_44",
    "age45-54":                                    "age_45_54",
    "age55-64":                                    "age_55_64",
    "age65over":                                   "age_65over",
    "gender_transgender,non-binary_or_two_spirit": "gender_other",
    "population_group_percentage":                 "population_group_pct",
})

df["date"] = pd.to_datetime(df["date"], format="%b-%y")
df["population_group_pct"] = df["population_group_pct"].str.replace("%", "", regex=False).astype(float)
df = df.sort_values(["population_group", "date"]).reset_index(drop=True)

print(f"Date range: {df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate group-month rows: {df.duplicated(['date', 'population_group']).sum()}")

# -----------------------------------------------------------------------------
# STEP 2: CHECK THAT EVERY GROUP IS AN UNBROKEN MONTHLY SERIES
# Forecasting models assume equally spaced observations with no gaps.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2: Series coverage by population group")
print("=" * 60)

rows = []
for group, g in df.groupby("population_group"):
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="MS")
    rows.append({
        "population_group": group,
        "months": len(g),
        "first": g["date"].min().strftime("%b %Y"),
        "last": g["date"].max().strftime("%b %Y"),
        "gaps": len(full.difference(g["date"])),
    })
coverage = pd.DataFrame(rows)
print(coverage.to_string(index=False))

short = coverage[coverage["months"] < coverage["months"].max()]
for _, r in short.iterrows():
    print(f"\nNote: {r['population_group']} only starts in {r['first']} ({r['months']} months), "
          "which is too short to compare fairly with the others.")

# -----------------------------------------------------------------------------
# STEP 3: CHECK THE COUNTS ADD UP
# The age and gender breakdowns should each sum to actively_homeless.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3: Do the breakdowns add up to actively_homeless?")
print("=" * 60)

for name, cols in [("Age groups", AGE_COLS), ("Gender groups", GENDER_COLS)]:
    diff = df[cols].sum(axis=1) - df["actively_homeless"]
    off = diff != 0
    print(f"\n{name}: {off.sum()} of {len(df)} rows do not sum exactly "
          f"({off.mean() * 100:.1f}%)")
    if off.any():
        pct = (diff[off].abs() / df.loc[off, "actively_homeless"] * 100)
        print(f"  Largest absolute difference: {diff.abs().max():.0f} people")
        print(f"  Largest relative difference: {pct.max():.2f}% of that month's total")

print("\nThe total (actively_homeless) is the figure we forecast, so small gaps in the")
print("age breakdown do not affect the forecasting target.")

# -----------------------------------------------------------------------------
# STEP 4: STOCK-FLOW CHECK (ALL POPULATION)
# actively_homeless this month should equal last month's total plus inflows
# minus outflows. Small residuals mean the flow columns explain most of the
# month-to-month change.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4: Stock-flow check (All Population)")
print("=" * 60)

allpop = df[df["population_group"] == "All Population"].set_index("date").sort_index()
net_flow = allpop[INFLOWS].sum(axis=1) - allpop[OUTFLOWS].sum(axis=1)
expected = allpop["actively_homeless"].shift(1) + net_flow
residual = (allpop["actively_homeless"] - expected).dropna()
actual_change = allpop["actively_homeless"].diff().dropna()

print(f"Average month-to-month change:      {actual_change.mean():>8.1f}")
print(f"Average absolute change:            {actual_change.abs().mean():>8.1f}")
print(f"Average residual (actual - flows):  {residual.mean():>8.1f}")
print(f"Residual std:                       {residual.std():>8.1f}")
print(f"Average absolute residual:          {residual.abs().mean():>8.1f}")

# -----------------------------------------------------------------------------
# STEP 5: SAVE
# Long format keeps every column. Wide format is one column per group,
# ready for the time series scripts.
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5: Saving prepared data")
print("=" * 60)

df["net_flow"] = df[INFLOWS].sum(axis=1) - df[OUTFLOWS].sum(axis=1)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

os.makedirs("data/processed", exist_ok=True)
df.to_csv(LONG_PATH, index=False)

wide = df.pivot(index="date", columns="population_group", values="actively_homeless")
wide.to_csv(WIDE_PATH)

print(f"Saved: {LONG_PATH}  {df.shape}")
print(f"Saved: {WIDE_PATH}  {wide.shape}")

# -----------------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("SERIES SUMMARY (actively_homeless)")
print("=" * 60)

summary = wide.agg(["count", "min", "mean", "max"]).T.round(0).astype(int)
print(summary.to_string())
print(f"\nMain series: All Population, {wide['All Population'].count()} months, "
      f"{wide.index.min().strftime('%b %Y')} to {wide.index.max().strftime('%b %Y')}")
print("\nSeries preparation complete. Ready for diagnostics.")
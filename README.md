# Toronto Shelter System: Time Series Forecasting

Forecasting the number of people actively experiencing homelessness in Toronto's shelter system, with honest benchmarking against simple baselines.

## Question

How many people will be actively homeless in Toronto over the next 12 months, and do statistical models forecast better than simple rules such as "next month equals this month"?

## Data

City of Toronto Open Data, Toronto Shelter System Flow. Monthly counts by population group, January 2018 to June 2026.

- Main series: All Population, 102 months, `actively_homeless`
- Raw file: `data/raw/toronto-shelter-system-flow.csv` (780 rows, 20 columns)
- Actively homeless means people who used a shelter or other homelessness service in the last 3 months and did not move to housing or become inactive

## Approach

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_prepare_series.py` | Standardizes columns, parses dates, checks every group is an unbroken monthly series, checks that breakdowns add up, saves long and wide datasets |
| 2 | `02_diagnostics.py` | STL decomposition, ADF and KPSS stationarity tests, ACF/PACF, COVID-19 break analysis, naive benchmarks |
| 3 | `03_models.py` | Fits ETS and SARIMA candidates, adds a COVID indicator, checks residuals |
| 4 | `04_backtest.py` | Rolling-origin backtest: 36 forecast origins, 1 to 6 months ahead, all models scored on the same months |
| 5 | `05_forecast.py` | 12-month forecast from the chosen model with simulated prediction intervals |

## Key findings

### The series

- Trend strength is 0.72 and seasonal strength is 0.17. The count is mostly trend with a modest seasonal pattern.
- Seasonally, December and March run about 250-290 people above trend, and September and October about 260-280 below it.
- The level series is not stationary (ADF p = 0.203, KPSS p = 0.01). After differencing the tests were inconclusive, which is expected with a structural break in 2020.

### The COVID-19 break

- The count fell from 9,834 (Feb 2020) to a low of 7,506 (Oct 2020), a drop of 23.7%.
- It was back above the Feb 2020 level by Mar 2022, 17 months after the low.
- The average was 9,495 before COVID, 7,912 during, and 10,170 after.
- A COVID-period indicator in SARIMA was not significant (coefficient 117 people, p = 0.450) and did not improve forecasts.

### Backtest results (36 origins, Jan 2023 to Dec 2025)

| Horizon | Naive MAE | Best model | Its MAE | MAE vs naive |
|---------|-----------|------------|---------|--------------|
| 1 month | 248 | SARIMA | 217 | 0.88 |
| 3 months | 554 | Seasonal naive | 489 | 0.88 |
| 6 months | 786 | Seasonal naive | 514 | 0.65 |

- At 1 month, SARIMA and ETS beat the naive benchmark by about 10-12%.
- Beyond 3 months, seasonal naive matches or beats every fitted model. ETS and SARIMA still beat plain naive by about 10% at 6 months.
- Average MAE across horizons 1 to 6: seasonal naive 500, ETS 500, SARIMA 545, SARIMA with COVID 556, naive 575, drift 609.
- ETS was chosen for the final forecast because it stayed at or near the top at every horizon.

### 12-month forecast (ETS, Jul 2026 to Jun 2027)

- Latest actual: 8,957 (Jun 2026).
- ETS projects a low of about 8,015 in Oct 2026, a rise to about 8,865 in Feb 2027, and about 8,171 in Jun 2027.
- Seasonal naive, which repeats last year's values, sits about 1,500 higher on average. The two lines bracket the likely range.
- Prediction intervals are wide. The 80% range for Jun 2027 is 4,821 to 11,634, so long-range numbers are directional only.
- Simulated 80% intervals were wider than real backtest errors beyond 2 months ahead, so they are cautious.

## Limitations

- The series is short (102 months) and contains one major structural break.
- The backtest covers 36 origins, so differences between models of a few percent are not conclusive.
- Seasonal naive does well at longer horizons partly because the backtest window contains a strong yearly pattern. It cannot anticipate policy or capacity changes.
- Forecasts assume the system keeps behaving as it has recently. Shelter capacity, housing supply, and migration policy can change the count quickly.

## Figures

All charts are in `outputs/figures`:

- `09_stl_decomposition.png`: trend, seasonal, and remainder
- `10_acf_pacf.png`: autocorrelation of monthly changes
- `11_covid_break.png`: the 2020 level shift
- `12_model_fits.png`: in-sample fit of the three models
- `13_backtest_mae.png`: error by forecast horizon
- `14_backtest_forecasts.png`: 3-month-ahead forecasts against actuals
- `15_forecast.png`: the 12-month forecast with intervals

## Project structure

    toronto-shelter-forecasting/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── scripts/
    │   ├── 01_prepare_series.py
    │   ├── 02_diagnostics.py
    │   ├── 03_models.py
    │   ├── 04_backtest.py
    │   └── 05_forecast.py
    ├── outputs/
    │   └── figures/
    ├── requirements.txt
    └── README.md

## How to run

    py -m pip install -r requirements.txt
    py scripts/01_prepare_series.py
    py scripts/02_diagnostics.py
    py scripts/03_models.py
    py scripts/04_backtest.py
    py scripts/05_forecast.py

## Tools

Python, pandas, numpy, scipy, matplotlib, statsmodels.

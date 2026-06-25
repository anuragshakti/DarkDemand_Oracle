# DarkDemand Oracle
### 2026 JLL Hackathon — Accelerate 2030: Pioneer Tomorrow's CRE Solutions with Data & AI

> **"We replace JLL's linear backward-looking models with an agentic platform that injects forward-looking demand signals — so the model catches market turning points before they happen, not after."**

---

## Problem

JLL's current CRE forecasting relies on linear regression-type statistical models that extrapolate from past trends. These models cannot detect non-linear market shocks or turning points before they happen.

**COVID Q2 2020 is the proof:** the linear model showed stability right until the shock hit. By the time the data reflected the downturn, the market had already moved.

JLL brokers, consultants, and investment managers are making decisions on yesterday's signal — missing demand that exists 12–18 months before a lease is ever signed.

---

## Solution

**DarkDemand Oracle** is a two-engine agentic intelligence platform for India commercial real estate.

### Engine 1 — Dark Demand Intelligence (Signal Scanner)

Scans 15 forward-looking signal types to identify corporate tenants likely to sign Mumbai office leases **12–18 months before** the deal is visible in any market data.

**Signals monitored:**
- Hiring velocity: volume and rate of India/Mumbai job postings
- Corporate announcements: GCC setups, expansion press releases
- Financial signals: earnings call mentions of India/APAC expansion
- Leasing velocity proxies: QoQ momentum, vacancy direction, precommit rates

**Output per company per quarter:**
- Dark Demand Score (0–100)
- Composite Sentiment Score per market (forward-looking)
- Natural language explanation via Anthropic Claude API (live at demo)

**Why Engine 1 comes first:** Its composite sentiment score is a required feature input for Engine 2. Without it, Engine 2 is just another backward-looking model.

---

### Engine 2 — AI Rent/Yield Forecasting (XGBoost ML)

XGBoost model trained on 25 years of Mumbai Office data. Engine 1's Dark Demand Score is the primary forward-looking feature — this is what allows the model to catch turning points.

**Markets covered:** Mumbai Office | Chennai Office | Mumbai Retail | Chennai Retail

**Targets predicted:**

| Variable | Description |
|---|---|
| `gross_rent` | INR/sqft/month |
| `net_rent` | INR/sqm/year |
| `capital_value_inr` | INR/sqft |
| `yield_gross` | Market yield (%) |
| `vacancy_rate` | Decimal |
| `absorption` | sqm |

**Backtest results (Test set 2023–2024 vs JLL linear baseline):**

| Target | ML RMSE | JLL Baseline RMSE | Improvement |
|---|---|---|---|
| Gross Rent | 23.6 | 152.7 | **+84.5%** |
| Net Rent | 2,854 | 19,494 | **+85.4%** |
| Capital Value | 3,798 | 17,782 | **+78.6%** |
| Yield | 0.003 | 0.011 | **+77.7%** |
| Vacancy Rate | 0.008 | 0.025 | **+68.2%** |
| Absorption | 28,242 | 111,395 | **+74.6%** |

68–85% RMSE improvement across all six targets.

---

## The Killer Demo

Three lines. One chart. The entire pitch.

1. **JLL Linear Forecast** — smooth extrapolation, misses all volatility
2. **Actual Outcome** — real market data including the COVID shock
3. **ML + Dark Demand Score** — tracks actual closely, catches the COVID drop early

The gap between line 1 and line 3 at Q2 2020 is the problem DarkDemand Oracle solves.

---

## Business Impact

| JLL Business Line | Value |
|---|---|
| **Leasing** | Brokers win mandates 12–18 months before competitors see the signal |
| **Advisory & Consulting** | Statistically rigorous research reports backed by non-linear ML |
| **Capital Markets** | Accurate yield forecasts for investment committees |
| **LaSalle Investment Management** | Portfolio allocation via predictive demand signals |
| **JLL Technologies** | Productisable across all APAC markets with zero re-engineering |

---

## Scalability

Chennai Office was added with **one config change and zero re-engineering**. All JLL REIS workbooks follow an identical data structure — DarkDemand Oracle covers JLL's entire APAC data estate.

---

## Repo Structure

```
darkdemand-oracle/
├── README.md                          ← This file
├── run_pipeline.py                    ← Master orchestrator
│
├── engine1_signal_scanner/
│   ├── Senti_app.py                   ← Streamlit broker dashboard
│   ├── Senti_config.py                ← API keys, market config (update Okta token before demo)
│   ├── Senti_database.py              ← SQLite schema and operations
│   ├── Senti_pipeline.py              ← Signal collection pipeline
│   ├── Senti_scraper.py               ← Selenium scraper (visible Chrome only)
│   ├── Senti_jll_gpt.py               ← Anthropic Claude API integration
│   ├── Signal_Scanner.py              ← Entry point wrapper
│   └── requirements.txt
│
└── engine2_forecasting/
    ├── data/
    │   └── darkdemand_oracle.db        ← 383 rows, 4 markets, 6 targets
    ├── models/                         ← Trained XGBoost models (JSON) + imputers (pkl)
    ├── outputs/
    │   ├── killer_chart_mumbai_gross_rent.png
    │   ├── backtest_metrics.csv
    │   └── model_metadata.json
    ├── ML_train_xgboost.py             ← Model training (all 6 targets)
    ├── ML_forecast.py                  ← Forward forecast (next quarter)
    ├── ML_dashboard.py                 ← Streamlit consultant/advisor dashboard
    ├── ML_fix_baseline.py              ← Linear baseline correction utility
    ├── home.py                         ← Combined home screen
    └── requirements.txt
```

---

## Quick Start

### Prerequisites
```
Python 3.11+
Chrome browser (visible, not headless)
Anthropic API key
```

### Install
```bash
pip install -r engine1_signal_scanner/requirements.txt
pip install -r engine2_forecasting/requirements.txt
```

### Run Engine 1 (Signal Scanner)
```bash
# Update Okta token in engine1_signal_scanner/Senti_config.py first
streamlit run engine1_signal_scanner/Senti_app.py
```

### Run Engine 2 (Forecasting Dashboard)
```bash
streamlit run engine2_forecasting/ML_dashboard.py
```

### Run Both (Orchestrator)
```bash
python run_pipeline.py
```

### Generate a forecast
```bash
python engine2_forecasting/ML_forecast.py --market Mumbai_Office
```

---

## Network Requirements

Accessible on JLL network: `*.jll.com`, `anthropic.com`, `github.com`, `pypi.org`

**Note:** Chrome must run in visible mode — Google blocks headless browsers on JLL network. Okta tokens in `Senti_config.py` expire approximately hourly and must be refreshed manually before running the signal scanner.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, scikit-learn, pandas, numpy |
| Signal scraping | Selenium (visible Chrome), SQLite |
| AI explanation | Anthropic Claude API |
| Dashboard | Streamlit, matplotlib |
| Data | SQLite (`darkdemand_oracle.db`) |
| Dependencies | pypi.org only — no external infrastructure |

---

## Problem Statement

*We are building an agentic intelligence platform to solve the problem of reactive deal-making and inaccurate market forecasting for JLL brokers, consultants, and investment managers, so they can identify corporate demand signals 12–18 months before lease signings and generate statistically rigorous rent/yield forecasts — directly expanding JLL's market share and advisory margins.*

---

*2026 JLL Hackathon | APAC Region | Team DarkDemand Oracle*

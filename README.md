# DarkDemand Oracle
### 2026 JLL Hackathon — Accelerate 2030: Pioneer Tomorrow's CRE Solutions with Data & AI

> **"We replace JLL's linear backward-looking models with an agentic platform that injects forward-looking demand signals — so the model catches market turning points before they happen, not after."**

---

## Problem

JLL's current CRE forecasting relies on linear regression-type statistical models that extrapolate from past trends. They cannot detect non-linear market shocks or turning points before they happen.

**COVID Q2 2020 is the proof:** the linear model showed stability right until the shock hit. By the time the data reflected the downturn, the market had already moved.

---

## Solution

**DarkDemand Oracle** is a two-engine agentic intelligence platform for India commercial real estate.

### Engine 1 — Dark Demand Intelligence (Signal Scanner)
Scans 15 forward-looking signal types to identify corporate tenants likely to sign Mumbai office leases **12–18 months before** the deal is visible in any market data. Output: Dark Demand Score (0–100) + Claude AI explanation per company.

### Engine 2 — AI Rent/Yield Forecasting (XGBoost ML)
XGBoost trained on 25 years of data across 4 markets. Engine 1's composite sentiment score is the primary forward-looking feature — this is what catches turning points early.

**Backtest results (2023–2024 vs JLL linear baseline):**

| Target | Improvement |
|---|---|
| Gross Rent | +84.5% RMSE |
| Net Rent | +85.4% RMSE |
| Capital Value | +78.6% RMSE |
| Yield | +77.7% RMSE |
| Vacancy Rate | +68.2% RMSE |
| Absorption | +74.6% RMSE |

---

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run home.py
```

Opens at `http://localhost:8501` — animated landing page with navigation to both engines.

> **Note:** Always use `streamlit run home.py` — do NOT use `python home.py`

### ⚠️ Before running Engine 1
Update the Okta token in `Senti_config.py` — tokens expire approximately every hour.

---

## Repo Structure

All files are flat in the root directory.

```
DarkDemand_Oracle/
├── home.py                  ← MAIN ENTRY POINT — streamlit run home.py
├── run_pipeline.py          ← Alternative launcher (auto-opens browser)
│
├── Senti_app.py             ← Engine 1 · Broker signal dashboard
├── Senti_config.py          ← Engine 1 · Config (update Okta token before demo)
├── Senti_database.py        ← Engine 1 · SQLite operations
├── Senti_pipeline.py        ← Engine 1 · Signal collection pipeline
├── Senti_scraper.py         ← Engine 1 · Selenium scraper (visible Chrome only)
├── Senti_jll_gpt.py         ← Engine 1 · Anthropic Claude API integration
├── Signal_Scanner.py        ← Engine 1 · Entry point wrapper
│
├── ML_dashboard.py          ← Engine 2 · Consultant/advisor dashboard
├── ML_train_xgboost.py      ← Engine 2 · Model training (all 6 targets)
├── ML_forecast.py           ← Engine 2 · Forward forecast (next quarter)
├── ML_fix_baseline.py       ← Engine 2 · Linear baseline utility
│
├── darkdemand_oracle.db     ← 383 rows, 4 markets, 6 targets
├── dark_demand.db           ← Engine 1 signals database
│
├── models/                  ← Trained XGBoost models + imputers
│   ├── *_model.json × 6
│   └── *_imputer.pkl × 6
│
├── outputs/
│   ├── killer_chart_mumbai_gross_rent.png
│   ├── backtest_metrics.csv
│   └── model_metadata.json
│
└── requirements.txt
```

---

## Business Impact

| JLL Business Line | Value |
|---|---|
| **Leasing** | Win mandates 12–18 months before competitors |
| **Advisory & Consulting** | Statistically rigorous research reports |
| **Capital Markets** | Accurate yield forecasts for investment committees |
| **LaSalle Investment Management** | Portfolio allocation via predictive signals |
| **JLL Technologies** | Productisable across all APAC markets, zero re-engineering |

**Scalability:** Chennai Office added with one config change, zero re-engineering.

---

## Problem Statement

*We are building an agentic intelligence platform to solve the problem of reactive deal-making and inaccurate market forecasting for JLL brokers, consultants, and investment managers, so they can identify corporate demand signals 12–18 months before lease signings and generate statistically rigorous rent/yield forecasts — directly expanding JLL's market share and advisory margins.*

---

*2026 JLL Hackathon | APAC Region | Team DarkDemand Oracle*

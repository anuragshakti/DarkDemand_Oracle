"""
DarkDemand Oracle — Forward Forecast
Usage:
  python forecast.py                          # all markets, next quarter only
  python forecast.py --market Mumbai_Office   # specific market, next quarter
  python forecast.py --quarter 2026Q3         # specific quarter (all markets)
  python forecast.py --market Mumbai_Office --quarter 2026Q3
"""

import sqlite3, warnings, argparse, sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import xgboost as xgb

warnings.filterwarnings('ignore')

BASE      = Path(__file__).parent
DB_PATH   = BASE / 'darkdemand_oracle.db'
MODEL_DIR = BASE / 'models'

TARGETS = ['gross_rent','net_rent','capital_value_inr',
           'yield_gross','vacancy_rate','absorption']

FEATURES = [
    'market_enc','quarter_enc','year_norm',
    'gross_rent_lag1','gross_rent_lag4',
    'net_rent_lag1','capital_value_inr_lag1',
    'yield_gross_lag1','vacancy_rate_lag1','absorption_lag1',
    'leasing_velocity_qoq','vacancy_change','absorption_ratio',
    'new_supply','precommit_pct','deal_count','total_leased_sqft',
    'composite_sentiment_score',
]

MARKET_ENC  = {'Chennai_Office':0,'Chennai_Retail':1,'Mumbai_Office':2,'Mumbai_Retail':3}
QUARTER_ENC = {'1Q':0,'2Q':1,'3Q':2,'4Q':3}

ALL_MARKETS = ['Mumbai_Office','Chennai_Office','Mumbai_Retail','Chennai_Retail']


def load_models():
    models, imputers = {}, {}
    for t in TARGETS:
        m = xgb.XGBRegressor()
        m.load_model(str(MODEL_DIR / f'{t}_model.json'))
        models[t]   = m
        imputers[t] = joblib.load(str(MODEL_DIR / f'{t}_imputer.pkl'))
    return models, imputers


def next_quarter(year, quarter):
    """Return (year, quarter) for the quarter after the given one."""
    q_map = {'1Q':'2Q','2Q':'3Q','3Q':'4Q','4Q':'1Q'}
    nq = q_map[quarter]
    ny = year + 1 if quarter == '4Q' else year
    return ny, nq


def forecast_one(market, qtr, yr, models, imputers):
    """Predict a single quarter for a single market."""
    conn = sqlite3.connect(str(DB_PATH))

    # Get seed data — last 4 quarters before target quarter
    seed = pd.read_sql(f"""
        SELECT year, quarter, gross_rent, net_rent, capital_value_inr,
               yield_gross, vacancy_rate, absorption,
               composite_sentiment_score, leasing_velocity_qoq,
               vacancy_change, absorption_ratio, new_supply,
               precommit_pct, deal_count, total_leased_sqft
        FROM market_features
        WHERE market='{market}'
        AND (year < {yr} OR (year={yr} AND quarter < '{qtr}'))
        ORDER BY year DESC, quarter DESC LIMIT 4
    """, conn)
    conn.close()

    if seed.empty:
        print(f"  No seed data for {market} before {yr}{qtr}")
        return None

    last = seed.iloc[0]  # most recent row
    lag4 = seed.iloc[3] if len(seed) >= 4 else last  # 4 quarters ago

    sentiment = float(last['composite_sentiment_score'] or 35.0)

    feat = [
        MARKET_ENC.get(market, 0), QUARTER_ENC.get(qtr, 0),
        (yr - 2001) / (2030 - 2001),
        float(last['gross_rent']        or 0),
        float(lag4['gross_rent']        or 0),
        float(last['net_rent']          or 0),
        float(last['capital_value_inr'] or 0),
        float(last['yield_gross']       or 0),
        float(last['vacancy_rate']      or 0),
        float(last['absorption']        or 0),
        float(last['leasing_velocity_qoq'] or 0),
        float(last['vacancy_change']       or 0),
        float(last['absorption_ratio']     or 1),
        float(last['new_supply']           or 0),
        float(last['precommit_pct']        or 0),
        float(last['deal_count']           or 0),
        float(last['total_leased_sqft']    or 0),
        sentiment,
    ]

    X     = np.array(feat).reshape(1, -1)
    preds = {t: round(float(models[t].predict(imputers[t].transform(X))[0]), 4)
             for t in TARGETS}

    return {
        'market': market, 'quarter': qtr, 'year': yr,
        'gross_rent_predicted':        preds['gross_rent'],
        'net_rent_predicted':          preds['net_rent'],
        'capital_value_inr_predicted': preds['capital_value_inr'],
        'yield_gross_predicted':       preds['yield_gross'],
        'vacancy_rate_predicted':      preds['vacancy_rate'],
        'absorption_predicted':        preds['absorption'],
        'gross_rent_actual': None, 'net_rent_actual': None,
        'capital_value_inr_actual': None, 'yield_gross_actual': None,
        'vacancy_rate_actual': None, 'absorption_actual': None,
        'composite_sentiment_score': sentiment,
        'data_status': 'predicted',
    }


def save_forecast(record):
    """Insert one forecast row. Skip if already exists."""
    conn = sqlite3.connect(str(DB_PATH))

    # Ensure forecasts table exists
    conn.execute("""CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market TEXT, quarter TEXT, year INTEGER,
        gross_rent_predicted REAL, net_rent_predicted REAL,
        capital_value_inr_predicted REAL, yield_gross_predicted REAL,
        vacancy_rate_predicted REAL, absorption_predicted REAL,
        gross_rent_actual REAL, net_rent_actual REAL,
        capital_value_inr_actual REAL, yield_gross_actual REAL,
        vacancy_rate_actual REAL, absorption_actual REAL,
        composite_sentiment_score REAL,
        data_status TEXT DEFAULT 'predicted',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    # Skip if already exists
    exists = conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE market=? AND year=? AND quarter=?",
        (record['market'], record['year'], record['quarter'])
    ).fetchone()[0]

    if exists:
        conn.close()
        return False

    conn.execute("""INSERT INTO forecasts
        (market,quarter,year,
         gross_rent_predicted,net_rent_predicted,capital_value_inr_predicted,
         yield_gross_predicted,vacancy_rate_predicted,absorption_predicted,
         gross_rent_actual,net_rent_actual,capital_value_inr_actual,
         yield_gross_actual,vacancy_rate_actual,absorption_actual,
         composite_sentiment_score,data_status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (record['market'], record['quarter'], record['year'],
         record['gross_rent_predicted'], record['net_rent_predicted'],
         record['capital_value_inr_predicted'], record['yield_gross_predicted'],
         record['vacancy_rate_predicted'], record['absorption_predicted'],
         record['gross_rent_actual'], record['net_rent_actual'],
         record['capital_value_inr_actual'], record['yield_gross_actual'],
         record['vacancy_rate_actual'], record['absorption_actual'],
         record['composite_sentiment_score'], record['data_status']))
    conn.commit()
    conn.close()
    return True


def parse_period(period_str):
    """Parse '2026Q3' into (2026, '3Q')."""
    yr  = int(period_str[:4])
    qn  = period_str[5]
    qtr = qn + 'Q'
    return yr, qtr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--market',  default=None)
    parser.add_argument('--quarter', default=None,
                        help="Period string e.g. 2026Q3")
    parser.add_argument('--preview-only', action='store_true',
                        help="Write ONLY to next_quarter_preview — never touches forecasts table")
    args = parser.parse_args()

    print("DarkDemand Oracle — Forecast")

    models, imputers = load_models()
    markets = [args.market] if args.market else ALL_MARKETS

    for market in markets:
        if args.quarter:
            yr, qtr = parse_period(args.quarter)
        else:
            # Default: next quarter after latest actual for this market
            conn = sqlite3.connect(str(DB_PATH))
            row  = conn.execute(
                f"SELECT year, quarter FROM market_features "
                f"WHERE market='{market}' AND gross_rent IS NOT NULL "
                f"ORDER BY year DESC, quarter DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                print(f"  {market}: no historical data found")
                continue
            yr, qtr = next_quarter(row[0], row[1])

        rec = forecast_one(market, qtr, yr, models, imputers)
        if rec:
            if args.preview_only:
                # Preview mode: write ONLY to next_quarter_preview, never to forecasts
                _pc = sqlite3.connect(str(DB_PATH))
                _pc.execute("CREATE TABLE IF NOT EXISTS next_quarter_preview ("
                            "market TEXT, year INTEGER, quarter TEXT,"
                            "gross_rent_predicted REAL, net_rent_predicted REAL,"
                            "capital_value_inr_predicted REAL, yield_gross_predicted REAL,"
                            "vacancy_rate_predicted REAL, absorption_predicted REAL)")
                _pc.execute("DELETE FROM next_quarter_preview WHERE market=?", (market,))
                _pc.execute(
                    "INSERT INTO next_quarter_preview "
                    "(market,year,quarter,gross_rent_predicted,net_rent_predicted,"
                    "capital_value_inr_predicted,yield_gross_predicted,"
                    "vacancy_rate_predicted,absorption_predicted) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (market, rec['year'], rec['quarter'],
                     rec['gross_rent_predicted'], rec['net_rent_predicted'],
                     rec['capital_value_inr_predicted'], rec['yield_gross_predicted'],
                     rec['vacancy_rate_predicted'], rec['absorption_predicted']))
                _pc.commit(); _pc.close()
                print(f"  {market} {yr}{qtr}: preview ₹{rec['gross_rent_predicted']:.2f} → NQP only")
            else:
                saved = save_forecast(rec)
                status = "saved" if saved else "already exists"
                print(f"  {market} {yr}{qtr}: {rec['gross_rent_predicted']:.4f} ({status})")


if __name__ == '__main__':
    main()
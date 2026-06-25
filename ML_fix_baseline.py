"""
One-time fix — corrects linear regression baseline values in predictions table.
Run once: python fix_baseline.py
"""
from pathlib import Path
import sqlite3, pandas as pd
from sklearn.linear_model import LinearRegression

DB = Path(__file__).parent / 'darkdemand_oracle.db'
conn = sqlite3.connect(str(DB))

TARGETS = ['gross_rent','net_rent','capital_value_inr',
           'yield_gross','vacancy_rate','absorption']

fixed = 0
for market in ['Mumbai_Office','Chennai_Office','Mumbai_Retail','Chennai_Retail']:
    for target in TARGETS:
        h = pd.read_sql(f"""SELECT year, quarter, {target}
            FROM market_features WHERE market='{market}'
            AND year<=2022 AND {target} IS NOT NULL
            ORDER BY year, quarter""", conn)
        if len(h) < 4: continue
        h['t'] = range(len(h))
        lr = LinearRegression().fit(h[['t']].values, h[target].values)
        n  = len(h)
        rows = pd.read_sql(f"""SELECT year, quarter FROM predictions
            WHERE market='{market}' AND target='{target}'
            ORDER BY year, quarter""", conn)
        for i, row in enumerate(rows.itertuples()):
            val = float(lr.predict([[n+i]])[0])
            conn.execute(f"""UPDATE predictions SET baseline=?
                WHERE market='{market}' AND target='{target}'
                AND year={row.year} AND quarter='{row.quarter}'""", (val,))
            fixed += 1

conn.commit()
conn.close()
print(f"✅ Fixed {fixed} baseline values. Run: streamlit run dashboard.py")

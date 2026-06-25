"""
DarkDemand Oracle — XGBoost Training
=====================================
Reads darkdemand_oracle.db (must be in same folder).
Trains 6 models. Saves to models/ and outputs/.
Run: python train_xgboost.py
"""

import sqlite3, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

BASE       = Path(__file__).parent
DB_PATH    = BASE / 'darkdemand_oracle.db'
MODEL_DIR  = BASE / 'models';  MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = BASE / 'outputs'; OUTPUT_DIR.mkdir(exist_ok=True)

TARGETS = ['gross_rent', 'net_rent', 'capital_value_inr',
           'yield_gross', 'vacancy_rate', 'absorption']

FEATURES = [
    'market_enc', 'quarter_enc', 'year_norm',
    'gross_rent_lag1', 'gross_rent_lag4',
    'net_rent_lag1', 'capital_value_inr_lag1',
    'yield_gross_lag1', 'vacancy_rate_lag1', 'absorption_lag1',
    'leasing_velocity_qoq', 'vacancy_change', 'absorption_ratio',
    'new_supply', 'precommit_pct', 'deal_count', 'total_leased_sqft',
    'composite_sentiment_score',
]

XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
)


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM market_features ORDER BY market, year, quarter", conn)
    conn.close()
    print(f"  Loaded {len(df)} rows from {DB_PATH.name}")
    return df


def engineer_features(df):
    df = df.copy()
    le_mkt = LabelEncoder()
    le_qtr = LabelEncoder()
    df['market_enc']  = le_mkt.fit_transform(df['market'])
    df['quarter_enc'] = le_qtr.fit_transform(df['quarter'])
    df['year_norm']   = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    df = df.sort_values(['market', 'year', 'quarter']).reset_index(drop=True)
    for mkt in df['market'].unique():
        mask = df['market'] == mkt
        for col, lags in [('gross_rent', [1, 4]), ('net_rent', [1]),
                          ('capital_value_inr', [1]), ('yield_gross', [1]),
                          ('vacancy_rate', [1]), ('absorption', [1])]:
            for lag in lags:
                df.loc[mask, f'{col}_lag{lag}'] = df.loc[mask, col].shift(lag)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
    return df


def jll_baseline(df_train, df_eval, target):
    """
    Linear regression baseline — fits a trend on all historical data
    and extrapolates forward. This is what JLL's backward-looking
    linear model does: it sees past trends and draws a straight line.
    """
    from sklearn.linear_model import LinearRegression
    valid = df_train[df_train[target].notna()].copy()
    valid = valid.sort_values(['year','quarter']).reset_index(drop=True)
    valid['t'] = range(len(valid))
    lr = LinearRegression()
    lr.fit(valid[['t']].values, valid[target].values)
    n = len(valid)
    preds = [float(lr.predict([[n + i]])[0])
             for i in range(len(df_eval))]
    return np.array(preds)


def train_and_evaluate(df):
    df_train = df[df['year'] <= 2022].copy()
    df_test  = df[(df['year'] >= 2023) & (df['year'] <= 2024)].copy()
    df_val   = df[df['year'] >= 2025].copy()
    print(f"  Train: {len(df_train)} | Test: {len(df_test)} | Val: {len(df_val)}")

    X_train = df_train[FEATURES].values
    X_test  = df_test[FEATURES].values
    X_val   = df_val[FEATURES].values
    results, preds_db = {}, []

    for target in TARGETS:
        y_train = df_train[target].values
        y_test  = df_test[target].values
        y_val   = df_val[target].values
        valid   = ~np.isnan(y_train)
        if valid.sum() < 20:
            print(f"  Skipping {target}"); continue

        imp  = SimpleImputer(strategy='median')
        Xtr  = imp.fit_transform(X_train[valid])
        Xte  = imp.transform(X_test)
        Xvl  = imp.transform(X_val)
        y_tr = y_train[valid]
        y_te = pd.Series(y_test).fillna(pd.Series(y_test).median()).values
        y_vl = pd.Series(y_val).fillna(pd.Series(y_val).median()).values

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(Xtr, y_tr, eval_set=[(Xte, y_te)], verbose=False)

        pred_test = model.predict(Xte)
        pred_val  = model.predict(Xvl)
        base_test = jll_baseline(df_train[df_train[target].notna()], df_test, target)

        rmse_ml   = np.sqrt(mean_squared_error(y_te, pred_test))
        rmse_base = np.sqrt(mean_squared_error(y_te, base_test))
        nz_te     = y_te != 0
        mape_ml   = mean_absolute_percentage_error(y_te[nz_te], pred_test[nz_te]) * 100 if nz_te.sum() > 0 else 0
        nz_vl     = y_vl != 0
        mape_val  = mean_absolute_percentage_error(y_vl[nz_vl], pred_val[nz_vl]) * 100 if nz_vl.sum() > 0 else 0
        rmse_val  = np.sqrt(mean_squared_error(y_vl, pred_val))
        fi        = dict(zip(FEATURES, model.feature_importances_))
        top_feat  = sorted(fi.items(), key=lambda x: -x[1])[:3]
        impr      = round((1 - rmse_ml / rmse_base) * 100, 1)

        results[target] = {
            'rmse_ml_test': round(rmse_ml, 4), 'rmse_base_test': round(rmse_base, 4),
            'mape_ml_test_pct': round(mape_ml, 2), 'rmse_improvement_pct': impr,
            'rmse_val': round(rmse_val, 4), 'mape_val_pct': round(mape_val, 2),
            'top_features': top_feat,
            'pred_test': pred_test, 'pred_val': pred_val,
            'y_test': y_te, 'baseline_test': base_test,
        }

        model.save_model(str(MODEL_DIR / f'{target}_model.json'))
        joblib.dump(imp, str(MODEL_DIR / f'{target}_imputer.pkl'))

        for i, (_, row) in enumerate(df_test.iterrows()):
            preds_db.append({'market': row['market'], 'quarter': row['quarter'],
                'year': row['year'], 'target': target, 'split': 'test',
                'predicted': float(pred_test[i]), 'actual': float(y_te[i]),
                'baseline': float(base_test[i])})
        for i, (_, row) in enumerate(df_val.iterrows()):
            preds_db.append({'market': row['market'], 'quarter': row['quarter'],
                'year': row['year'], 'target': target, 'split': 'validation',
                'predicted': float(pred_val[i]), 'actual': float(y_vl[i]),
                'baseline': None})

        print(f"  {target:25s}  +{impr}% vs JLL  (RMSE: {rmse_ml:.3f} vs {rmse_base:.3f})")

    return results, preds_db


def plot_killer_chart(df, results):
    target = 'gross_rent'
    if target not in results: return
    mkt   = df[df['market'] == 'Mumbai_Office'].sort_values(['year', 'quarter'])
    r     = results[target]
    hist  = mkt[mkt['year'] <= 2022]
    bt    = mkt[(mkt['year'] >= 2023) & (mkt['year'] <= 2024) & mkt[target].notna()]
    val   = mkt[mkt['year'] >= 2025]
    base_bt  = jll_baseline(hist[hist[target].notna()], bt,  target)
    base_val = jll_baseline(hist[hist[target].notna()], val, target)

    fig, (ax, ax_fi) = plt.subplots(2, 1, figsize=(14, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117'); ax_fi.set_facecolor('#0d1117')

    h_x  = list(range(len(hist)))
    bt_x = list(range(len(hist), len(hist) + len(bt)))
    v_x  = list(range(len(hist) + len(bt), len(hist) + len(bt) + len(val)))

    if bt_x:
        ax.axvspan(bt_x[0]-0.5, bt_x[-1]+0.5, alpha=0.12, color='#ffe066')
        ax.axvline(bt_x[0]-0.5, color='#444', linewidth=0.8, linestyle='--')
    if v_x:
        ax.axvline(v_x[0]-0.5, color='#e63946', linewidth=0.8, linestyle='--', alpha=0.5)

    act_x = h_x + bt_x + v_x
    act_y = (list(hist[target].values) + list(bt[target].values) + list(val[target].values))
    ax.plot(act_x, act_y, color='#e6edf3', linewidth=1.8, label='Actual', zorder=3)

    bx = [h_x[-1]] + bt_x + v_x
    by = ([hist[target].values[-1]] + list(r['pred_test'][:len(bt)]) + list(r['pred_val'][:len(val)]))
    ax.plot(bx[:len(bt_x)+1], by[:len(bt_x)+1], color='#e63946', linewidth=2.0,
            label='DarkDemand Oracle', zorder=4)
    ax.plot(bx[len(bt_x):], by[len(bt_x):], color='#e63946', linewidth=2.0,
            linestyle=':', zorder=4)

    base_x = [h_x[-1]] + bt_x + v_x
    base_y = [hist[target].values[-1]] + list(base_bt) + list(base_val)
    ax.plot(base_x, base_y, color='#666', linewidth=1.2, linestyle=':',
            label='JLL Linear Baseline', zorder=2)

    for i, (_, row) in enumerate(mkt.iterrows()):
        if row['year'] == 2020 and str(row['quarter']) == '2Q' and i < len(act_y):
            ax.annotate('COVID-19\nShock', xy=(i, act_y[i]),
                xytext=(i-6, act_y[i]-15),
                arrowprops=dict(arrowstyle='->', color='#d29922', lw=1.2),
                fontsize=8, color='#d29922', fontweight='bold'); break

    if h_x:  ax.text(np.mean(h_x),  ax.get_ylim()[1]*0.97, 'TRAINING',
                     ha='center', fontsize=7, color='#666', fontfamily='monospace')
    if bt_x: ax.text(np.mean(bt_x), ax.get_ylim()[1]*0.97, 'BACKTEST',
                     ha='center', fontsize=7, color='#d29922', fontfamily='monospace')
    if v_x:  ax.text(np.mean(v_x),  ax.get_ylim()[1]*0.97, 'FORECAST →',
                     ha='center', fontsize=7, color='#e63946', fontfamily='monospace')

    tick_pos  = list(range(0, len(act_x), 8))
    tick_lbls = [str(int(mkt.iloc[i]['year'])) for i in tick_pos if i < len(mkt)]
    ax.set_xticks(tick_pos[:len(tick_lbls)])
    ax.set_xticklabels(tick_lbls, fontsize=7, color='#8b949e', rotation=45)
    ax.set_ylabel('Gross Rent (INR/sq.ft/month)', fontsize=9, color='#8b949e')
    ax.set_title('Mumbai Office — Gross Rent\nDarkDemand Oracle vs JLL Traditional Method',
                 fontsize=12, fontweight='bold', color='#e6edf3')
    ax.legend(fontsize=8, labelcolor='#e6edf3', framealpha=0.0)
    ax.yaxis.grid(True, color='#21262d', linewidth=0.5)
    ax.tick_params(axis='y', colors='#8b949e')
    for sp in ax.spines.values(): sp.set_color('#21262d')

    fi_lbls = [f[0].replace('_',' ')[:22] for f in r['top_features']]
    fi_vals = [f[1] for f in r['top_features']]
    colours = ['#e63946' if 'sentiment' in l else '#58a6ff' for l in fi_lbls]
    ax_fi.barh(fi_lbls, fi_vals, color=colours)
    ax_fi.set_title('Top Feature Importances (Gross Rent)', fontsize=9, color='#e6edf3')
    ax_fi.tick_params(colors='#8b949e', labelsize=7)
    ax_fi.set_xlabel('Importance', fontsize=8, color='#8b949e')
    for sp in ax_fi.spines.values(): sp.set_color('#21262d')
    ax_fi.xaxis.grid(True, color='#21262d', linewidth=0.5)

    plt.tight_layout(pad=2.0)
    out = OUTPUT_DIR / 'killer_chart_mumbai_gross_rent.png'
    plt.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Killer chart → {out}")


def write_predictions(preds_db):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.execute("""CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        market TEXT, quarter TEXT, year INTEGER,
        target TEXT, split TEXT,
        predicted REAL, actual REAL, baseline REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.executemany("""INSERT INTO predictions
        (market,quarter,year,target,split,predicted,actual,baseline)
        VALUES (:market,:quarter,:year,:target,:split,:predicted,:actual,:baseline)""",
        preds_db)
    conn.commit(); conn.close()


def main():
    print("="*60)
    print("DarkDemand Oracle — XGBoost Training")
    print("="*60)

    df = load_data()

    print("\n[1/3] Engineering features...")
    df = engineer_features(df)

    print("\n[2/3] Training models...")
    results, preds_db = train_and_evaluate(df)

    print("\n[3/3] Saving outputs...")
    write_predictions(preds_db)
    print(f"  {len(preds_db)} predictions → DB")
    plot_killer_chart(df, results)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for t, r in results.items():
        print(f"  {t:25s}  +{r['rmse_improvement_pct']}% better than JLL")

    meta = {t: {k: v for k, v in r.items()
                if k not in ['pred_test','pred_val','y_test','baseline_test']}
            for t, r in results.items()}
    for t, v in meta.items():
        v['top_features'] = [(f, round(float(s),4)) for f,s in v['top_features']]
    with open(str(OUTPUT_DIR / 'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    pd.DataFrame([{
        'target': t, 'rmse_ml': r['rmse_ml_test'], 'rmse_jll': r['rmse_base_test'],
        'improvement_pct': r['rmse_improvement_pct'], 'mape_val_pct': r['mape_val_pct']}
        for t, r in results.items()
    ]).to_csv(str(OUTPUT_DIR / 'backtest_metrics.csv'), index=False)

    print(f"\n✅ models/   → {MODEL_DIR}")
    print(f"✅ outputs/  → {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
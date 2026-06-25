"""
DarkDemand Oracle — Engine 2 Dashboard
JLL Brand. Magic wand logo top-left. Big numbers. Clean cards.
"""

import streamlit as st
import sqlite3, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")
from pathlib import Path
from sklearn.linear_model import LinearRegression
import streamlit.components.v1 as components

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
DB_PATH   = BASE / 'darkdemand_oracle.db'
META_PATH = BASE / 'outputs' / 'model_metadata.json'

TARGETS = {
    'gross_rent':        ('Gross Rent',    'INR/sq.ft/mo'),
    'net_rent':          ('Net Rent',      'INR/sq.m/yr'),
    'capital_value_inr': ('Capital Value', 'INR/sq.ft'),
    'yield_gross':       ('Market Yield',  '%'),
    'vacancy_rate':      ('Vacancy Rate',  '%'),
    'absorption':        ('Net Absorption','sq.m'),
}
T_LIST   = list(TARGETS.keys())
QUARTERS = ['1Q','2Q','3Q','4Q']
YEARS_FC = list(range(2023, 2031))  # 2023-2030 covers backtest + forecast
FROM_YR  = 2019

J = dict(
    red='#E30613', space='#003E51', night='#131E29',
    ocean='#BCDEE6', white='#FFFFFF',
    s50='#F3F4F4', s200='#DEDFE1', s400='#A6AAAE', s600='#5A6169',
    ok='#1B7A3E', ok_bg='#D4EDDA',
    warn='#7D4000', warn_bg='#FFF3CD',
    info='#005D76',
)
LC = dict(
    hist='#000000',   # Actual — pure black
    xgb='#1B7A3E',    # XGBoost — green
    lr='#F5A623',     # Linear Regression — amber dashed
    fc='#005D76',     # Forecast — teal dotted
    act='#1B7A3E',
    ctx='#DEDFE1',
)

@st.cache_resource
def conn():
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts (
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS next_quarter_preview (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT, year INTEGER, quarter TEXT,
            gross_rent_predicted REAL, net_rent_predicted REAL,
            capital_value_inr_predicted REAL, yield_gross_predicted REAL,
            vacancy_rate_predicted REAL, absorption_predicted REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    c.commit()
    return c

def qr(sql):  return pd.read_sql(sql, conn())
def qw(s, p): c=conn(); c.execute(s,p); c.commit()

def fq(y, q):
    q = str(q).strip()
    n = q[0] if q[0].isdigit() else q[1]
    return f"{int(y)}Q{n}"

# ── DATA ──────────────────────────────────────────────────────────────────────

def get_hist(market):
    cols = ','.join(T_LIST)
    return qr(f"SELECT year,quarter,{cols} FROM market_features "
              f"WHERE market='{market}' ORDER BY year,quarter")

def get_preds(market):
    df = qr(f"SELECT year,quarter,target,predicted,actual,baseline "
            f"FROM predictions WHERE market='{market}' "
            f"ORDER BY year,quarter,target")
    if df.empty: return pd.DataFrame()
    w = df.pivot_table(index=['year','quarter'], columns='target',
        values=['predicted','actual','baseline'], aggfunc='first').reset_index()
    w.columns = ['_'.join(str(c) for c in col).strip('_') for col in w.columns]
    w['period'] = w.apply(lambda r: fq(r['year'],r['quarter']), axis=1)
    return w.sort_values(['year','quarter']).reset_index(drop=True)

def get_fcs(market):
    df = qr(f"SELECT id,year,quarter,"
            f"gross_rent_predicted,net_rent_predicted,"
            f"capital_value_inr_predicted,yield_gross_predicted,"
            f"vacancy_rate_predicted,absorption_predicted,"
            f"gross_rent_actual,net_rent_actual,"
            f"capital_value_inr_actual,yield_gross_actual,"
            f"vacancy_rate_actual,absorption_actual,data_status "
            f"FROM forecasts WHERE market='{market}' ORDER BY year,quarter")
    if not df.empty:
        df['period'] = df.apply(lambda r: fq(r['year'],r['quarter']), axis=1)
    return df

def get_sent(market, year, quarter):
    row = conn().execute(
        f"SELECT composite_sentiment_score,leasing_velocity_qoq,"
        f"vacancy_change,absorption_ratio,precommit_pct,sentiment_label "
        f"FROM market_features WHERE market='{market}' "
        f"AND year={int(year)} AND quarter='{quarter}'").fetchone()
    if row: return row
    # fallback to latest available
    row2 = conn().execute(
        f"SELECT composite_sentiment_score,leasing_velocity_qoq,"
        f"vacancy_change,absorption_ratio,precommit_pct,sentiment_label "
        f"FROM market_features WHERE market='{market}' "
        f"AND composite_sentiment_score IS NOT NULL "
        f"ORDER BY year DESC,quarter DESC LIMIT 1").fetchone()
    return row2

def lr_ext(h, t, n):
    v = h[h[t].notna()].copy(); v['t']=range(len(v))
    if len(v)<4: return [],[]
    m = LinearRegression().fit(v[['t']].values, v[t].values)
    k = len(v)
    return list(range(k,k+n)),[float(m.predict([[k+i]])[0]) for i in range(n)]

# ── CHART ─────────────────────────────────────────────────────────────────────

def build_charts(dh, dp, df, market):
    """3 clean lines: Actual, XGBoost, Linear Regression. Shared x-axis, no bridges."""
    from sklearn.linear_model import LinearRegression as _LR
    import numpy as _np
    import warnings; warnings.filterwarnings('ignore')

    fig = plt.figure(figsize=(15, 8), facecolor=J['white'])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.56, wspace=0.32)

    for idx, target in enumerate(T_LIST):
        label, unit = TARGETS[target]
        ax = fig.add_subplot(gs[idx//3, idx%3])
        ax.set_facecolor(J['s50'])
        for sp in ax.spines.values():
            sp.set_color(J['s200']); sp.set_linewidth(0.6)
        ax.yaxis.grid(True, color=J['s200'], linewidth=0.4, alpha=0.6)
        ax.xaxis.grid(False)

        # History from 2022
        h = dh[(dh[target].notna()) & (dh['year'] >= 2022)].sort_values(['year','quarter']).reset_index(drop=True)
        if h.empty: continue

        p_col = f'predicted_{target}'
        b_col = f'baseline_{target}'

        # Unified sorted timeline — each (year,quarter) appears ONCE
        all_periods = sorted(set(
            [(int(r['year']), str(r['quarter'])) for _, r in h.iterrows()] +
            ([(int(r['year']), str(r['quarter'])) for _, r in dp.iterrows()] if not dp.empty else [])
        ), key=lambda k: (k[0], int(k[1][0])))

        pos  = {k: i for i, k in enumerate(all_periods)}
        n    = len(all_periods)

        # Build value arrays indexed by position
        act_vals  = {}
        xgb_vals  = {}
        lr_vals   = {}

        for _, r in h.iterrows():
            k = (int(r['year']), str(r['quarter']))
            act_vals[pos[k]] = float(r[target])

        if not dp.empty and p_col in dp.columns:
            for _, r in dp.sort_values(['year','quarter']).iterrows():
                k = (int(r['year']), str(r['quarter']))
                if k in pos:
                    if pd.notna(r.get(p_col)): xgb_vals[pos[k]] = float(r[p_col])
                    if b_col in dp.columns and pd.notna(r.get(b_col)):
                        lr_vals[pos[k]] = float(r[b_col])

        # If no baseline stored, compute LR from recent history
        if not lr_vals and xgb_vals and len(act_vals) >= 4:
            ax_arr = _np.array(sorted(act_vals.keys()))
            ay_arr = _np.array([act_vals[i] for i in ax_arr])
            _lrm   = _LR().fit(ax_arr.reshape(-1,1), ay_arr)
            for xi in sorted(xgb_vals.keys()):
                lr_vals[xi] = float(_lrm.predict([[xi]])[0])

        # Plot — sorted x values for each line
        def _plot_sorted(vdict, color, lw, ls='-', zorder=3):
            xs = sorted(vdict.keys())
            ys = [vdict[x] for x in xs]
            ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder,
                    solid_capstyle='round')

        if act_vals:  _plot_sorted(act_vals, '#000000', 2.2, zorder=5)

        # Anchor prediction lines to last actual point so they start from same origin
        if act_vals:
            _last_ax = max(act_vals.keys())        # x of last actual
            _last_ay = act_vals[_last_ax]          # y of last actual
            # Find last actual BEFORE predictions start
            if xgb_vals:
                _pred_start = min(xgb_vals.keys())
                # Use actual at pred_start-1 as anchor (quarter just before predictions)
                _anchor_x = max(x for x in act_vals if x < _pred_start) if any(x < _pred_start for x in act_vals) else _pred_start
                _anchor_y = act_vals.get(_anchor_x, _last_ay)
            else:
                _anchor_x, _anchor_y = _last_ax, _last_ay

        if xgb_vals:
            xs = sorted(xgb_vals.keys())
            ys = [xgb_vals[x] for x in xs]
            # Prepend anchor point so line starts from same place as actual
            ax.plot([_anchor_x] + xs, [_anchor_y] + ys,
                    color='#1B7A3E', lw=2.0, zorder=4, solid_capstyle='round')

        if lr_vals:
            xs = sorted(lr_vals.keys())
            ys = [lr_vals[x] for x in xs]
            ax.plot([_anchor_x] + xs, [_anchor_y] + ys,
                    color='#F5A623', lw=1.4, ls='--', zorder=3, solid_capstyle='round')

        # Separator at start of predictions
        if xgb_vals:
            sep = min(xgb_vals.keys()) - 0.5
            ax.axvline(sep, color='#E30613', lw=0.9, ls='--', alpha=0.55, zorder=6)
            ylim = ax.get_ylim()
            ax.text(sep-0.2, ylim[1], 'TRAINED',   fontsize=5, color='#888',
                    ha='right', va='top', fontfamily='monospace')
            ax.text(sep+0.2, ylim[1], 'PREDICTED', fontsize=5, color='#E30613',
                    ha='left',  va='top', fontfamily='monospace')

        # X-axis: year labels at Q1
        tp, tl = [], []
        for i, (yr, qt) in enumerate(all_periods):
            if qt == '1Q': tp.append(i); tl.append(str(yr))
        ax.set_xticks(tp)
        ax.set_xticklabels(tl, fontsize=7, rotation=0, color='#333', ha='center')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_title(f'{label}  ({unit})', fontsize=8.5, fontweight='bold',
                     color=J['night'], pad=5)
        ax.tick_params(colors=J['night'], labelsize=7)

    handles = [
        Line2D([0],[0], color='#000000', lw=2.2,          label='Actual'),
        Line2D([0],[0], color='#1B7A3E', lw=2.0,          label='XGBoost Predicted'),
        Line2D([0],[0], color='#F5A623', lw=1.4, ls='--', label='JLL Linear Regression'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=8, framealpha=0.95, edgecolor=J['s200'],
               labelcolor=J['night'], bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def pred_card_html(label, unit, pv, av):
    pstr = f"{float(pv):.4f}" if pv is not None else "—"
    has  = (av is not None and pd.notna(av))

    if has and pv is not None and float(av) != 0:
        err   = abs(float(pv)-float(av)) / abs(float(av)) * 100
        acc   = 100 - err
        good  = acc >= 60
        a_bg  = "#D4EDDA" if good else "#FFF3CD"
        a_fc  = "#1B7A3E" if good else "#7D4000"
        a_sym = "✓" if good else "△"
        a_val = f"{float(av):.4f}"
        bottom = (
            '<div style="margin-top:12px;padding-top:10px;'
            'border-top:2px solid #E5EBED;text-align:center">'
            '<div style="font-size:11px;color:#5A6169;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">'
            'Confirmed Actual</div>'
            '<div style="font-size:22px;font-weight:800;color:#131E29;'
            'font-family:monospace">' + a_val + '</div>'
            '<div style="margin-top:8px">'
            '<span style="background:' + a_bg + ';color:' + a_fc + ';'
            'border-radius:5px;padding:4px 12px;font-size:13px;font-weight:800">'
            + a_sym + ' ' + f"{acc:.1f}%" + ' Accurate</span>'
            '</div></div>'
        )
    else:
        bottom = (
            '<div style="margin-top:12px;padding-top:10px;'
            'border-top:2px dashed #DEDFE1;text-align:center">'
            '<div style="font-size:12px;color:#A6AAAE;font-style:italic">'
            'Awaiting REIS actual data</div>'
            '</div>'
        )

    return (
        '<div style="background:#F0F4F8;border:1.5px solid #DEDFE1;'
        'border-left:5px solid #003E51;border-radius:10px;'
        'padding:16px 14px;margin-bottom:10px;'
        'box-shadow:0 2px 6px rgba(0,62,81,0.08)">'

        # Centered label with background
        '<div style="text-align:center;background:#003E51;'
        'border-radius:6px;padding:6px 10px;margin-bottom:12px">'
        '<div style="font-size:13px;color:#FFFFFF;font-weight:800;'
        'text-transform:uppercase;letter-spacing:1px">' + label + '</div>'
        '</div>'

        # Predicted section
        '<div style="text-align:center">'
        '<div style="font-size:10px;color:#5A6169;font-weight:700;'
        'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px">'
        'XGBoost Predicted</div>'
        '<div style="font-size:26px;font-weight:800;color:#003E51;'
        'font-family:monospace;background:#E5EBED;'
        'border-radius:6px;padding:6px 12px;display:inline-block">'
        + pstr + '</div>'
        '<div style="font-size:10px;color:#A6AAAE;margin-top:4px">' + unit + '</div>'
        '</div>'

        + bottom + '</div>'
    )


def metric_card(title, val, sub, vc=None):
    c = vc or J['night']
    return (
        '<div style="background:#FFFFFF;border:1.5px solid #DEDFE1;'
        'border-radius:10px;padding:18px 20px;'
        'box-shadow:0 2px 6px rgba(0,0,0,0.07)">'
        '<div style="font-size:11px;color:#5A6169;text-transform:uppercase;'
        'letter-spacing:1px;font-weight:700;margin-bottom:8px">' + title + '</div>'
        '<div style="font-size:28px;font-weight:800;color:' + (c or "#131E29") + ';'
        'font-family:monospace;letter-spacing:-0.5px;line-height:1.1">' + val + '</div>'
        '<div style="font-size:12px;color:#5A6169;margin-top:5px;font-weight:500">' + sub + '</div>'
        '</div>'
    )

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    try:
        st.set_page_config(page_title="DarkDemand Oracle", page_icon="🪄",
                           layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html,body,[class*="css"]{{
        font-family:'Inter',sans-serif!important;
        background:{J['s50']}!important;
        color:{J['night']}!important;
    }}
    .stApp{{background:{J['s50']}!important}}
    /* Main content area white */
    section[data-testid="stSidebar"]{{background:{J['space']}!important}}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb]{{background:{J['white']}!important}}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb] span{{color:{J['night']}!important}}
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb] svg{{fill:{J['night']}!important}}
    section[data-testid="stSidebar"] .stSelectbox label{{
        color:{J['ocean']}!important;font-size:11px!important;
        font-weight:700!important;text-transform:uppercase!important}}
    section[data-testid="stSidebar"] p{{color:{J['ocean']}!important}}
    /* Section header */
    .sec{{font-size:13px;font-weight:800;color:{J['space']};
          text-transform:uppercase;letter-spacing:1.5px;
          padding-bottom:8px;border-bottom:3px solid {J['red']};
          margin:24px 0 16px}}
    /* Buttons */
    .stButton>button{{
        border-radius:6px!important;font-size:14px!important;
        font-weight:800!important;
        border:2px solid {J['space']}!important;
        background:{J['space']}!important;
        color:#FFFFFF!important;padding:8px 16px!important}}
    .stButton>button:hover{{
        background:{J['red']}!important;
        border-color:{J['red']}!important;
        color:#FFFFFF!important}}
    div[data-testid="stExpander"]{{
        border:1px solid {J['s200']}!important;
        border-radius:8px!important;background:{J['white']}!important}}
    div[data-testid="stExpander"] summary{{
        font-size:15px!important;font-weight:700!important;
        color:{J['night']}!important}}
    div[data-testid="stExpander"] summary p{{
        font-size:15px!important;font-weight:700!important;
        color:{J['night']}!important}}
    div[data-testid="stExpander"] p{{
        font-size:14px!important;color:{J['night']}!important}}
    .stCaption p{{font-size:12px!important;color:{J['s600']}!important;
        font-weight:500!important}}
    </style>""", unsafe_allow_html=True)

    try: meta = json.load(open(str(META_PATH)))
    except: meta = {}

    # DB VALIDATION
    try:
        _tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn()
        )['name'].tolist()
    except Exception as _e:
        st.error(f'Cannot open database: {DB_PATH}\n\n{_e}')
        st.info('Copy darkdemand_oracle.db (383-row version) into the Final folder, then refresh.')
        st.stop()
    if 'market_features' not in _tables:
        st.error('market_features table missing from darkdemand_oracle.db')
        st.warning(f'Database path checked: {DB_PATH}')
        st.info('Fix: copy your working darkdemand_oracle.db into the Final folder and refresh. Or run: python ML_train_xgboost.py')
        st.stop()
    if 'predictions' not in _tables:
        st.warning('No predictions table. Run python ML_train_xgboost.py to generate predictions.')



    # ── TOP NAV BAR — wand left, title center
    col_logo, col_title, col_r = st.columns([1, 5, 1])
    with col_logo:
        components.html("""
        <style>
        @keyframes wandspin{0%{transform:rotate(-15deg)}
            50%{transform:rotate(15deg)}100%{transform:rotate(-15deg)}}
        @keyframes sparkle{0%,100%{opacity:0;transform:scale(0)}
            50%{opacity:1;transform:scale(1)}}
        .wand{font-size:44px;display:inline-block;
              animation:wandspin 2s ease-in-out infinite;
              filter:drop-shadow(0 0 10px #BCDEE6);position:relative}
        .sp{position:absolute;font-size:13px;
            animation:sparkle 1.5s ease-in-out infinite}
        .sp1{top:-4px;left:28px;animation-delay:0s}
        .sp2{top:8px;left:48px;animation-delay:0.5s}
        .sp3{top:-8px;left:18px;animation-delay:1s}
        </style>
        <div style="padding:10px 0;position:relative;display:inline-block">
            <div class="wand">🪄
                <div class="sp sp1">✨</div>
                <div class="sp sp2">⭐</div>
                <div class="sp sp3">💫</div>
            </div>
        </div>
        """, height=70)
    with col_title:
        st.markdown(f"""
        <div style="text-align:center;padding:10px 0 6px">
            <div style="font-size:28px;font-weight:900;letter-spacing:-0.5px;
                line-height:1.15;
                background:linear-gradient(90deg,{J['space']} 0%,{J['red']} 50%,{J['space']} 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;display:inline-block">
                DarkDemand Oracle &nbsp;·&nbsp; ML Engine</div>
            <div style="font-size:11px;font-weight:700;color:{J['s600']};
                text-transform:uppercase;letter-spacing:2.5px;margin-top:5px">
                AI Rent &amp; Yield Forecasting &nbsp;·&nbsp; India CRE
                &nbsp;·&nbsp; XGBoost vs Linear Regression</div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        components.html("""
        <style>
          .hbtn {
            background:rgba(0,62,81,0.10); color:#003E51;
            border:1px solid rgba(0,62,81,0.28);
            padding:5px 13px; border-radius:6px;
            font-size:12px; font-weight:700; cursor:pointer;
            letter-spacing:0.4px; font-family:'Segoe UI',Arial,sans-serif;
            transition:background .2s; outline:none; float:right;
          }
          .hbtn:hover { background:rgba(0,62,81,0.22); }
        </style>
        <button class="hbtn" onclick="window.parent.location.href='/'">
          &#127968;&nbsp; Home
        </button>
        """, height=42)

    st.markdown(f"<hr style='border:none;border-top:3px solid {J['red']};margin:0 0 20px'>",
                unsafe_allow_html=True)

    # ── SIDEBAR (Space blue background via CSS)
    with st.sidebar:
        st.markdown(f"""<div style="color:{J['ocean']};font-size:12px;
            font-weight:800;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:12px">🌏 Market Selection</div>""",
            unsafe_allow_html=True)
        city   = st.selectbox("City",   ["Mumbai","Chennai"])
        sector = st.selectbox("Sector", ["Office","Retail"])
        market = f"{city}_{sector}"

        st.markdown("---")
        st.markdown(f"""<div style="color:{J['ocean']};font-size:12px;
            font-weight:800;text-transform:uppercase;letter-spacing:1px;
            margin-bottom:12px">🔮 Predict Quarter</div>""",
            unsafe_allow_html=True)
        py = st.selectbox("Year",    YEARS_FC)
        pq = st.selectbox("Quarter", QUARTERS)
        pb = st.button("▶  Predict", use_container_width=True)

        st.markdown("---")
        impr = meta.get('gross_rent',{}).get('rmse_improvement_pct','—')
        mape = meta.get('gross_rent',{}).get('mape_val_pct','—')
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;
                    padding:14px;margin-bottom:10px;border:1px solid rgba(188,222,230,0.2)">
            <div style="font-size:10px;color:{J['ocean']};text-transform:uppercase;
                font-weight:700;letter-spacing:0.8px">XGBoost vs Linear Reg.</div>
            <div style="font-size:30px;font-weight:800;color:#FFFFFF;
                font-family:monospace">+{impr}%</div>
            <div style="font-size:10px;color:{J['ocean']}">RMSE · 2023–2024 test set</div>
        </div>
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;
                    padding:12px 14px;margin-bottom:14px;
                    border:1px solid rgba(188,222,230,0.2)">
            <div style="font-size:10px;color:{J['ocean']};text-transform:uppercase;
                font-weight:700;letter-spacing:0.8px">Validation MAPE</div>
            <div style="font-size:28px;font-weight:800;color:#FFFFFF;
                font-family:monospace">{mape}%</div>
            <div style="font-size:10px;color:{J['ocean']}">2025–2026Q2 out-of-sample</div>
        </div>
        <div style="font-size:11px;color:rgba(188,222,230,0.8);
                line-height:2.2;font-weight:600">
            <span style="color:#1B7A3E">●</span> Engine 1 Sentiment active<br>
            <span style="color:#1B7A3E">●</span> Engine 2 XGBoost · 6 models<br>
            <span style="color:{J['s400']}">○</span> GDELT Layer 2 · pending<br>
            <span style="color:{J['s400']}">○</span> Engine 1 Live · demo
        </div>""", unsafe_allow_html=True)

    # ── LOAD DATA
    dh = get_hist(market)
    dp = get_preds(market)
    df = get_fcs(market)

    # Handle predict
    # ── ACTIVE QUARTER — clean simple logic
    # Reset on market change
    if st.session_state.get('_mkt') != market:
        st.session_state['_mkt']    = market
        st.session_state['_picked'] = ''

    just_deleted = st.session_state.pop('just_deleted', False)

    # All available periods across predictions + forecasts (combined, sorted)
    _all_avail = sorted(set(
        (list(dp['period'].values) if not dp.empty and 'period' in dp.columns else []) +
        (list(df['period'].values) if not df.empty and 'period' in df.columns else [])
    ))
    # Default = LATEST available quarter for this market (backtest or forecast)
    _default = _all_avail[-1] if _all_avail else ''

    # Handle Predict button
    if pb:
        period  = fq(py, pq)
        _yr_chk = int(period[:4])
        _qt_chk = period[5] + 'Q'   # e.g. "3Q"

        # Query DB directly — don't rely on loaded dp/df
        _chk_conn = sqlite3.connect(str(DB_PATH))
        _in_pred = _chk_conn.execute(
            "SELECT COUNT(*) FROM predictions "
            "WHERE market=? AND year=? AND quarter=?",
            (market, _yr_chk, _qt_chk)).fetchone()[0]
        _in_fore = _chk_conn.execute(
            "SELECT COUNT(*) FROM forecasts "
            "WHERE market=? AND year=? AND quarter=?",
            (market, _yr_chk, _qt_chk)).fetchone()[0]
        _chk_conn.close()

        if _in_pred > 0 or _in_fore > 0:
            # Exists in DB — show it
            st.session_state['_picked'] = period
            st.cache_resource.clear()
            st.rerun()
        else:
            # Not in DB — run forecast.py to generate
            with st.spinner(f"Generating forecast for {period}..."):
                import subprocess, sys, os
                env = os.environ.copy()
                env['PYTHONIOENCODING'] = 'utf-8'
                result = subprocess.run(
                    [sys.executable, str(BASE / 'ML_forecast.py'),
                     '--market', market,
                     '--quarter', period],
                    capture_output=True, text=True, cwd=str(BASE),
                    env=env, encoding='utf-8', errors='replace'
                )
            if result.returncode == 0:
                # Re-check DB directly
                _chk2 = sqlite3.connect(str(DB_PATH))
                _now  = _chk2.execute(
                    "SELECT COUNT(*) FROM forecasts "
                    "WHERE market=? AND year=? AND quarter=?",
                    (market, _yr_chk, _qt_chk)).fetchone()[0]
                _chk2.close()
                if _now > 0:
                    st.session_state['_picked'] = period
                    # Refresh next_quarter_preview for this market
                    _c_nqp = sqlite3.connect(str(DB_PATH))
                    _c_nqp.execute("DELETE FROM next_quarter_preview WHERE market=?", (market,))
                    _c_nqp.commit(); _c_nqp.close()
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.info(
                        f"**{period}** — ML_forecast.py ran but could not generate "
                        f"this quarter. This may be a historical quarter already "
                        f"in Prediction History below, or the seed data is missing.")
            else:
                st.error(f"ML_forecast.py error: {result.stderr[-300:]}")

    # Determine active — validate it still exists in the combined period list
    _picked = st.session_state.get('_picked', '')
    if _picked and _picked in _all_avail:
        active = _picked
    else:
        active = _default
        if _picked:                          # was set but no longer valid (deleted)
            st.session_state['_picked'] = ''

    # ── METRIC CARDS — dynamic based on active quarter
    # lh = latest actual row BUT capped to latest prediction quarter
    # (market_features may have actuals beyond what we've modelled)
    _lh_conn  = sqlite3.connect(str(DB_PATH))
    _lh_limit = _lh_conn.execute(
        "SELECT year, quarter FROM predictions WHERE market=? "
        "ORDER BY year DESC, quarter DESC LIMIT 1", (market,)).fetchone()
    _lh_conn.close()
    if _lh_limit:
        lh = dh[(dh['gross_rent'].notna()) &
                ((dh['year'] < _lh_limit[0]) |
                 ((dh['year'] == _lh_limit[0]) &
                  (dh['quarter'] <= _lh_limit[1])))].iloc[-1]
    else:
        lh = dh[dh['gross_rent'].notna()].iloc[-1]
    impr_v = meta.get('gross_rent',{}).get('rmse_improvement_pct','—')

    # Card 1: current active quarter gross rent (predicted or actual)
    # Card 2: next forecast after active
    # Card 3: sentiment for active quarter
    # Card 4: accuracy if actuals available, else XGBoost improvement

    # Auto-generate next forecast if forecasts table is empty
    if df.empty or 'period' not in df.columns or len(df) == 0:
        _nxt_conn = sqlite3.connect(str(DB_PATH))
        # Use latest PREDICTION row as seed (not market_features)
        # so we generate the next quarter after what we've predicted
        _lh_pred = _nxt_conn.execute(
            f"SELECT DISTINCT year, quarter FROM predictions "
            f"WHERE market='{market}' "
            f"ORDER BY year DESC, quarter DESC LIMIT 1"
        ).fetchone()
        _nxt_conn.close()
        if _lh_pred:
            import subprocess, sys, os
            _nyr, _nqt = _lh_pred
            _qmap = {'1Q':'2Q','2Q':'3Q','3Q':'4Q','4Q':'1Q'}
            _nqt2 = _qmap[_nqt]
            _nyr2 = _nyr + 1 if _nqt == '4Q' else _nyr
            _nperiod = f"{_nyr2}Q{_nqt2[0]}"
            _env = os.environ.copy(); _env['PYTHONIOENCODING'] = 'utf-8'
            subprocess.run(
                [sys.executable, str(BASE/'ML_forecast.py'),
                 '--market', market, '--quarter', _nperiod],
                capture_output=True, cwd=str(BASE), env=_env
            )
            st.cache_resource.clear()
            df = get_fcs(market)

    # Active quarter values
    _act_conn = sqlite3.connect(str(DB_PATH))
    _card_pv, _card_av, _card_period = None, None, active or fq(lh['year'], lh['quarter'])
    if active:
        _ayr = int(active[:4]); _aqt = active[5] + 'Q'
        _pr = _act_conn.execute(
            "SELECT predicted, actual FROM predictions "
            "WHERE market=? AND year=? AND quarter=? AND target='gross_rent'",
            (market, _ayr, _aqt)).fetchone()
        if _pr:
            _card_pv, _card_av = _pr
        else:
            _fr = _act_conn.execute(
                "SELECT gross_rent_predicted, gross_rent_actual FROM forecasts "
                "WHERE market=? AND year=? AND quarter=?",
                (market, _ayr, _aqt)).fetchone()
            if _fr: _card_pv, _card_av = _fr
    _act_conn.close()

    # Next forecast after active
    lh_yr = int(active[:4]) if active else int(lh['year'])
    lh_qn = int(active[5]) if active else int(str(lh['quarter'])[0])
    _lf_all = df[df['gross_rent_predicted'].notna()].copy() if not df.empty else pd.DataFrame()
    if not _lf_all.empty and 'period' in _lf_all.columns:
        _lf_all['_yr'] = _lf_all['year'].astype(int)
        _lf_all['_qn'] = _lf_all['quarter'].astype(str).str[0].astype(int)
        _after = _lf_all[
            (_lf_all['_yr'] > lh_yr) |
            ((_lf_all['_yr'] == lh_yr) & (_lf_all['_qn'] > lh_qn))
        ]
        lf = _after.iloc[0].to_dict() if not _after.empty else None
    else:
        lf = None

    # Sentiment for active quarter
    _syr = int(active[:4]) if active else int(lh['year'])
    _sqt = (active[5]+'Q') if active else str(lh['quarter'])
    sr   = get_sent(market, _syr, _sqt) or get_sent(market, int(lh['year']), str(lh['quarter']))

    # Card 4: accuracy if both pred+actual exist
    _card4_title = "XGBoost vs Linear Reg."
    _card4_val   = f"+{impr_v}%"
    _card4_sub   = "RMSE improvement · test set"
    _card4_color = J['ok']
    if _card_pv is not None and _card_av is not None and pd.notna(_card_av) and float(_card_av) != 0:
        _err  = abs(float(_card_pv)-float(_card_av))/abs(float(_card_av))*100
        _acc  = 100-_err
        _card4_title = f"Accuracy · {_card_period}"
        _card4_val   = f"{_acc:.1f}%"
        _card4_sub   = f"Gross Rent · {_err:.1f}% error"
        _card4_color = J['ok'] if _acc >= 60 else J['warn']

    # ── METRIC CARDS — single source of truth: 'active' quarter
    def _db_query(sql, params=()):
        c = sqlite3.connect(str(DB_PATH)); r = c.execute(sql, params).fetchone(); c.close(); return r

    # ── Card 1: Latest CONFIRMED ACTUAL only (not prediction)
    # Check predictions table first, then forecasts for confirmed actual
    _c1_per = active or fq(lh['year'], lh['quarter'])
    if active:
        _ayr = int(active[:4]); _aqt = active[5] + 'Q'
        _pa = _db_query("SELECT actual FROM predictions WHERE market=? AND year=? AND quarter=? AND target='gross_rent' AND actual IS NOT NULL", (market,_ayr,_aqt))
        if not _pa:
            _pa = _db_query("SELECT gross_rent_actual FROM forecasts WHERE market=? AND year=? AND quarter=? AND gross_rent_actual IS NOT NULL", (market,_ayr,_aqt))
        _c1_act  = float(_pa[0]) if _pa and _pa[0] else None
        _c1_val  = f"₹{_c1_act:.1f}" if _c1_act else "Awaiting REIS data"
        _c1_sub  = "Gross Rent · Confirmed Actual" if _c1_act else "No actual data yet"
        # Also get predicted for card 4 accuracy
        _pp = _db_query("SELECT predicted FROM predictions WHERE market=? AND year=? AND quarter=? AND target='gross_rent'", (market,_ayr,_aqt))
        if not _pp:
            _pp = _db_query("SELECT gross_rent_predicted FROM forecasts WHERE market=? AND year=? AND quarter=?", (market,_ayr,_aqt))
        _c1_pred = float(_pp[0]) if _pp and _pp[0] else None
    else:
        _c1_act, _c1_pred = None, None
        _c1_val = "—"; _c1_sub = "No quarter selected"

    # ── Card 2: Next quarter — always auto-predicted via ML_forecast.py --preview-only
    # Writes ONLY to next_quarter_preview, never to forecasts table.

    def _refresh_nqp(mkt):
        """Auto-predict next quarter for Card 2 via --preview-only mode.
        Writes ONLY to next_quarter_preview — NEVER touches the forecasts table.
        Forecasts table is only written when user presses Predict explicitly."""
        import subprocess as _sp, sys as _sys, os as _os
        _c = sqlite3.connect(str(DB_PATH))

        # Latest confirmed actual across predictions and forecasts
        _lp = _c.execute(
            "SELECT year, quarter FROM predictions "
            "WHERE market=? AND actual IS NOT NULL "
            "ORDER BY year DESC, quarter DESC LIMIT 1", (mkt,)).fetchone()
        _lf = _c.execute(
            "SELECT year, quarter FROM forecasts "
            "WHERE market=? AND gross_rent_actual IS NOT NULL "
            "ORDER BY year DESC, quarter DESC LIMIT 1", (mkt,)).fetchone()
        _c.close()

        def _pi(r): return r[0] * 10 + int(str(r[1])[0]) if r else 0
        _lat = _lp if _pi(_lp) >= _pi(_lf) else _lf
        if not _lat:
            return

        _qmap  = {'1Q':'2Q','2Q':'3Q','3Q':'4Q','4Q':'1Q'}
        _nqt   = _qmap[_lat[1]]
        _nyr   = _lat[0] + (1 if _lat[1] == '4Q' else 0)
        _nperiod = f"{_nyr}Q{_nqt[0]}"

        # Run ML_forecast.py with --preview-only:
        #   → computes XGBoost prediction for next quarter
        #   → writes ONLY to next_quarter_preview
        #   → NEVER inserts into forecasts (so deleted quarters stay deleted)
        _env = _os.environ.copy(); _env['PYTHONIOENCODING'] = 'utf-8'
        _sp.run(
            [_sys.executable, str(BASE / 'ML_forecast.py'),
             '--market', mkt, '--quarter', _nperiod, '--preview-only'],
            capture_output=True, cwd=str(BASE), env=_env)

    # Always refresh NQP on every load — guarantees it reflects latest confirmed actual
    _refresh_nqp(market)
    _nqp_row = _db_query(
        "SELECT year, quarter, gross_rent_predicted FROM next_quarter_preview WHERE market=?",
        (market,))
    if _nqp_row and _nqp_row[2]:
        _c2_per = fq(_nqp_row[0], _nqp_row[1])
        _c2_val = f"₹{float(_nqp_row[2]):.1f}"
    else:
        _c2_per, _c2_val = '—', '—'

    # ── Card 3: Engine 1 sentiment for active quarter
    _syr = int(active[:4]) if active else int(lh['year'])
    _sqt = (active[5]+'Q') if active else str(lh['quarter'])
    sr   = get_sent(market, _syr, _sqt)

    # ── Card 4: Accuracy (actual vs predicted) for active quarter
    impr_v = meta.get('gross_rent',{}).get('rmse_improvement_pct','—')
    if _c1_pred and _c1_act and _c1_act != 0:
        _err = abs(_c1_pred - _c1_act) / abs(_c1_act) * 100
        _acc = 100 - _err
        _c4_title = f"Accuracy · {_c1_per}"
        _c4_val   = f"{_acc:.1f}%"
        _c4_sub   = f"Gross Rent · {_err:.1f}% error"
        _c4_color = J['ok'] if _acc >= 60 else J['warn']
    else:
        _c4_title = "XGBoost vs Linear Reg."
        _c4_val   = f"+{impr_v}%"
        _c4_sub   = "RMSE improvement · test set"
        _c4_color = J['ok']


    mc = st.columns(4)
    mc[0].markdown(metric_card(
        f"Current · {_c1_per}", _c1_val, _c1_sub),
        unsafe_allow_html=True)
    mc[1].markdown(metric_card(
        f"Next · {_c2_per}", _c2_val,
        "Gross Rent · INR/sq.ft/month", J['space']),
        unsafe_allow_html=True)
    mc[2].markdown(metric_card(
        f"Engine 1 Sentiment · {_c1_per}",
        f"{sr[0]:.1f}/100" if sr else "—",
        sr[5] if sr else "No Engine 1 data"),
        unsafe_allow_html=True)
    mc[3].markdown(metric_card(
        _c4_title, _c4_val, _c4_sub, _c4_color),
        unsafe_allow_html=True)

    # ── ACTIVE PREDICTION PANEL
    if active:
        _pm  = dp[dp['period']==active]
        prow = _pm.iloc[0].to_dict() if not dp.empty and not _pm.empty else None
        _fm  = df[df['period']==active] if not df.empty and 'period' in df.columns else pd.DataFrame()
        frow = _fm.iloc[0].to_dict() if not _fm.empty else None

        if prow is not None or frow is not None:
            h1, h2 = st.columns([8,1])
            with h1:
                st.markdown(
                    f'<div class="sec">Prediction — '
                    f'{market.replace("_"," ")} · {active}</div>',
                    unsafe_allow_html=True)
            with h2:
                st.markdown("<div style='height:16px'></div>",
                            unsafe_allow_html=True)
                if st.button("🗑 Delete", key="clr"):
                    _yr  = int(active[:4])
                    _qtr = active[5] + 'Q'

                    # Build sorted list of ALL periods before deleting
                    _all = []
                    if not dp.empty and 'period' in dp.columns:
                        _all += list(dp['period'].values)
                    if not df.empty and 'period' in df.columns:
                        _all += list(df['period'].values)
                    _all  = sorted(set(_all))
                    _all_after = [p for p in _all if p != active]  # exclude deleted
                    _prev = _all_after[-1] if _all_after else ''   # latest surviving

                    # DELETE — direct non-cached connection (avoids stale cache reads)
                    _del_c = sqlite3.connect(str(DB_PATH))
                    if frow is not None:
                        _del_c.execute(
                            "DELETE FROM forecasts WHERE market=? AND year=? AND quarter=?",
                            (market, _yr, _qtr))
                    elif prow is not None:
                        _del_c.execute(
                            "DELETE FROM predictions WHERE market=? AND year=? AND quarter=?",
                            (market, _yr, _qtr))
                    _del_c.commit()
                    _del_c.close()

                    # Refresh next_quarter_preview so card 2 updates
                    _d_nqp = sqlite3.connect(str(DB_PATH))
                    _d_nqp.execute("DELETE FROM next_quarter_preview WHERE market=?", (market,))
                    _d_nqp.commit(); _d_nqp.close()
                    # Go to previous quarter
                    st.session_state['_picked'] = _prev
                    st.session_state['just_deleted'] = True
                    st.cache_resource.clear()
                    st.rerun()

            yr  = int(prow['year']) if prow else int(frow['year'])
            qtr = str(prow['quarter']) if prow else str(frow['quarter'])

            # Check sentiment — warn only for forecast quarters with no Engine 1 data
            _s_check = get_sent(market, yr, qtr)
            _is_fc_quarter = (frow is not None and prow is None)
            if not _s_check and _is_fc_quarter:
                # Find latest available sentiment for this market
                _last_sent = conn().execute(
                    f"SELECT year, quarter FROM market_features "
                    f"WHERE market='{market}' AND composite_sentiment_score IS NOT NULL "
                    f"ORDER BY year DESC, quarter DESC LIMIT 1"
                ).fetchone()
                _last_str = (f"{_last_sent[0]}Q{str(_last_sent[1])[0]}"
                             if _last_sent else "unknown")
                st.info(
                    f"ℹ️ **Engine 1 has not run for {active}.** "
                    f"Forecast is using last available sentiment from **{_last_str}**. "
                    f"Run **Engine 1 Signal Scanner** to get live demand signals for this quarter "
                    f"before presenting to clients."
                )

            # 6 variable cards in 3 cols
            vc = st.columns(3)
            pvals = {}
            for i,(t,(l,u)) in enumerate(TARGETS.items()):
                if prow:
                    pv=prow.get(f'predicted_{t}')
                    av=prow.get(f'actual_{t}')
                    is_fc=False; fid=None
                else:
                    pv=frow.get(f'{t}_predicted')
                    av=frow.get(f'{t}_actual')
                    is_fc=True
                    fid=int(frow['id']) if 'id' in frow else None
                pvals[t]={'pred':pv,'actual':av,'is_fc':is_fc,'fid':fid}
                with vc[i%3]:
                    st.markdown(pred_card_html(l,u,pv,av),
                                unsafe_allow_html=True)

            # Actuals form
            all_ok = all(pd.notna(pvals[t]['actual']) and
                         pvals[t]['actual'] is not None for t in TARGETS)
            if not all_ok:
                with st.expander("➕ Submit Actual REIS Values"):
                    st.caption("⚠️ Enter the REAL confirmed REIS values below. "
                               "These fields are pre-filled with XGBoost predictions — "
                               "replace them with actual REIS data before saving.")
                    with st.form(key=f"f_{yr}_{qtr}"):
                        fc3=st.columns(3); inp={}
                        for i,(t,(l,u)) in enumerate(TARGETS.items()):
                            info=pvals[t]
                            if pd.notna(info['actual']) and \
                               info['actual'] is not None: continue
                            with fc3[i%3]:
                                inp[t]=st.number_input(f"{l} ({u})",
                                    value=float(info['pred']) if info['pred'] else 0.0,
                                    format="%.4f", key=f"i_{t}_{yr}_{qtr}")
                        if inp and st.form_submit_button("💾 Save Actuals",
                                                         use_container_width=True):
                            for t,v in inp.items():
                                info=pvals[t]
                                if info['is_fc'] and info['fid']:
                                    qw(f"UPDATE forecasts SET {t}_actual=?,"
                                       f"data_status='actual_available' WHERE id=?",
                                       (v,info['fid']))
                                else:
                                    qw("UPDATE predictions SET actual=? "
                                       "WHERE market=? AND year=? "
                                       "AND quarter=? AND target=?",
                                       (v,market,yr,qtr,t))
                            # Clear NQP so card 2 refreshes to next quarter
                            _s_nqp = sqlite3.connect(str(DB_PATH))
                            _s_nqp.execute("DELETE FROM next_quarter_preview WHERE market=?", (market,))
                            _s_nqp.commit(); _s_nqp.close()
                            st.cache_resource.clear()
                            st.success("✅ Saved. Rerun ML_train_xgboost.py to retrain.")
                            st.rerun()
            else:
                # Accuracy summary row
                st.markdown(f"""<div style="background:{J['ok_bg']};
                    border-left:5px solid {J['ok']};border-radius:8px;
                    padding:12px 18px;margin-top:8px">
                    <div style="font-size:13px;font-weight:800;color:{J['ok']}">
                    ✓ All actuals confirmed for {active}</div></div>""",
                    unsafe_allow_html=True)

    # ── HISTORICAL PREDICTIONS — NEWEST FIRST, SKIP ACTIVE
    st.markdown('<div class="sec">Prediction History</div>', unsafe_allow_html=True)
    st.caption("Newest first · Expand row to see all 6 variables · "
               "Blue = XGBoost Predicted · Black = Actual · Green ≥ 60% = accurate")

    # ── Build unified history: predictions (dp) + forecasts (df) excl. active
    def _norm_fcs_row(frow):
        """Map forecasts columns → predictions column format for history display."""
        return {
            'period':   frow.get('period'),
            'year':     frow.get('year'),
            'quarter':  frow.get('quarter'),
            '_fcs_id':  frow.get('id'),           # to update forecasts table
            **{f'predicted_{t}': frow.get(f'{t}_predicted') for t in TARGETS},
            **{f'actual_{t}':    frow.get(f'{t}_actual')    for t in TARGETS},
        }

    _hist_rows = []   # list of (source, row_dict)  source='pred'|'fcs'
    if not dp.empty:
        for _, r in dp.iterrows():
            if r['period'] != active:
                _hist_rows.append(('pred', r.to_dict()))
    if not df.empty:
        for _, r in df.iterrows():
            if r['period'] != active:
                _hist_rows.append(('fcs', _norm_fcs_row(r.to_dict())))
    # Sort newest first
    _hist_rows.sort(key=lambda x: x[1]['period'], reverse=True)

    if not _hist_rows:
        st.info("No prediction history.")
    else:
        for _src, row in _hist_rows:
            period = row['period']

            gp  = row.get('predicted_gross_rent')
            ga  = row.get('actual_gross_rent')
            has = pd.notna(ga) if ga is not None else False

            if has and ga!=0 and gp is not None:
                err=abs(float(gp)-float(ga))/abs(float(ga))*100
                acc=100-err
                ok  = acc>=60
                bstr= f"{'✓' if ok else '△'} {acc:.1f}%"
                bc  = J['ok'] if ok else J['warn']
                bbg = J['ok_bg'] if ok else J['warn_bg']
                badge_html=(f'<span style="background:{bbg};color:{bc};'
                            f'border-radius:4px;padding:2px 8px;'
                            f'font-size:12px;font-weight:800">{bstr}</span>')
            else:
                badge_html=(f'<span style="color:{J["s400"]};font-size:12px">'
                            f'Awaiting actual</span>')
            _src_tag = (f'<span style="color:{J["s400"]};font-size:11px;'
                        f'margin-left:8px">[Forecast]</span>'
                        if _src == 'fcs' else '')

            exp_lbl = (f"{period}   ·   Gross Rent: "
                       f"Pred {float(gp):.2f}" if gp else period)
            if has: exp_lbl += f"   /   Actual {float(ga):.2f}"

            with st.expander(exp_lbl):
                # View button
                qb,_ = st.columns([1,5])
                with qb:
                    if st.button("View ↑", key=f"v_{period}"):
                        st.session_state['_picked'] = period
                        st.rerun()

                # 6 variable grid
                vc2=st.columns(3)
                for i,(t,(l,u)) in enumerate(TARGETS.items()):
                    pv=row.get(f'predicted_{t}')
                    av=row.get(f'actual_{t}')
                    with vc2[i%3]:
                        st.markdown(pred_card_html(l,u,pv,av),
                                    unsafe_allow_html=True)

                # Save actuals if missing
                has_all=all(pd.notna(row.get(f'actual_{t}')) for t in TARGETS)
                if not has_all:
                    st.markdown("<div style='height:6px'></div>",
                                unsafe_allow_html=True)
                    u3=st.columns(3); ui={}
                    for i,(t,(l,u)) in enumerate(TARGETS.items()):
                        av=row.get(f'actual_{t}')
                        pv=row.get(f'predicted_{t}')
                        if pd.notna(av): continue
                        with u3[i%3]:
                            ui[t]=st.number_input(l,
                                value=float(pv) if pv else 0.0,
                                format="%.4f", key=f"h_{t}_{period}")
                    if ui and st.button("💾 Save", key=f"sv_{period}"):
                        if _src == 'pred':
                            for t,v in ui.items():
                                qw("UPDATE predictions SET actual=? "
                                   "WHERE market=? AND year=? "
                                   "AND quarter=? AND target=?",
                                   (v,market,int(row['year']),str(row['quarter']),t))
                        else:
                            for t,v in ui.items():
                                qw(f"UPDATE forecasts SET {t}_actual=? "
                                   "WHERE market=? AND year=? AND quarter=?",
                                   (v,market,int(row['year']),str(row['quarter'])))
                        _h_nqp = sqlite3.connect(str(DB_PATH))
                        _h_nqp.execute("DELETE FROM next_quarter_preview WHERE market=?", (market,))
                        _h_nqp.commit(); _h_nqp.close()
                        st.success("Saved.")
                        st.rerun()
                else:
                    st.markdown(
                        f'<span style="color:{J["ok"]};font-size:13px;'
                        f'font-weight:800">✓ All actuals confirmed</span>',
                        unsafe_allow_html=True)

    # ── 6 CHARTS — BELOW HISTORY
    st.markdown('<div class="sec">All Variables — Full Timeline</div>',
                unsafe_allow_html=True)
    st.caption(f"Showing from {FROM_YR} onwards · "
               "Black = Actual · Space Blue = XGBoost · "
               "Red dashed = Linear Regression · Teal dotted = Forecast")
    # Force chart redraw on every interaction by passing DB mod time
    import os as _os
    _db_mtime = _os.path.getmtime(str(DB_PATH))
    fig = build_charts(dh, dp, df, market)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── FOOTER
    st.markdown(f"""<div style="margin-top:40px;padding:16px 20px;
                background:{J['space']};border-radius:10px;
                display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:11px;color:{J['ocean']};font-weight:600">
            DarkDemand Oracle · Engine 2 · XGBoost replacing Linear Regression
        </span>
        <span style="font-size:11px;color:{J['ocean']}">
            4 markets · 6 targets · 68–85% RMSE improvement
        </span></div>""", unsafe_allow_html=True)

if __name__ == '__main__': main()
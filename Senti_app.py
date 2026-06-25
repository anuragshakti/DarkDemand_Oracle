"""
app.py — Dark Demand Intelligence | Streamlit UI
Run: streamlit run app.py
"""

import json
import time
import pandas as pd
import streamlit as st

import Senti_config as config
import Senti_database as database
import Senti_pipeline as pipeline

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(
        page_title="Dark Demand Intelligence | JLL",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# INIT DB ON FIRST RUN
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init():
    database.init_db()
    return True

init()

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── BUTTON — fixed text wrap, JLL Space theme ── */
  .stButton > button {
    background: linear-gradient(135deg, #003E51 0%, #005a76 100%);
    color: white; font-weight: 700;
    border: none; padding: 0.7rem 1rem;
    font-size: 0.88rem; border-radius: 8px;
    width: 100%; line-height: 1.3; white-space: normal;
    box-shadow: 0 2px 8px rgba(0,62,81,0.35);
    transition: all 0.2s; cursor: pointer; letter-spacing: 0.3px;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #002E3D 0%, #004460 100%);
    box-shadow: 0 4px 14px rgba(0,62,81,0.45);
    transform: translateY(-1px);
  }

  /* ── SCORE NUMBERS — colour-coded by signal level ── */
  .score-high  { font-size:2.8rem; font-weight:900; color:#16A34A; line-height:1; }
  .score-med   { font-size:2.8rem; font-weight:900; color:#D97706; line-height:1; }
  .score-watch { font-size:2.8rem; font-weight:900; color:#2563EB; line-height:1; }

  /* ── BADGES — HIGH = green (positive demand signal) ── */
  .badge-high  { background:#16A34A; color:#fff; padding:3px 10px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .badge-med   { background:#D97706; color:#fff; padding:3px 10px; border-radius:10px; font-size:0.78rem; font-weight:700; }
  .badge-watch { background:#2563EB; color:#fff; padding:3px 10px; border-radius:10px; font-size:0.78rem; font-weight:700; }

  /* ── COMPANY CARDS ── */
  .card-high  { border-left:4px solid #16A34A; background:#f0fdf4; padding:14px 18px; margin-bottom:12px; border-radius:0 10px 10px 0; }
  .card-med   { border-left:4px solid #D97706; background:#fffbeb; padding:14px 18px; margin-bottom:12px; border-radius:0 10px 10px 0; }
  .card-watch { border-left:4px solid #2563EB; background:#eff6ff; padding:14px 18px; margin-bottom:12px; border-radius:0 10px 10px 0; }

  .evidence   { color:#555; font-style:italic; font-size:0.88rem; margin-top:4px; }
  .log-box {
    background:#0D1117; color:#58D68D; padding:12px 16px;
    border-radius:6px; font-family:monospace; font-size:0.8rem;
    max-height:300px; overflow-y:auto; line-height:1.5;
  }
  .src-link { font-size:0.8rem; color:#1a73e8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
# ── HOME NAVIGATION — components.html iframe → window.parent nav (same tab) ──
import streamlit.components.v1 as _cv_home
_cv_home.html("""
<style>
  .hbtn {
    background:rgba(0,62,81,0.10); color:#003E51;
    border:1px solid rgba(0,62,81,0.28);
    padding:5px 15px; border-radius:6px;
    font-size:12px; font-weight:700; cursor:pointer;
    letter-spacing:0.4px; font-family:'Segoe UI',Arial,sans-serif;
    transition:background .2s; outline:none;
  }
  .hbtn:hover { background:rgba(0,62,81,0.22); }
</style>
<button class="hbtn" onclick="window.parent.location.href='/'">
  &#127968;&nbsp; Home
</button>
""", height=42)
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏢 JLL Dark Demand")
    st.markdown("---")

    # Market selector
    st.markdown("### 🌏 Select Market")
    region  = st.selectbox("Region",  options=list(config.MARKETS.keys()))
    country = st.selectbox("Country", options=list(config.MARKETS[region].keys()))
    market  = st.selectbox("City",    options=config.MARKETS[region][country])

    st.markdown("---")
    st.markdown("### 🗓️ Time Period")
    quarter = st.selectbox("Quarter", options=["1Q", "2Q", "3Q", "4Q"], index=1)
    year    = st.selectbox("Year",    options=list(range(2020, 2028)), index=6)
    st.caption(f"Scanning news from: **{config.QUARTER_DATES[quarter][0]}/{year}** → **{config.QUARTER_DATES[quarter][1]}/{year}**")

    st.markdown("---")
    run_btn = st.button("🔍  Find Dark Demand")

    st.markdown("---")
    # DB stats
    stats = database.get_db_stats()
    st.markdown("### 📊 Database")
    st.caption(f"✅ Scans completed: **{stats['total_scans']}**")
    st.caption(f"🏢 Companies tracked: **{stats['total_companies']}**")
    st.caption(f"📰 Sentiment records: **{stats['sentiment_records']}**")
    st.caption(f"🌍 Markets scanned: **{stats['markets_scanned']}**")
    st.caption(f"💾 Cache entries: **{stats['cache_entries']}**")

    st.markdown("---")
    st.caption("⚡ JLL GPT · Selenium Stealth · SQLite")
    st.caption("🔒 Raw text never stored")

# ─────────────────────────────────────────────────────────────────────────────
# TITLE HEADER  (matches ML engine style)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit.components.v1 as _cv1
_cv1.html("""
<style>
  /* ── Antenna ring pulse ── */
  @keyframes ringExpand {
    0%   { transform:scale(0.55); opacity:0.85; }
    100% { transform:scale(2.20); opacity:0;    }
  }
  /* ── Icon bob ── */
  @keyframes antennaBob {
    0%,100% { transform:translateY(0px) scale(1.00); }
    50%     { transform:translateY(-5px) scale(1.08); }
  }
  /* ── Tip flicker ── */
  @keyframes tipFlicker {
    0%,100% { opacity:1; }
    45%     { opacity:0.3; }
    55%     { opacity:1; }
  }
  body,html { margin:0;padding:0; }
  .wrap {
    display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    padding:10px 0 4px;
    text-align:center;
  }
  .icon-shell {
    position:relative;
    width:78px;height:78px;
    display:flex;align-items:center;justify-content:center;
    margin-bottom:6px;
  }
  /* Three expanding rings */
  .ring {
    position:absolute;border-radius:50%;
    border:2px solid rgba(0,62,81,0.75);
    width:100%;height:100%;
  }
  .ring1 { animation:ringExpand 2.0s ease-out 0.0s infinite; }
  .ring2 { animation:ringExpand 2.0s ease-out 0.6s infinite; }
  .ring3 { animation:ringExpand 2.0s ease-out 1.2s infinite; }
  .antenna {
    font-size:52px;line-height:1;
    position:relative;z-index:2;
    animation:antennaBob 2.4s ease-in-out infinite;
    filter:
      drop-shadow(0 0 8px rgba(0,62,81,0.9))
      drop-shadow(0 0 18px rgba(188,222,230,0.7))
      drop-shadow(0 0 32px rgba(0,62,81,0.5));
  }
  /* Signal tip dot */
  .tip {
    position:absolute;top:6px;right:14px;
    width:9px;height:9px;border-radius:50%;
    background:#E30613;
    box-shadow:0 0 6px 3px rgba(227,6,19,0.75);
    animation:tipFlicker 1.6s ease-in-out infinite;
    z-index:3;
  }
  .title-text {
    font-family:'Segoe UI',Arial,sans-serif;
    font-size:24px;font-weight:900;color:#003E51;
    letter-spacing:0.4px;line-height:1.15;
  }
  .sub-text {
    font-family:'Segoe UI',Arial,sans-serif;
    font-size:12px;color:#64748b;
    letter-spacing:0.3px;margin-top:3px;
  }
  .accent-bar {
    width:100%;max-width:520px;height:3px;margin-top:8px;
    background:linear-gradient(90deg,transparent,#003E51 25%,#BCDEE6 50%,#003E51 75%,transparent);
    border-radius:2px;
  }
</style>
<div class="wrap">
  <div class="icon-shell">
    <div class="ring ring1"></div>
    <div class="ring ring2"></div>
    <div class="ring ring3"></div>
    <div class="antenna">📡</div>
    <div class="tip"></div>
  </div>
  <div class="title-text">Dark Demand Intelligence</div>
  <div class="sub-text">Signal Scanner &middot; Sentiment Analyzer &middot; JLL APAC 2026</div>
  <div class="accent-bar"></div>
</div>
""", height=148)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_scan, tab_history, tab_ml = st.tabs([
    "🔍 Live Scan", "📋 Scan History", "🤖 ML Export"
])

# ── TAB 1: LIVE SCAN ─────────────────────────────────────────────────────────
with tab_scan:

    if not run_btn:
        st.markdown(f"""
        ### How Dark Demand Works

        **Step 1 — Sector Discovery** ({len(config.CATEGORIES)} sectors scanned)
        Search Google across Financial Services, Technology, Consulting, Healthcare,
        Manufacturing and Retail for companies hiring or expanding in your selected market.

        **Step 2 — AI Company Extraction**
        JLL GPT reads search results and extracts company names showing expansion signals.

        **Step 3 — Targeted Deep Scan**
        For the top **{config.PIPELINE['max_companies_phase2']} companies**,
        run {len(config.COMPANY_QUERIES)} targeted Google searches each and read full articles.

        **Step 4 — Sentiment Scoring**
        JLL GPT scores each article: signal strength, sentiment, space estimate, timeline.

        **Step 5 — Dark Demand Score (0–100)**
        Aggregated score. Companies ≥75 = high probability of leasing in 12–18 months.

        ---
        > Select **{region} → {country} → {market}** and click **Find Dark Demand**
        """)

    else:
        st.markdown(f"### 📡 Scanning: **{market}, {country}** — **{quarter} {year}** ({region})")

        log_box       = st.empty()
        progress_bar  = st.progress(0)
        status_line   = st.empty()

        log_lines = []
        stage_pct = {
            "init": 5, "phase1": 30, "extract": 50,
            "phase2": 70, "score": 88, "complete": 100, "error": 100,
        }

        emoji = {
            "init":"🔧","phase1":"🔍","extract":"🤖",
            "phase2":"📰","score":"📊","complete":"🎯","error":"❌",
        }

        def progress_cb(stage: str, msg: str):
            line = f"{emoji.get(stage,'•')} {msg}"
            log_lines.append(line)
            log_box.markdown(
                f'<div class="log-box">' + "<br>".join(log_lines[-22:]) + "</div>",
                unsafe_allow_html=True,
            )
            progress_bar.progress(stage_pct.get(stage, 50))
            status_line.caption(msg[:100])

        t0 = time.time()
        results = []
        error_msg = None

        try:
            results = pipeline.run(
                market=market, country=country, region=region,
                quarter=quarter, year=year,
                progress_cb=progress_cb,
            )
        except Exception as e:
            error_msg = str(e)

        elapsed = int(time.time() - t0)
        progress_bar.progress(100)
        status_line.empty()

        st.divider()

        if error_msg:
            st.error(f"Pipeline error: {error_msg}")

        elif not results:
            st.warning("No high-signal companies found. Try a different market or check your network.")

        else:
            # ── SUMMARY METRICS ──────────────────────────────────────────────
            st.markdown(
                f"### 🎯 {len(results)} High-Signal Companies — {market}, {country} {quarter} {year} "
                f"<span style='color:#888;font-size:0.85rem;'>({elapsed}s)</span>",
                unsafe_allow_html=True,
            )

            _high   = sum(1 for r in results if r["dark_demand_score"] >= 75)
            _avg_s  = sum(r["dark_demand_score"] for r in results) // len(results)
            _spaces = [r["estimated_space_sqft"] for r in results if r.get("estimated_space_sqft")]
            _avg_sp = f"{int(sum(_spaces)/len(_spaces)):,} sq ft" if _spaces else "N/A"
            st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0 24px;">
  <div style="background:#fff;border:1.5px solid #e2e8f0;border-top:4px solid #003E51;
              border-radius:0 0 12px 12px;padding:20px 16px;text-align:center;
              box-shadow:0 2px 8px rgba(0,0,0,0.05);">
    <div style="font-size:2.2rem;font-weight:900;color:#003E51;">{len(results)}</div>
    <div style="font-size:0.7rem;color:#64748b;margin-top:6px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.8px;">Companies Found</div>
  </div>
  <div style="background:#fff;border:1.5px solid #e2e8f0;border-top:4px solid #16A34A;
              border-radius:0 0 12px 12px;padding:20px 16px;text-align:center;
              box-shadow:0 2px 8px rgba(0,0,0,0.05);">
    <div style="font-size:2.2rem;font-weight:900;color:#16A34A;">{_high}</div>
    <div style="font-size:0.7rem;color:#64748b;margin-top:6px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.8px;">High Signal (&ge;75)</div>
  </div>
  <div style="background:#fff;border:1.5px solid #e2e8f0;border-top:4px solid #BCDEE6;
              border-radius:0 0 12px 12px;padding:20px 16px;text-align:center;
              box-shadow:0 2px 8px rgba(0,0,0,0.05);">
    <div style="font-size:2.2rem;font-weight:900;color:#003E51;">{_avg_s}<span style="font-size:0.9rem;color:#94a3b8;">/100</span></div>
    <div style="font-size:0.7rem;color:#64748b;margin-top:6px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.8px;">Avg Dark Demand Score</div>
  </div>
  <div style="background:#fff;border:1.5px solid #e2e8f0;border-top:4px solid #BCDEE6;
              border-radius:0 0 12px 12px;padding:20px 16px;text-align:center;
              box-shadow:0 2px 8px rgba(0,0,0,0.05);">
    <div style="font-size:1.5rem;font-weight:900;color:#003E51;">{_avg_sp}</div>
    <div style="font-size:0.7rem;color:#64748b;margin-top:6px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.8px;">Avg Estimated Space</div>
  </div>
</div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── COMPANY CARDS ─────────────────────────────────────────────────
            for r in results:
                s = r["dark_demand_score"]
                badge_cls = "badge-high" if s >= 75 else ("badge-med" if s >= 50 else "badge-watch")
                card_cls  = "card-high"  if s >= 75 else ("card-med"  if s >= 50 else "card-watch")
                badge_lbl = r["badge"].split(" ", 1)[1]

                # Coloured top accent + subtle shadow per company card
                accent = "#16A34A" if s >= 75 else ("#D97706" if s >= 50 else "#2563EB")
                bg     = "#f0fdf4" if s >= 75 else ("#fffbeb" if s >= 50 else "#eff6ff")
                st.markdown(
                    f'''<div style="border-left:5px solid {accent};background:{bg};
                    border-radius:0 10px 10px 0;padding:10px 16px 2px;
                    margin:8px 0 0;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
                    </div>''',
                    unsafe_allow_html=True,
                )
                with st.container():
                    left, right = st.columns([1, 5])

                    with left:
                        score_cls = "score-high" if s >= 75 else ("score-med" if s >= 50 else "score-watch")
                        st.markdown(
                            f'<div class="{score_cls}">{s}</div>'
                            f'<div style="color:#aaa;font-size:0.8rem">/ 100</div>'
                            f'<span class="{badge_cls}">{badge_lbl}</span>',
                            unsafe_allow_html=True,
                        )

                    with right:
                        st.markdown(f"#### {r['company']}")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            val = f"{r['estimated_space_sqft']:,} sq ft" if r.get("estimated_space_sqft") else "TBD"
                            st.markdown(f"**Est. Space**<br>{val}", unsafe_allow_html=True)
                        with c2:
                            val = f"~{r['timeline_months']} months" if r.get("timeline_months") else "12–18 months"
                            st.markdown(f"**Timeline**<br>{val}", unsafe_allow_html=True)
                        with c3:
                            signal_map = {
                                "GCC_expansion":"GCC Expansion","hiring_surge":"Hiring Surge",
                                "new_office":"New Office","relocation":"Relocation",
                                "renewal":"Renewal","none":"—",
                            }
                            st.markdown(
                                f"**Signal**<br>{signal_map.get(r['signal_type'], r['signal_type'])}",
                                unsafe_allow_html=True,
                            )

                        # Key evidence
                        if r.get("key_evidence"):
                            st.markdown(
                                f'<div class="evidence">💡 {r["key_evidence"]}</div>',
                                unsafe_allow_html=True,
                            )

                        # Broker summary
                        if r.get("broker_summary"):
                            with st.expander("📋 Broker Alert"):
                                st.info(r["broker_summary"])

                        # Sources
                        sources = r.get("sources", [])
                        if sources:
                            with st.expander(f"🔗 {len(sources)} Source(s)"):
                                for src in sources:
                                    url   = src.get("url", "")
                                    title = src.get("title", url[:60])
                                    if url:
                                        st.markdown(
                                            f'<a class="src-link" href="{url}" target="_blank">'
                                            f'↗ {title[:80]}</a>',
                                            unsafe_allow_html=True,
                                        )

                    st.divider()

            # Download JSON
            st.download_button(
                "⬇️ Download Results (JSON)",
                data=json.dumps(results, indent=2, default=str),
                file_name=f"dark_demand_{market}_{country}.json",
                mime="application/json",
            )

# ── TAB 2: SCAN HISTORY ──────────────────────────────────────────────────────
with tab_history:
    st.markdown("### 📋 Recent Scans")
    recent = database.get_recent_scans(20)

    if not recent:
        st.info("No scans yet. Run your first scan from the Live Scan tab.")
    else:
        df = pd.DataFrame(recent)
        df["started_at"] = pd.to_datetime(df["started_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df["duration"]   = df["duration_sec"].apply(lambda x: f"{x}s" if x else "—")
        st.dataframe(
            df[["id","region","country","market","quarter","year","status","started_at","duration","total_companies"]],
            use_container_width=True,
            hide_index=True,
        )

        # Load details for a specific scan
        scan_ids = [r["id"] for r in recent if r["status"] == "complete"]
        if scan_ids:
            selected = st.selectbox("View scan details:", scan_ids)
            if selected:
                details = database.get_scan_results(selected)
                if details:
                    for co in details:
                        st.markdown(f"**{co['name']}** — Score: {co['dark_demand_score']}/100")
                        st.caption(f"Signal: {co['signal_type']} | Records: {len(co['sentiment_records'])}")

# ── TAB 3: ML EXPORT ─────────────────────────────────────────────────────────
with tab_ml:
    st.markdown("### 🤖 ML Training Data Export")
    st.markdown(
        "Export structured sentiment records for ML model training. "
        "Each row = one article scored. Raw article text is never stored — "
        "only extracted features."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        exp_market = st.text_input("Filter by market (optional)", placeholder="e.g. Mumbai")
    with col2:
        exp_country = st.text_input("Filter by country (optional)", placeholder="e.g. India")
    with col3:
        min_conf = st.selectbox("Min confidence", ["medium", "high", "low"], index=0)

    if st.button("📥 Generate ML Dataset"):
        records = database.export_ml_dataset(
            market=exp_market or None,
            country=exp_country or None,
            min_confidence=min_conf,
        )

        if not records:
            st.warning("No records match the filters. Run more scans first.")
        else:
            df_ml = pd.DataFrame(records)
            st.success(f"✅ {len(df_ml)} sentiment records ready for ML training")
            st.dataframe(df_ml, use_container_width=True, hide_index=True)

            # Schema explanation
            with st.expander("📖 Column definitions"):
                st.markdown("""
                | Column | Type | ML Use |
                |---|---|---|
                | `sentiment_score` | float -1 to 1 | Target / feature |
                | `demand_signal_strength` | int 0-10 | Target / feature |
                | `signal_type` | categorical | Classification label |
                | `estimated_space_sqft` | int | Regression target |
                | `timeline_months` | int | Regression target |
                | `confidence` | categorical | Sample weight |
                | `market`, `country`, `region` | categorical | Feature |
                | `company_name` | text | Entity feature |
                | `created_at` | timestamp | Time-series feature |
                """)

            # CSV download
            st.download_button(
                "⬇️ Download ML Dataset (CSV)",
                data=df_ml.to_csv(index=False),
                file_name=f"dark_demand_ml_{exp_market or 'all'}_{exp_country or 'all'}.csv",
                mime="text/csv",
            )
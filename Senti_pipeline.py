"""
Senti_pipeline.py — Dark Demand Intelligence orchestrator

ARCHITECTURE:
  ONE Chrome window opens and stays for Phase 1 + Phase 2 scraping.
  LLM scoring runs in parallel threads (HTTP only, no browser).

THREAD SAFETY:
  _score_company() makes ZERO Streamlit/UI calls — it only does HTTP.
  All update_fn/progress_cb calls are made from the main thread only.
  This prevents Streamlit from crashing with silent threading errors.
"""

import gc
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import Senti_config as config
import Senti_database as database
import Senti_scraper as scraper
import Senti_jll_gpt as jll_gpt


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_dark_demand_score(scores: list) -> int:
    if not scores:
        return 0
    conf_map      = {"high": 15, "medium": 8, "low": 3}
    avg_signal    = sum(s.get("demand_signal_strength", 0) for s in scores) / len(scores)
    avg_sentiment = sum(s.get("sentiment_score", 0)        for s in scores) / len(scores)
    top_conf      = max(conf_map.get(s.get("confidence", "low"), 3) for s in scores)
    return min(100, int((avg_signal / 10) * 60 + ((avg_sentiment + 1) / 2) * 25 + top_conf))


def _badge(score: int) -> str:
    if score >= 75: return "🔴 HIGH"
    if score >= 50: return "🟡 MEDIUM"
    return "🟢 WATCH"


def _quarter_to_dates(quarter: str, year: int):
    """Return Google date filter range for the target quarter.
    - Past quarter  → use actual quarter start/end dates
    - Current/future quarter → search past 6 months to today (captures
      forward-looking expansion announcements before the quarter starts)
    """
    from datetime import date, timedelta, datetime
    start_mm_dd, end_mm_dd = config.QUARTER_DATES[quarter]
    q_start = datetime.strptime(f"{start_mm_dd}/{year}", "%m/%d/%Y").date()
    q_end   = datetime.strptime(f"{end_mm_dd}/{year}", "%m/%d/%Y").date()
    today   = date.today()
    if q_start > today:
        # Future quarter: scan recent 6-month lookback for forward-looking news
        lookback = today - timedelta(days=180)
        return lookback.strftime("%m/%d/%Y"), today.strftime("%m/%d/%Y")
    else:
        # Past/current quarter: use actual quarter window
        end_capped = min(q_end, today)          # don't query future end dates
        return q_start.strftime("%m/%d/%Y"), end_capped.strftime("%m/%d/%Y")


# ─────────────────────────────────────────────────────────────────────────────
# LLM SCORER — background thread, NO UI calls whatsoever
# ─────────────────────────────────────────────────────────────────────────────

def _score_company(co_info: dict, articles: list,
                   market: str, country: str,
                   quarter: str, year: int) -> dict | None:
    """
    Score articles for one company using parallel LLM HTTP calls.
    Makes ZERO Streamlit UI calls — safe to run in ThreadPoolExecutor.
    Returns result dict on success, None if below threshold or error.
    """
    company = co_info.get("name", "Unknown")

    try:
        if not articles:
            return None

        article_scores = []
        with ThreadPoolExecutor(max_workers=config.PIPELINE["llm_threads"]) as lx:
            futures = {
                lx.submit(jll_gpt.score_article,
                          art["text"], company, market, country,
                          quarter, year): art   # quarter+year → future-aware prompt
                for art in articles
            }
            for future in as_completed(futures):
                art = futures[future]
                try:
                    score = future.result()
                    score["source_url"]   = art.get("url", "")
                    score["source_title"] = art.get("title", "") or art.get("search_title", "")
                    score["_raw_text"]    = art.get("text", "")
                    article_scores.append(score)
                    art["text"] = None
                except Exception as llm_err:
                    print(f"[LLM] {company} article scoring failed: {llm_err}")

        del articles
        gc.collect()

        if not article_scores:
            return None

        dark_score = compute_dark_demand_score(article_scores)
        if dark_score < config.PIPELINE["min_signal_score"]:
            print(f"[pipeline] {company}: {dark_score}/100 — below threshold")
            del article_scores
            return None

        try:
            summary = jll_gpt.generate_summary(company, dark_score, article_scores, market)
        except Exception:
            summary = co_info.get("signal", "")

        best = max(article_scores, key=lambda s: s.get("demand_signal_strength", 0))

        return {
            "company":              company,
            "dark_demand_score":    dark_score,
            "badge":                _badge(dark_score),
            "signal_type":          best.get("signal_type", "unknown"),
            "estimated_space_sqft": best.get("estimated_space_sqft"),
            "timeline_months":      best.get("timeline_months"),
            "confidence":           best.get("confidence", ""),
            "key_evidence":         best.get("key_evidence", ""),
            "broker_summary":       summary,
            "quarter":              quarter,
            "year":                 year,
            "sources": [
                {"url": s.get("source_url", ""), "title": s.get("source_title", "")}
                for s in article_scores if s.get("source_url")
            ],
            "_article_scores": article_scores,
        }

    except Exception as e:
        # Log to terminal (NOT to Streamlit — we're in a background thread)
        print(f"[pipeline] _score_company FAILED for {company}:")
        print(traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run(market: str, country: str, region: str,
        quarter: str = "2Q", year: int = 2026,
        progress_cb=None) -> list:
    """
    Full pipeline.
    - ONE Chrome browser for all web scraping (Phase 1 + Phase 2 sequential).
    - Parallel LLM scoring via HTTP after browser closes.
    - All progress_cb/update calls from main thread only (thread-safe).
    """

    def update(stage: str, msg: str):
        """Main-thread only — never call from ThreadPoolExecutor workers."""
        if progress_cb:
            progress_cb(stage, msg)
        else:
            print(f"[{stage}] {msg}")

    start_time = time.time()
    scan_id    = database.create_scan(region, country, market, quarter, year)
    driver     = None
    results    = []
    date_start, date_end = _quarter_to_dates(quarter, year)

    try:
        # ── INIT ──────────────────────────────────────────────────────────
        update("init", f"Scanning **{market}, {country}** — **{quarter} {year}** ({region})")
        update("init", f"🗓️  Date range: **{date_start}** → **{date_end}**")
        database.purge_expired_cache()  # removes entries older than TTL (24h); each quarter has its own cache key

        try:
            jll_gpt.refresh_token()
            update("init", "✅ JLL GPT token refreshed")
        except Exception as e:
            update("init", f"⚠️  Using existing token ({e})")

        # ── OPEN ONE BROWSER ──────────────────────────────────────────────
        update("init", "🌐 Opening browser — solve CAPTCHA once, session carries through...")
        driver = scraper.create_driver()
        update("init", "✅ Browser ready")

        # ── PHASE 1: CATEGORY SEARCH ──────────────────────────────────────
        update("phase1", f"**Phase 1** — {len(config.CATEGORIES)} sectors...")
        snippet_text = scraper.run_category_searches(
            driver, market, country,
            date_start=date_start, date_end=date_end,
            year=year,
            progress_cb=progress_cb,
        )

        if not snippet_text.strip():
            update("phase1", "⚠️  No results — check network / CAPTCHA")
            database.fail_scan(scan_id)
            return []

        # ── COMPANY EXTRACTION ─────────────────────────────────────────────
        update("extract", "🤖 Extracting company names from search results...")
        try:
            companies_raw = jll_gpt.extract_companies(snippet_text, market, country)
        except Exception as e:
            update("extract", f"⚠️  Extraction failed: {e}")
            database.fail_scan(scan_id)
            return []

        del snippet_text
        gc.collect()

        if not companies_raw:
            update("extract", "⚠️  No companies identified")
            database.fail_scan(scan_id)
            return []

        conf_order = {"high": 0, "medium": 1, "low": 2}
        companies_raw.sort(key=lambda c: conf_order.get(c.get("confidence", "low"), 2))
        companies = companies_raw[: config.PIPELINE["max_companies_phase2"]]
        del companies_raw
        gc.collect()

        update("extract", f"✅ **{len(companies)} candidates**: " +
               ", ".join(c["name"] for c in companies))

        # ── PHASE 2: COMPANY SEARCHES (sequential, same browser) ─────────
        update("phase2", f"**Phase 2** — searching {len(companies)} companies "
                         f"(1 browser, sequential)...")

        company_articles: dict = {}
        for co_info in companies:
            company = co_info["name"]
            update("phase2", f"  📰 Fetching: **{company}**...")
            try:
                articles = scraper.run_company_searches(
                    driver, company, market, country, year,
                    date_start=date_start, date_end=date_end,
                    progress_cb=progress_cb,
                )
                company_articles[company] = {"info": co_info, "articles": articles}
                update("phase2", f"  ↳ {len(articles)} articles found")
            except Exception as e:
                update("phase2", f"  ⚠️  Search failed for {company}: {e}")
                company_articles[company] = {"info": co_info, "articles": []}

        # ── CLOSE BROWSER — all scraping done ─────────────────────────────
        scraper.quit_driver(driver)
        driver = None
        update("score", "✅ Browser closed — starting LLM scoring...")

        # ── PHASE 2: LLM SCORING ─────────────────────────────────────────
        # Workers run _score_company (NO UI calls).
        # Main thread calls update() after each future completes.
        scoring_threads = min(config.PIPELINE["llm_threads"], len(company_articles))
        raw_results     = []
        total           = len(company_articles)
        done            = 0

        with ThreadPoolExecutor(max_workers=scoring_threads) as executor:
            futures = {
                executor.submit(
                    _score_company,
                    data["info"], data["articles"],
                    market, country,
                    quarter, year,
                ): company
                for company, data in company_articles.items()
            }

            for future in as_completed(futures):
                company = futures[future]
                done += 1
                try:
                    r = future.result()          # Never raises — _score_company catches all
                except Exception as e:
                    # Defensive: should never happen given _score_company's guard
                    update("score", f"  ⚠️  Unexpected error for {company}: {e}")
                    print(traceback.format_exc())
                    r = None

                if r:
                    raw_results.append(r)
                    update("score",
                           f"  ✅ **{company}** — **{r['dark_demand_score']}/100** "
                           f"{r['badge']}  ({done}/{total})")
                else:
                    update("score", f"  ↳ {company}: no signal / below threshold  ({done}/{total})")

        del company_articles
        gc.collect()

        # ── SAVE TO DB ────────────────────────────────────────────────────
        for r in raw_results:
            article_scores = r.pop("_article_scores", [])
            try:
                company_id = database.save_company(scan_id, r)
                database.save_sentiment_records(
                    company_id=company_id, scan_id=scan_id,
                    company_name=r["company"],
                    region=region, country=country, market=market,
                    quarter=quarter, year=year,
                    article_scores=article_scores,
                )
            except Exception as db_err:
                update("score", f"  ⚠️  DB save failed for {r['company']}: {db_err}")
                print(traceback.format_exc())
            finally:
                del article_scores
                gc.collect()

            results.append(r)

        # ── COMPLETE ──────────────────────────────────────────────────────
        elapsed = int(time.time() - start_time)
        database.complete_scan(scan_id, len(results), elapsed)
        results.sort(key=lambda r: r["dark_demand_score"], reverse=True)
        update("complete",
               f"🎯 Done in **{elapsed}s** — **{len(results)} companies** flagged")

    except Exception as e:
        err_detail = traceback.format_exc()
        update("error", f"Pipeline error: {e if str(e) else 'see terminal for traceback'}")
        print(f"[pipeline] FULL TRACEBACK:\n{err_detail}")
        database.fail_scan(scan_id)
        raise

    finally:
        if driver:
            scraper.quit_driver(driver)
        gc.collect()

    return results
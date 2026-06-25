"""
database.py — SQLite manager for Dark Demand Intelligence
Serves two purposes:
  1. Real-time demo: store scan results for display
  2. ML training data: structured sentiment records for future model training

Memory management: raw article text is NEVER stored — only hashes + extracted fields.
"""

import sqlite3
import hashlib
import json
import gc
from datetime import datetime, timedelta
from contextlib import contextmanager
import Senti_config as config


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    """Context manager: open connection, yield, close + GC. Never leak."""
    conn = sqlite3.connect(config.DB["path"])
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA INIT
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
-- One row per user-triggered scan
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    region      TEXT    NOT NULL,
    country     TEXT    NOT NULL,
    market      TEXT    NOT NULL,
    quarter     TEXT    NOT NULL DEFAULT '2Q',   -- 1Q|2Q|3Q|4Q
    year        INTEGER NOT NULL DEFAULT 2026,
    status      TEXT    NOT NULL DEFAULT 'running',   -- running|complete|failed
    started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    duration_sec INTEGER,
    total_companies INTEGER DEFAULT 0
);

-- One row per company identified in a scan (demo display + ML label)
CREATE TABLE IF NOT EXISTS companies (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id              INTEGER NOT NULL,
    name                 TEXT    NOT NULL,
    dark_demand_score    INTEGER NOT NULL,
    signal_type          TEXT,
    estimated_space_sqft INTEGER,
    timeline_months      INTEGER,
    confidence           TEXT,
    broker_summary       TEXT,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

-- One row per article scored — PRIMARY ML TRAINING TABLE
-- Raw article text is NEVER stored here (memory/privacy)
-- Only structured extracted fields + hash for dedup
CREATE TABLE IF NOT EXISTS sentiment_records (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id              INTEGER NOT NULL,
    scan_id                 INTEGER NOT NULL,
    region                  TEXT    NOT NULL,
    country                 TEXT    NOT NULL,
    market                  TEXT    NOT NULL,
    quarter                 TEXT    NOT NULL,
    year                    INTEGER NOT NULL,
    company_name            TEXT    NOT NULL,
    article_url             TEXT,
    article_title           TEXT,
    article_text_hash       TEXT,
    sentiment_score         REAL    NOT NULL,
    demand_signal_strength  INTEGER NOT NULL,
    signal_type             TEXT    NOT NULL,
    estimated_space_sqft    INTEGER,
    timeline_months         INTEGER,
    key_evidence            TEXT    NOT NULL,
    confidence              TEXT    NOT NULL,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE,
    FOREIGN KEY (scan_id)    REFERENCES scans(id)    ON DELETE CASCADE
);

-- Search result cache — avoid re-scraping same query within TTL
-- Results stored as JSON, expired rows purged automatically
CREATE TABLE IF NOT EXISTS search_cache (
    query_hash  TEXT PRIMARY KEY,
    query_text  TEXT,
    results_json TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at  TIMESTAMP NOT NULL
);

-- Index for ML exports: filter by market/country/date range
CREATE INDEX IF NOT EXISTS idx_sentiment_market   ON sentiment_records(market, country, region);
CREATE INDEX IF NOT EXISTS idx_sentiment_company  ON sentiment_records(company_name);
CREATE INDEX IF NOT EXISTS idx_sentiment_signal   ON sentiment_records(signal_type, confidence);
CREATE INDEX IF NOT EXISTS idx_sentiment_date     ON sentiment_records(created_at);
CREATE INDEX IF NOT EXISTS idx_cache_expires      ON search_cache(expires_at);
"""


def init_db():
    """Create all tables and indexes. Safe to call multiple times."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print("✅ Database initialised:", config.DB["path"])


# ─────────────────────────────────────────────────────────────────────────────
# SCAN OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_scan(region: str, country: str, market: str,
                quarter: str = "2Q", year: int = 2026) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO scans (region, country, market, quarter, year, status) VALUES (?,?,?,?,?,'running')",
            (region, country, market, quarter, year)
        )
        return cur.lastrowid


def complete_scan(scan_id: int, total_companies: int, duration_sec: int):
    with get_conn() as conn:
        conn.execute(
            """UPDATE scans SET status='complete', finished_at=CURRENT_TIMESTAMP,
               duration_sec=?, total_companies=? WHERE id=?""",
            (duration_sec, total_companies, scan_id)
        )


def fail_scan(scan_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE scans SET status='failed', finished_at=CURRENT_TIMESTAMP WHERE id=?",
            (scan_id,)
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPANY + SENTIMENT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_company(scan_id: int, result: dict) -> int:
    """Save one company result. Returns company_id."""
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO companies
               (scan_id, name, dark_demand_score, signal_type,
                estimated_space_sqft, timeline_months, confidence, broker_summary)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                scan_id,
                result["company"],
                result["dark_demand_score"],
                result.get("signal_type"),
                result.get("estimated_space_sqft"),
                result.get("timeline_months"),
                result.get("confidence"),
                result.get("broker_summary"),
            )
        )
        return cur.lastrowid


def save_sentiment_records(company_id: int, scan_id: int, company_name: str,
                            region: str, country: str, market: str,
                            quarter: str, year: int,
                            article_scores: list):
    """
    Save all article-level sentiment scores for a company.
    Raw article text is hashed — never stored.
    """
    with get_conn() as conn:
        for score in article_scores:
            raw_text = score.pop("_raw_text", "")
            text_hash = hashlib.sha256(raw_text.encode()).hexdigest() if raw_text else None

            conn.execute(
                """INSERT INTO sentiment_records
                   (company_id, scan_id, region, country, market, quarter, year,
                    company_name, article_url, article_title, article_text_hash,
                    sentiment_score, demand_signal_strength, signal_type,
                    estimated_space_sqft, timeline_months, key_evidence, confidence)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    company_id, scan_id, region, country, market, quarter, year,
                    company_name,
                    score.get("source_url"),
                    score.get("source_title"),
                    text_hash,
                    score.get("sentiment_score", 0.0),
                    score.get("demand_signal_strength", 0),
                    score.get("signal_type", "none"),
                    score.get("estimated_space_sqft"),
                    score.get("timeline_months"),
                    score.get("key_evidence", ""),
                    score.get("confidence", "low"),
                )
            )


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _query_hash(query: str) -> str:
    return hashlib.md5(query.strip().lower().encode()).hexdigest()


def get_cached_search(query: str):
    """Return cached results list if not expired, else None."""
    qhash = _query_hash(query)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT results_json FROM search_cache WHERE query_hash=? AND expires_at > CURRENT_TIMESTAMP",
            (qhash,)
        ).fetchone()
    if row:
        return json.loads(row["results_json"])
    return None


def cache_search(query: str, results: list):
    """Cache search results for TTL hours."""
    qhash = _query_hash(query)
    expires = datetime.now() + timedelta(hours=config.DB["search_cache_ttl_hrs"])
    with get_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO search_cache (query_hash, query_text, results_json, expires_at)
               VALUES (?,?,?,?)""",
            (qhash, query, json.dumps(results), expires.isoformat())
        )


def purge_expired_cache():
    """Delete expired cache rows. Call at app start."""
    with get_conn() as conn:
        deleted = conn.execute(
            "DELETE FROM search_cache WHERE expires_at <= CURRENT_TIMESTAMP"
        ).rowcount
    if deleted:
        print(f"  🗑️  Purged {deleted} expired cache entries")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY / EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_scan_results(scan_id: int) -> list:
    """Fetch all companies + their sentiment records for a completed scan."""
    with get_conn() as conn:
        companies = conn.execute(
            "SELECT * FROM companies WHERE scan_id=? ORDER BY dark_demand_score DESC",
            (scan_id,)
        ).fetchall()
        result = []
        for co in companies:
            sentiments = conn.execute(
                "SELECT * FROM sentiment_records WHERE company_id=?",
                (co["id"],)
            ).fetchall()
            result.append({
                **dict(co),
                "sentiment_records": [dict(s) for s in sentiments]
            })
    return result


def export_ml_dataset(market: str = None, country: str = None,
                       min_confidence: str = "medium") -> list:
    """
    Export structured sentiment records for ML model training.
    Filters by market/country and minimum confidence level.
    Returns list of dicts — ready for pandas DataFrame.
    """
    conf_filter = {"high": ("high",), "medium": ("high", "medium"), "low": ("high", "medium", "low")}
    conf_values = conf_filter.get(min_confidence, ("high", "medium"))
    placeholders = ",".join("?" * len(conf_values))

    query = f"""
        SELECT
            sr.id, sr.created_at, sr.region, sr.country, sr.market,
            sr.company_name, sr.signal_type, sr.sentiment_score,
            sr.demand_signal_strength, sr.estimated_space_sqft,
            sr.timeline_months, sr.key_evidence, sr.confidence,
            sr.article_url, sr.article_title
        FROM sentiment_records sr
        WHERE sr.confidence IN ({placeholders})
    """
    params = list(conf_values)

    if market:
        query += " AND sr.market = ?"
        params.append(market)
    if country:
        query += " AND sr.country = ?"
        params.append(country)

    query += " ORDER BY sr.created_at DESC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(r) for r in rows]


def get_recent_scans(limit: int = 10) -> list:
    """Get most recent scans for history display."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, region, country, market, quarter, year, status,
                      started_at, duration_sec, total_companies
               FROM scans ORDER BY started_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_db_stats() -> dict:
    """Dashboard stats for sidebar."""
    with get_conn() as conn:
        stats = {
            "total_scans":      conn.execute("SELECT COUNT(*) FROM scans WHERE status='complete'").fetchone()[0],
            "total_companies":  conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0],
            "sentiment_records":conn.execute("SELECT COUNT(*) FROM sentiment_records").fetchone()[0],
            "markets_scanned":  conn.execute("SELECT COUNT(DISTINCT market) FROM scans").fetchone()[0],
            "cache_entries":    conn.execute("SELECT COUNT(*) FROM search_cache WHERE expires_at > CURRENT_TIMESTAMP").fetchone()[0],
        }
    return stats

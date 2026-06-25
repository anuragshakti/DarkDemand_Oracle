# =============================================================================
# config.py — DARK DEMAND INTELLIGENCE
# ALL credentials and tuning variables live here.
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  CREDENTIALS TO UPDATE BEFORE EACH SESSION                         │
# │  ──────────────────────────────────────────────────────────────────│
# │  1. refresh_token  — paste from Okta (lasts longer, use this)      │
# │  2. access_token   — paste latest from config.py or Okta console   │
# │  3. token_expiry_ms — set to: int(time.time()*1000) + 3600000      │
# │                                                                     │
# │  Tokens expire ~1 hour. jll_gpt.py auto-refreshes using            │
# │  refresh_token while the session is running.                        │
# └─────────────────────────────────────────────────────────────────────┘
# =============================================================================

from pathlib import Path

_BASE = Path(__file__).parent   # absolute path to this directory

# ── JLL GPT API ───────────────────────────────────────────────────────────────
# ⚠️  UPDATE THESE TOKENS BEFORE EACH DEMO SESSION
JLL_API = {
    "api_url":          "",
    "subscription_key": "",
    "token_url":        "",
    "client_id":        "",

    # ── UPDATE THESE TWO ON TOKEN EXPIRY ──────────────────────────────────────
    "refresh_token":    "",
    "access_token":     "",
    "token_expiry_ms":  ,
    # ─────────────────────────────────────────────────────────────────────────

    "model_fast":       "GPT_35_TURBO",   # Phase 1 company extraction
    "model_smart":      "GPT_4",          # Phase 2 sentiment scoring
}

# ── DATABASE ──────────────────────────────────────────────────────────────────
# Absolute path — works regardless of which directory you run from
DB = {
    "path":                 str(_BASE / "dark_demand.db"),
    "search_cache_ttl_hrs": 24,
}

# ── PIPELINE TUNING ───────────────────────────────────────────────────────────
PIPELINE = {
    "max_companies_phase2":     5,
    "max_results_per_query":    8,
    "max_articles_per_company": 3,
    "article_max_words":        400,
    "min_signal_score":         45,
    "page_load_wait_sec":       2,
    "scroll_pause_sec":         0.5,
    "between_requests_sec":     1.5,
    "phase2_threads":           1,      # Single browser — no parallel Chrome windows
    "llm_threads":              4,
    "phase1_queries_per_cat":   1,
}

# ── SELENIUM / BROWSER ────────────────────────────────────────────────────────
BROWSER = {
    "window_size":  "1920,1080",
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "languages":    ["en-US", "en"],
    "vendor":       "Google Inc.",
    "platform":     "Win32",
    "webgl_vendor": "Intel Inc.",
    "renderer":     "Intel Iris OpenGL Engine",
}

# ── GLOBAL MARKETS ────────────────────────────────────────────────────────────
MARKETS = {
    "APAC": {
        "India":       ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune"],
        "China":       ["Shanghai", "Beijing", "Shenzhen", "Guangzhou"],
        "Indonesia":   ["Jakarta"],
        "Singapore":   ["Singapore"],
        "Australia":   ["Sydney", "Melbourne"],
        "Japan":       ["Tokyo"],
        "South Korea": ["Seoul"],
    },
    "EMEA": {
        "UK":          ["London", "Manchester"],
        "Germany":     ["Frankfurt", "Munich", "Berlin"],
        "France":      ["Paris"],
        "UAE":         ["Dubai", "Abu Dhabi"],
        "Netherlands": ["Amsterdam"],
    },
    "AMER": {
        "USA":         ["New York", "San Francisco", "Chicago", "Los Angeles", "Dallas"],
        "Canada":      ["Toronto", "Vancouver"],
        "Brazil":      ["São Paulo"],
    },
}

# ── QUARTER DATE RANGES ───────────────────────────────────────────────────────
QUARTER_DATES = {
    "1Q": ("01/01", "03/31"),
    "2Q": ("04/01", "06/30"),
    "3Q": ("07/01", "09/30"),
    "4Q": ("10/01", "12/31"),
}

# ── PHASE 1 CATEGORY SEARCHES ─────────────────────────────────────────────────
CATEGORIES = {
    "Financial Services": {
        "description": "Banks, insurance, investment firms",
        "queries": [
            "{market} {country} financial services company office expansion hiring {prev_year} {year}",
            "bank insurance firm {market} new office space leased announced {year}",
            "investment firm {market} {country} corporate office expansion real estate {year}",
        ],
    },
    "Technology": {
        "description": "Tech companies, software, R&D, IT services",
        "queries": [
            "technology company new office {market} {country} expanding {prev_year} {year}",
            "tech company {market} {country} hiring engineers office space leased {year}",
            "software IT company {market} {country} new office operations expansion {year}",
        ],
    },
    "Consulting": {
        "description": "Consulting, professional services, Big4",
        "queries": [
            "consulting firm expanding {market} {country} new office {prev_year} {year}",
            "professional services company {market} {country} office lease expansion {year}",
        ],
    },
    "Healthcare & Pharma": {
        "description": "Pharma, biotech, life sciences",
        "queries": [
            "pharmaceutical company {market} {country} office expansion new operations {year}",
            "healthcare company {market} {country} new office hiring expansion announced {year}",
        ],
    },
    "Manufacturing & Industrial": {
        "description": "Industrial MNCs, engineering, automotive",
        "queries": [
            "manufacturing company {country} office {market} expansion new lease {year}",
            "engineering industrial firm {market} {country} new office headquarters expansion {year}",
        ],
    },
    "Retail & FMCG": {
        "description": "Consumer goods, retail, e-commerce",
        "queries": [
            "retail consumer company {market} {country} corporate office expansion {year}",
            "e-commerce brand company new office {market} {country} lease expansion {year}",
        ],
    },
}

# ── PHASE 2 TARGETED COMPANY QUERIES ─────────────────────────────────────────
COMPANY_QUERIES = [
    "{company} {country} office expansion {year}",
    "{company} {market} hiring new office space {year}",
    "{company} {market} office lease signed {year}",
    "{company} {country} headcount growth new office {year}",
    "{company} {market} real estate office expansion announcement",
]

# ── RELEVANCE FILTER ──────────────────────────────────────────────────────────
RELEVANCE_KEYWORDS = [
    "office", "expand", "hiring", "lease", "headquarter",
    "capability centre", "capability center", "real estate",
    "sq ft", "sqft", "square feet", "headcount", "workspace",
    "new office", "campus", "facility", "operations", "relocation",
]

SKIP_DOMAINS = [
    "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
    "youtube.com", "amazon.com", "flipkart.com", "naukri.com",
    "glassdoor.com", "indeed.com", "wikipedia.org",
]
"""
jll_gpt.py — JLL GPT API wrapper
Handles token refresh, API calls, and JSON parsing.
"""

import re
import time
import json
import requests
from urllib.parse import urlencode
import Senti_config as config


# In-memory token store (updated on refresh)
_token_store = {
    "access_token":    config.JLL_API["access_token"],
    "refresh_token":   config.JLL_API["refresh_token"],
    "token_expiry_ms": config.JLL_API["token_expiry_ms"],
}


def refresh_token():
    """Refresh Okta access token. Call once at app start."""
    print("  🔑 Refreshing JLL GPT access token...")
    data = {
        "grant_type":    "refresh_token",
        "refresh_token": _token_store["refresh_token"],
        "client_id":     config.JLL_API["client_id"],
    }
    r = requests.post(
        config.JLL_API["token_url"],
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=urlencode(data),
        timeout=15,
    )
    if not r.ok:
        raise Exception(f"Token refresh failed: {r.status_code} — {r.text[:200]}")

    tokens = r.json()
    _token_store["access_token"]    = tokens["access_token"]
    _token_store["refresh_token"]   = tokens["refresh_token"]
    _token_store["token_expiry_ms"] = int(time.time() * 1000) + tokens["expires_in"] * 1000
    print("  ✅ Token refreshed successfully")
    return tokens["access_token"]


def get_token():
    """Return a valid access token, refreshing if needed."""
    now_ms = int(time.time() * 1000)
    buffer_ms = 5 * 60 * 1000  # refresh 5 min before expiry
    if _token_store["token_expiry_ms"] - now_ms > buffer_ms:
        return _token_store["access_token"]
    return refresh_token()


def call(messages: list, model: str = None, expect_json: bool = False) -> str:
    """
    Call JLL GPT API.

    Args:
        messages:    OpenAI-format message list
        model:       Override model (default: config fast model)
        expect_json: If True, strip markdown fences and validate JSON

    Returns:
        Response content string (or parsed dict if expect_json=True)
    """
    model = model or config.JLL_API["model_fast"]
    token = get_token()

    r = requests.post(
        config.JLL_API["api_url"],
        headers={
            "Authorization":    f"Bearer {token}",
            "Subscription-Key": config.JLL_API["subscription_key"],
            "Content-Type":     "application/json",
        },
        json={"model": model, "messages": messages},
        timeout=30,
    )

    if not r.ok:
        raise Exception(f"JLL GPT API error: {r.status_code} — {r.text[:200]}")

    content = r.json()["choices"][0]["message"]["content"]

    if expect_json:
        # Strip markdown code fences if model wraps response
        content = re.sub(r"```json\n?|```\n?", "", content).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise Exception(f"JSON parse failed: {e}\nRaw content: {content[:300]}")

    return content


def extract_companies(snippets: str, market: str, country: str) -> list:
    """
    Phase 1 — Extract company names from Google search snippets.
    Returns list of {name, signal, confidence} dicts.
    """
    result = call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a commercial real estate analyst identifying companies "
                    "likely to lease office space. Return only valid JSON, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"From these search results about companies hiring or expanding in "
                    f"{market}, {country}, extract company names showing office demand signals.\n\n"
                    f"Search results:\n{snippets}\n\n"
                    f"Return JSON:\n"
                    f'{{ "companies": [{{"name": "string", "signal": "string", "confidence": "high|medium|low"}}] }}'
                ),
            },
        ],
        model=config.JLL_API["model_fast"],
        expect_json=True,
    )
    return result.get("companies", [])


def score_article(article_text: str, company: str, market: str, country: str,
                  quarter: str = None, year: int = None) -> dict:
    """
    Phase 2 — Score a single article for dark demand signals.
    When quarter/year supplied and is a future period, the prompt reasons about
    whether the described event has already occurred or is likely to occur in
    the target quarter — enabling forward-looking Dark Demand scoring.
    Returns sentiment and demand signal fields.
    """
    from datetime import date, datetime

    # Determine if we are forecasting a future quarter
    is_future = False
    target_label = ""
    if quarter and year:
        from Senti_config import QUARTER_DATES
        start_mm_dd = QUARTER_DATES[quarter][0]
        try:
            q_start = datetime.strptime(f"{start_mm_dd}/{year}", "%m/%d/%Y").date()
            is_future = q_start > date.today()
        except Exception:
            pass
        target_label = f"{quarter} {year}"

    if is_future:
        system_msg = (
            "You are a commercial real estate forward-demand analyst. "
            "You receive articles published in the last 6 months and must "
            "infer whether a company is likely to sign an office lease in the "
            "target future quarter. Return only valid JSON, no markdown."
        )
        user_msg = (
            f"Target quarter: {target_label} (this quarter is IN THE FUTURE — "
            f"articles are from the last 6 months as forward-looking signals).\n"
            f"Company: {company} | Market: {market}, {country}\n\n"
            f"Article:\n{article_text}\n\n"
            f"Reason step by step (internally) then return JSON:\n"
            f"1. Has the described expansion event ALREADY HAPPENED before today?\n"
            f"2. If not yet happened, does evidence suggest it will occur in {target_label}?\n"
            f"3. What is the demand signal strength for future lease activity in {market}?\n\n"
            f"Return JSON:\n"
            f"{{\n"
            f'  "sentiment_score": <float -1.0 to 1.0>,\n'
            f'  "demand_signal_strength": <int 0-10, higher if expansion is planned for target quarter>,\n'
            f'  "signal_type": "planned_expansion|hiring_surge|new_office|announced_lease|relocation|none",\n'
            f'  "event_status": "already_happened|in_progress|planned_for_target_quarter|uncertain",\n'
            f'  "estimated_space_sqft": <int or null>,\n'
            f'  "timeline_months": <int or null, months until likely lease signing>,\n'
            f'  "key_evidence": "one sentence — what in the article suggests {target_label} activity",\n'
            f'  "confidence": "high|medium|low"\n'
            f"}}"
        )
    else:
        system_msg = (
            "You are a commercial real estate demand analyst. "
            "Return only valid JSON, no markdown."
        )
        user_msg = (
            f"Analyze this article for office space demand signals.\n"
            f"Company: {company} | Market: {market}, {country}"
            + (f" | Quarter: {target_label}" if target_label else "") + "\n\n"
            f"Article:\n{article_text}\n\n"
            f"Return JSON:\n"
            f"{{\n"
            f'  "sentiment_score": <float -1.0 to 1.0>,\n'
            f'  "demand_signal_strength": <int 0-10>,\n'
            f'  "signal_type": "hiring_surge|new_office|announced_lease|relocation|renewal|none",\n'
            f'  "estimated_space_sqft": <int or null>,\n'
            f'  "timeline_months": <int or null>,\n'
            f'  "key_evidence": "one sentence max",\n'
            f'  "confidence": "high|medium|low"\n'
            f"}}"
        )

    return call(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        model=config.JLL_API["model_smart"],
        expect_json=True,
    )


def generate_summary(company: str, score: int, signals: list, market: str) -> str:
    """
    Generate a broker-friendly natural language summary for the dashboard.
    """
    signal_text = "\n".join(
        f"- {s.get('key_evidence', '')} (confidence: {s.get('confidence', '')})"
        for s in signals[:3]
    )
    return call(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JLL leasing broker assistant. "
                    "Write concise, professional, actionable insights."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Write a 2-sentence broker alert for this dark demand signal.\n\n"
                    f"Company: {company}\n"
                    f"Dark Demand Score: {score}/100\n"
                    f"Market: {market}\n"
                    f"Evidence:\n{signal_text}\n\n"
                    f"Format: '[Company] is showing strong/moderate expansion signals in [market]. "
                    f"[Specific evidence]. Recommended action: [broker action].'"
                ),
            },
        ],
        model=config.JLL_API["model_fast"],
        expect_json=False,
    )
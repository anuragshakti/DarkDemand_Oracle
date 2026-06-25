"""
scraper.py — Selenium Stealth Google scraper + article parser
Driver: reads Chrome version from Windows registry, downloads ChromeDriver
directly via requests (verify=False) — bypasses JLL proxy SSL issues.
No webdriver_manager, no headless (Google blocks headless).
"""

import re
import io
import os
import time
import gc
import random
import zipfile
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from urllib.parse import quote_plus, urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium_stealth import stealth
from bs4 import BeautifulSoup

import Senti_config as config
import Senti_database as database

# Script directory for caching ChromeDriver
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# CHROME VERSION — Windows registry (no internet needed)
# ─────────────────────────────────────────────────────────────────────────────

def _get_chrome_version() -> str | None:
    """Read Chrome version from Windows registry. Returns e.g. '124.0.6367.82'."""
    try:
        import winreg
        reg_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Google\Chrome\BLBeacon"),
            (winreg.HKEY_CURRENT_USER,  r"SOFTWARE\Google\Chrome\BLBeacon"),
            (winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\Google\Update\Clients\{8A69D345-D564-463C-AFF1-A69D9E530F96}"),
        ]
        for hive, path in reg_paths:
            try:
                key = winreg.OpenKey(hive, path)
                version, _ = winreg.QueryValueEx(key, "version")
                winreg.CloseKey(key)
                if version:
                    return version
            except Exception:
                pass
    except ImportError:
        pass  # Not on Windows — fall through
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHROMEDRIVER DOWNLOAD — direct URL, verify=False (JLL proxy safe)
# ─────────────────────────────────────────────────────────────────────────────

def _get_chromedriver_path() -> str:
    """
    Download ChromeDriver matching installed Chrome.
    Uses storage.googleapis.com with verify=False — works through JLL proxy.
    Caches chromedriver.exe locally so download only happens once per version.
    """
    cache_dir = os.path.join(_SCRIPT_DIR, ".chromedriver_cache")
    os.makedirs(cache_dir, exist_ok=True)

    chrome_version = _get_chrome_version()
    if not chrome_version:
        raise RuntimeError(
            "Could not read Chrome version from Windows registry.\n"
            "Ensure Google Chrome is installed."
        )

    major = chrome_version.split(".")[0]
    print(f"   Chrome {chrome_version} (major: {major})")

    cached = os.path.join(cache_dir, f"chromedriver_{chrome_version}_win64.exe")
    if os.path.exists(cached):
        print(f"   Using cached ChromeDriver")
        return cached

    base = "https://storage.googleapis.com/chrome-for-testing-public"
    url  = f"{base}/{chrome_version}/win64/chromedriver-win64.zip"
    print(f"   Downloading ChromeDriver...")

    dl = requests.get(url, verify=False, timeout=60, stream=True)

    if dl.status_code == 404:
        print(f"   Exact version not found — fetching latest for milestone {major}...")
        latest_resp = requests.get(
            f"https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_{major}",
            verify=False, timeout=15
        )
        if latest_resp.ok:
            url = f"{base}/{latest_resp.text.strip()}/win64/chromedriver-win64.zip"
            dl  = requests.get(url, verify=False, timeout=60, stream=True)

    if not dl.ok:
        raise RuntimeError(f"ChromeDriver download failed: {dl.status_code} — {url}")

    with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
        for name in zf.namelist():
            if name.endswith("chromedriver.exe"):
                with zf.open(name) as src, open(cached, "wb") as dst:
                    dst.write(src.read())
                print(f"   ChromeDriver cached")
                return cached

    raise RuntimeError("chromedriver.exe not found in downloaded zip.")


# ─────────────────────────────────────────────────────────────────────────────
# DRIVER FACTORY — visible browser, stealth patches, no webdriver_manager
# ─────────────────────────────────────────────────────────────────────────────

def create_driver() -> webdriver.Chrome:
    """
    Create a stealth Chrome driver.
    - NO headless (Google actively blocks headless Chrome)
    - ChromeDriver downloaded directly via requests (JLL proxy safe)
    - Stealth patches applied to avoid bot detection
    """
    driver_path = _get_chromedriver_path()
    service     = Service(driver_path)

    opts = Options()
    # ── NO headless — Google blocks it ──
    opts.add_argument("--start-maximized")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-notifications")
    opts.add_argument("--ignore-certificate-errors")
    opts.add_argument("--accept-lang=en-US,en;q=0.9")
    opts.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=service, options=opts)

    # JS stealth patches
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": """
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        if (!window.chrome) window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        Object.defineProperty(navigator, 'plugins',   { get: () => [1,2,3,4,5] });
    """})

    stealth(
        driver,
        languages=config.BROWSER["languages"],
        vendor=config.BROWSER["vendor"],
        platform=config.BROWSER["platform"],
        webgl_vendor=config.BROWSER["webgl_vendor"],
        renderer=config.BROWSER["renderer"],
        fix_hairline=True,
    )

    driver.set_page_load_timeout(30)
    return driver


def quit_driver(driver):
    """Safely quit driver and free memory."""
    try:
        driver.quit()
    except Exception:
        pass
    finally:
        del driver
        gc.collect()


def is_driver_alive(driver) -> bool:
    try:
        _ = driver.window_handles
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CAPTCHA HANDLING — pause and wait for user to solve, then auto-continue
# ─────────────────────────────────────────────────────────────────────────────

def _is_captcha(driver) -> bool:
    """Detect genuine CAPTCHA/block pages only — avoid false positives."""
    try:
        title    = driver.title.lower().strip()
        body_len = driver.execute_script(
            "return document.body ? document.body.innerText.length : 0;"
        )
        src = driver.page_source.lower()

        hard = ["unusual traffic", "verify you're not a robot",
                "confirm you're not a robot", "are you a robot"]
        soft = ["before you continue", "i'm not a robot", "recaptcha"]

        if any(p in title for p in hard): return True
        if any(p in src   for p in hard): return True
        if body_len < 800 and any(p in src or p in title for p in soft):
            return True
        return False
    except Exception:
        return False


def wait_for_captcha(driver):
    """
    If CAPTCHA detected: print message and poll every 3s until cleared.
    User solves it in the visible browser — script auto-continues.
    """
    if not _is_captcha(driver):
        return

    print("\n" + "=" * 60)
    print("  CAPTCHA detected — please solve it in the browser.")
    print("  Script will continue automatically once cleared.")
    print("=" * 60)

    while _is_captcha(driver):
        time.sleep(3)

    print("  CAPTCHA cleared — continuing.\n")
    time.sleep(1)




# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE SEARCH
# ─────────────────────────────────────────────────────────────────────────────



def google_search(driver, query: str,
                  date_start: str = None, date_end: str = None) -> list:
    """
    Execute one Google search by typing into the search box (human-like).
    Cache key = query + date range so each quarter gets its own cache entry.
    Returns list of {title, url, snippet}.
    """
    # Include date range in cache key — different quarters never share a cache entry
    _cache_key = query if not date_start else f"{query}|{date_start}|{date_end}"
    cached = database.get_cached_search(_cache_key)
    if cached is not None:
        return cached

    results = []
    try:
        driver.get("https://www.google.com")
        time.sleep(random.uniform(1.5, 2.5))
        wait_for_captcha(driver)

        # Dismiss cookie consent if present
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for btn in buttons:
                try:
                    if "accept" in (btn.text or "").lower():
                        btn.click()
                        time.sleep(0.8)
                        break
                except Exception:
                    pass
        except Exception:
            pass

        # Type query character by character — more human-like
        box = driver.find_element(By.NAME, "q")
        box.clear()
        for char in query:
            box.send_keys(char)
            time.sleep(random.uniform(0.04, 0.10))
        time.sleep(random.uniform(0.3, 0.6))
        box.send_keys(Keys.RETURN)
        time.sleep(random.uniform(2.5, 3.5))
        wait_for_captcha(driver)

        # Apply date filter if specified — navigate to filtered URL
        if date_start and date_end:
            current_url = driver.current_url
            if "tbs=" not in current_url:
                filtered = (current_url +
                            f"&tbs=cdr:1,cd_min:{date_start},cd_max:{date_end}")
                driver.get(filtered)
                time.sleep(random.uniform(2, 3))
                wait_for_captcha(driver)

        results = _parse_google_html(driver.page_source)

    except Exception as e:
        print(f"    ⚠️  Search error: {e}")
    finally:
        gc.collect()

    database.cache_search(_cache_key, results)
    time.sleep(config.PIPELINE["between_requests_sec"])
    return results


def _parse_google_html(html: str) -> list:
    """Parse Google SERP HTML into list of {title, url, snippet}."""
    soup = BeautifulSoup(html, "lxml")
    results = []

    # Google uses div.g for standard results
    for div in soup.find_all("div", class_=re.compile(r"\bg\b|tF2Cxc|yuRUbf")):
        h3 = div.find("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)
        if not title or len(title) < 5:
            continue

        # Extract URL
        a_tag = div.find("a", href=True)
        raw_href = a_tag["href"] if a_tag else ""
        # Unwrap /url?q=... Google redirect
        url_match = re.search(r"/url\?q=([^&]+)", raw_href)
        clean_url = url_match.group(1) if url_match else raw_href
        # Skip non-http
        if not clean_url.startswith("http"):
            continue

        # Extract snippet
        snip_div = div.find(class_=re.compile(r"VwiC3b|s3v9rd|IsZvec|lEBKkf"))
        snippet = snip_div.get_text(strip=True)[:200] if snip_div else ""

        results.append({"title": title, "url": clean_url, "snippet": snippet})

    del soup
    gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ARTICLE FETCHER + CLEANER
# ─────────────────────────────────────────────────────────────────────────────

def _is_skip_domain(url: str) -> bool:
    try:
        return any(d in urlparse(url).netloc for d in config.SKIP_DOMAINS)
    except Exception:
        return False


def fetch_article(driver, url: str) -> dict:
    """
    Fetch article, clean HTML, return {text, title, url}.
    Returns None if page is invalid or too short.
    Raw text is returned for hashing — caller must not persist it.
    """
    if _is_skip_domain(url):
        return None

    try:
        driver.get(url)
        time.sleep(config.PIPELINE["page_load_wait_sec"])

        # Try to wait for article content
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except TimeoutException:
            pass

        html = driver.page_source
        page_title = driver.title

    except Exception:
        return None

    clean = _clean_article_html(html)
    del html
    gc.collect()

    if len(clean.split()) < 60:
        return None

    return {"text": clean, "title": page_title, "url": url}


def _clean_article_html(html: str) -> str:
    """
    Strip noise from article HTML.
    Returns clean text truncated to article_max_words.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noise tags
    for tag in soup(["nav", "header", "footer", "script", "style",
                     "iframe", "form", "button", "aside", "noscript"]):
        tag.decompose()

    # Remove noise by class/id pattern
    noise_re = re.compile(
        r"ad|banner|popup|sidebar|related|social|share|"
        r"comment|cookie|newsletter|promo|widget|menu",
        re.IGNORECASE,
    )
    for tag in soup.find_all(True, {"class": noise_re}):
        tag.decompose()
    for tag in soup.find_all(True, {"id": noise_re}):
        tag.decompose()

    # Prefer article body
    body = (
        soup.find("article")
        or soup.find(class_=re.compile(r"article|story|post|content|body", re.I))
        or soup.find("main")
        or soup.body
    )

    text = body.get_text(separator=" ", strip=True) if body else ""
    text = re.sub(r"\s+", " ", text).strip()

    del soup
    gc.collect()

    return " ".join(text.split()[: config.PIPELINE["article_max_words"]])


# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def is_relevant(title: str, market: str, country: str, snippet: str = "") -> bool:
    """Check title + snippet — expansion news is usually in the snippet, not just title."""
    text = (title + " " + snippet + " " + market + " " + country).lower()
    return any(kw in text for kw in config.RELEVANCE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CATEGORY SEARCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_category_searches(driver, market: str, country: str,
                          date_start: str = None, date_end: str = None,
                          year: int = None, progress_cb=None) -> str:
    """
    Run category searches with optional date range.
    Uses only first query per category (speed mode).
    """
    all_snippets = []
    max_q = config.PIPELINE.get("phase1_queries_per_cat", 1)

    for cat_name, cat_data in config.CATEGORIES.items():
        if progress_cb:
            progress_cb("phase1", f"Scanning **{cat_name}**...")

        for tmpl in cat_data["queries"][:max_q]:
            _yr  = year or __import__('datetime').date.today().year
            _pyr = _yr - 1
            query = tmpl.format(market=market, country=country,
                                year=_yr, prev_year=_pyr)
            results = google_search(driver, query,
                                    date_start=date_start, date_end=date_end)

            for r in results:
                if is_relevant(r["title"], market, country, r.get("snippet", "")):
                    all_snippets.append(f"{r['title']} — {r['snippet'][:120]}")

    if progress_cb:
        progress_cb("phase1", f"✅ {len(all_snippets)} relevant results from {len(config.CATEGORIES)} sectors")

    snippet_text = "\n".join(all_snippets)
    del all_snippets
    gc.collect()
    return snippet_text


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TARGETED COMPANY SEARCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_company_searches(driver, company: str, market: str, country: str,
                         year: int, date_start: str = None, date_end: str = None,
                         progress_cb=None) -> list:
    """
    Run targeted searches + fetch articles for one company.
    Returns list of article dicts. Raw text included for scoring — 
    caller must hash and discard text before DB save.
    """
    articles = []
    seen_urls = set()
    max_articles = config.PIPELINE["max_articles_per_company"]

    for tmpl in config.COMPANY_QUERIES:
        if len(articles) >= max_articles:
            break

        query = tmpl.format(company=company, market=market,
                            country=country, year=year)
        results = google_search(driver, query,
                                date_start=date_start, date_end=date_end)

        for r in results:
            if len(articles) >= max_articles:
                break
            if r["url"] in seen_urls:
                continue
            if not is_relevant(r["title"], market, country, r.get("snippet", "")):
                continue

            seen_urls.add(r["url"])

            if progress_cb:
                progress_cb("phase2", f"  📰 Reading: {r['title'][:65]}...")

            article = fetch_article(driver, r["url"])
            if article:
                article["search_title"] = r["title"]
                articles.append(article)

    del seen_urls
    gc.collect()
    return articles
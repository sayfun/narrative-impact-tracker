"""
mediacloud_client.py
--------------------
MediaCloud API client for the Narrative Impact Tracker.

MediaCloud provides academic access to millions of news stories across
thousands of outlets, returning actual article text rather than GDELT's
headline-only metadata. This enables full-text ERS/PCF/NCS extraction.

API key: register free at https://search.mediacloud.org/register
Documentation: https://github.com/mitmedialab/MediaCloud-API-Client

Rate limits: polite usage; no hard limit for academic keys.
Coverage notes:
  - Strong on US/UK/international English-language press.
  - story_text is typically the full article body (not truncated).
  - date filter uses UTC; publish_date in response is UTC.
  - max 500 stories per request; paginate for longer windows.
"""

import time
import requests
import pandas as pd
from typing import Optional


# ── constants ─────────────────────────────────────────────────────────────────

MC_API_BASE      = "https://api.mediacloud.org/api/v2"
REQUEST_TIMEOUT  = 30
RATE_LIMIT_DELAY = 0.5   # seconds between requests
MAX_RETRIES      = 3
MAX_PER_REQUEST  = 500


# ── request helper ────────────────────────────────────────────────────────────

def _mc_get(endpoint: str, params: dict, api_key: str) -> dict:
    """GET a MediaCloud v2 endpoint with exponential backoff."""
    url    = f"{MC_API_BASE}/{endpoint}"
    params = {**params, "key": api_key}
    delay  = 2.0

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 503) or resp.status_code >= 500:
            if attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()

    return {}


# ── story search ──────────────────────────────────────────────────────────────

def search_stories(
    query:           str,
    start:           "str | pd.Timestamp",
    end:             "str | pd.Timestamp",
    api_key:         str,
    max_stories:     int  = 200,
    include_text:    bool = True,
    language:        str  = "en",
) -> pd.DataFrame:
    """
    Search MediaCloud for news stories matching *query* in the date window.

    Parameters
    ----------
    query        : Free-text or Solr query string. Boolean AND/OR/NOT supported.
                   Example: '"prediction market" AND (Iran OR attack)'
    start, end   : Date window. Strings or Timestamps; converted to UTC.
    api_key      : MediaCloud API key.
    max_stories  : Maximum stories to return (hard cap: 500/request).
    include_text : Request full story body text (recommended; slightly slower).
    language     : ISO language code filter ('en' = English only). Pass '' to skip.

    Returns
    -------
    DataFrame with columns:
        date, title, url, domain, language, text, word_count,
        media_name, stories_id
    """
    if not api_key or not api_key.strip():
        raise ValueError("MediaCloud API key required. Get one free at https://search.mediacloud.org/register")

    start_ts = _to_utc(start)
    end_ts   = _to_utc(end)

    # MediaCloud Solr date filter format
    date_fq = (
        f"publish_date:[{start_ts.strftime('%Y-%m-%dT%H:%M:%SZ')} "
        f"TO {end_ts.strftime('%Y-%m-%dT%H:%M:%SZ')}]"
    )

    fq_parts = [date_fq]
    if language:
        fq_parts.append(f"language:{language}")

    params = {
        "q":    query,
        "fq":   " AND ".join(fq_parts),
        "rows": min(max_stories, MAX_PER_REQUEST),
        "sort": "publish_date asc",
    }
    if include_text:
        params["text"] = 1

    try:
        data = _mc_get("stories/list", params, api_key)
    except Exception as e:
        raise RuntimeError(
            f"MediaCloud API error: {e}\n"
            "Check your API key and network connection. "
            "Keys are free at https://search.mediacloud.org/register"
        ) from e

    # MediaCloud v2 wraps results in response.docs
    docs = (
        data.get("response", {}).get("docs", [])
        or data.get("stories", [])   # fallback for v2 variations
    )
    if not docs:
        return pd.DataFrame(columns=["date", "title", "url", "domain",
                                      "language", "text", "word_count",
                                      "media_name", "stories_id"])

    rows = []
    for s in docs:
        raw_text = s.get("story_text") or s.get("text", "")
        rows.append({
            "date":       _parse_mc_date(s.get("publish_date", "")),
            "title":      s.get("title", ""),
            "url":        s.get("url", ""),
            "domain":     _extract_domain(s.get("media_url", "") or s.get("url", "")),
            "language":   s.get("language", ""),
            "text":       raw_text,
            "word_count": len(str(raw_text).split()) if raw_text else 0,
            "media_name": s.get("media_name", ""),
            "stories_id": str(s.get("stories_id", "")),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    return df.sort_values("date").reset_index(drop=True)


def search_stories_windowed(
    query:          str,
    start:          "str | pd.Timestamp",
    end:            "str | pd.Timestamp",
    api_key:        str,
    window_days:    int  = 14,
    max_per_window: int  = 200,
    include_text:   bool = True,
    verbose:        bool = True,
) -> pd.DataFrame:
    """
    Paginate search_stories() by splitting the date range into sub-windows.

    Use this for windows longer than ~2 weeks to stay within the 500-story
    cap and to avoid timeouts on very large queries.
    """
    from datetime import timedelta

    start_ts = _to_utc(start)
    end_ts   = _to_utc(end)

    all_frames = []
    cursor = start_ts

    while cursor < end_ts:
        window_end = min(cursor + timedelta(days=window_days), end_ts)
        if verbose:
            print(f"  MediaCloud: {cursor.date()} → {window_end.date()} …", end=" ")
        try:
            chunk = search_stories(
                query, start=cursor, end=window_end,
                api_key=api_key, max_stories=max_per_window,
                include_text=include_text,
            )
            if verbose:
                print(f"{len(chunk)} stories")
            if not chunk.empty:
                all_frames.append(chunk)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

        cursor = window_end
        time.sleep(RATE_LIMIT_DELAY)

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["url"]).sort_values("date").reset_index(drop=True)
    return result


# ── query builder ─────────────────────────────────────────────────────────────

def build_mediacloud_query(
    topic_terms:         list,
    include_polymarket:  bool = True,
) -> str:
    """
    Build a MediaCloud Solr query for prediction-market coverage of a topic.

    Mirrors gdelt.build_prediction_market_query() but uses MediaCloud's
    Solr syntax (quoted phrases, uppercase AND/OR).

    Example
    -------
    >>> build_mediacloud_query(["Iran", "attack", "war"])
    '("polymarket" OR "prediction market") AND (Iran OR attack OR war)'
    """
    if include_polymarket:
        market_clause = '("polymarket" OR "prediction market")'
    else:
        market_clause = '("prediction market" OR "betting odds")'

    capped = topic_terms[:4]
    topic_clause = " OR ".join(
        f'"{t}"' if " " in t else t for t in capped
    )

    return f"{market_clause} AND ({topic_clause})"


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_utc(dt: "str | pd.Timestamp") -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _parse_mc_date(s: str) -> pd.Timestamp:
    """Parse MediaCloud publish_date strings (various ISO formats)."""
    if not s:
        return pd.NaT
    try:
        # MediaCloud v2 returns "2026-01-15 10:30:00" (space-separated, no Z)
        ts = pd.Timestamp(s.replace(" ", "T") + ("Z" if "Z" not in s and "+" not in s else ""))
        return ts if ts.tzinfo is not None else ts.tz_localize("UTC")
    except Exception:
        return pd.NaT


def _extract_domain(url: str) -> str:
    """Strip protocol and path from a URL to get the domain."""
    return url.replace("https://", "").replace("http://", "").split("/")[0]


# ── validation helper ─────────────────────────────────────────────────────────

def validate_api_key(api_key: str) -> bool:
    """
    Quick check that an API key is accepted by MediaCloud.
    Returns True if the key is valid, False otherwise.
    """
    if not api_key or not api_key.strip():
        return False
    try:
        data = _mc_get("auth/profile", {}, api_key)
        return bool(data)
    except Exception:
        return False

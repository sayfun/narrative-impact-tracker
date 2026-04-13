"""
gdelt.py
--------
GDELT API client for the Narrative Impact Tracker.

Uses two GDELT v2 Doc API modes:
  1. TimelineVol   – daily article-count time series for a query  (quantitative layer)
  2. ArtList       – article metadata list with tone scores        (qualitative layer)

No authentication required. Rate limit: ~1 req/sec.

GDELT coverage notes (be honest with your reviewers):
  - Skews toward English-language wire services and major outlets.
  - Full article text is NOT returned — only metadata + derived NLP features.
  - Tone score is GDELT's own sentiment (-100 to +100); treat as a proxy.
  - For full-text analysis you must fetch the article URLs yourself.
"""

import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional


# ── constants ────────────────────────────────────────────────────────────────

GDELT_DOC_API  = "https://api.gdeltproject.org/api/v2/doc/doc"
REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 1.1       # seconds between requests (stay within GDELT limits)


# ── helpers ───────────────────────────────────────────────────────────────────

def _gdelt_datetime(dt: "str | datetime | pd.Timestamp") -> str:
    """Convert to GDELT's YYYYMMDDHHMMSS format."""
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.strftime("%Y%m%d%H%M%S")


def _parse_gdelt_date(s: str) -> pd.Timestamp:
    """Parse GDELT's YYYYMMDDTHHMMSSZ article date strings."""
    try:
        return pd.Timestamp(s, tz="UTC")
    except Exception:
        return pd.NaT


# ── timeline volume (coverage intensity) ─────────────────────────────────────

def fetch_coverage_timeline(
    query:       str,
    start:       "str | datetime | pd.Timestamp",
    end:         "str | datetime | pd.Timestamp | None" = None,
    smooth:      bool = True,
) -> pd.DataFrame:
    """
    Fetch a daily article-count time series from GDELT for *query*.

    This is your primary quantitative coverage variable — the 'how much
    media attention did this topic receive on day X' measure.

    Parameters
    ----------
    query  : GDELT query string. Supports boolean operators.
             Example: '"polymarket" OR "prediction market" (Trump OR Biden)'
    start  : Window start.
    end    : Window end (defaults to now).
    smooth : If True, return 3-day rolling mean alongside raw counts.

    Returns
    -------
    DataFrame with columns:
        date          (datetime64[ns, UTC])
        article_count (int)
        volume_norm   (float, normalised 0–1 within window)
        rolling_3d    (float, 3-day rolling mean, if smooth=True)
    """
    if end is None:
        end = datetime.now(timezone.utc)

    params = {
        "query":      query,
        "mode":       "TimelineVol",
        "format":     "json",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime":   _gdelt_datetime(end),
    }

    resp = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    # GDELT occasionally returns an empty body on valid requests
    if not resp.text.strip():
        return pd.DataFrame(columns=["date", "article_count", "volume_norm"])
    data = resp.json()

    # GDELT TimelineVol returns: {"timeline": [{"data": [{"date": ..., "value": ...}]}]}
    series_data = []
    timeline = data.get("timeline", [])
    if timeline:
        # handle both "series" (old) and "data" (current) field names
        series_data = timeline[0].get("data") or timeline[0].get("series", [])

    if not series_data:
        return pd.DataFrame(columns=["date", "article_count", "volume_norm"])

    df = pd.DataFrame(series_data)
    df = df.rename(columns={"date": "date_str", "value": "volume_intensity"})
    df["date"] = pd.to_datetime(df["date_str"], format="%Y%m%dT%H%M%SZ", utc=True)
    # volume_intensity is GDELT's relative volume metric (not raw article count).
    # It represents share of global GDELT coverage on that day — multiply by ~10M
    # for a rough article-count proxy, but treat as ordinal in analyses.
    df["volume_intensity"] = pd.to_numeric(df["volume_intensity"], errors="coerce").fillna(0)
    # Keep 'article_count' as an alias for backward compatibility (treat as intensity proxy)
    df["article_count"] = df["volume_intensity"]

    # Normalise within window
    max_val = df["volume_intensity"].max()
    df["volume_norm"] = df["volume_intensity"] / max_val if max_val > 0 else 0.0

    if smooth:
        df["rolling_3d"] = df["volume_intensity"].rolling(3, center=True, min_periods=1).mean()

    df = df[["date", "volume_intensity", "article_count", "volume_norm"] + (["rolling_3d"] if smooth else [])]
    return df.sort_values("date").reset_index(drop=True)


# ── article list (metadata + tone) ────────────────────────────────────────────

def fetch_articles(
    query:      str,
    start:      "str | datetime | pd.Timestamp",
    end:        "str | datetime | pd.Timestamp | None" = None,
    max_records: int = 250,
) -> pd.DataFrame:
    """
    Fetch a list of articles matching *query* with tone scores.

    GDELT returns up to 250 records per request. For longer windows,
    use fetch_articles_windowed() to paginate by sub-period.

    Returns
    -------
    DataFrame with columns:
        date, title, url, domain, tone, positive_score, negative_score,
        polarity, activity_ref_density, word_count
    """
    if end is None:
        end = datetime.now(timezone.utc)

    params = {
        "query":         query,
        "mode":          "ArtList",
        "maxrecords":    min(max_records, 250),
        "format":        "json",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime":   _gdelt_datetime(end),
        "sort":          "DateDesc",
    }

    resp = requests.get(GDELT_DOC_API, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    if not resp.text.strip():
        return pd.DataFrame()
    data = resp.json()

    articles = data.get("articles", [])
    if not articles:
        return pd.DataFrame()

    rows = []
    for a in articles:
        rows.append({
            "date":                  _parse_gdelt_date(a.get("seendate", "")),
            "title":                 a.get("title", ""),
            "url":                   a.get("url", ""),
            "domain":                a.get("domain", ""),
            "language":              a.get("language", ""),
            "tone":                  float(a.get("tone", 0) or 0),
            "positive_score":        float(a.get("positive", 0) or 0),
            "negative_score":        float(a.get("negative", 0) or 0),
            "polarity":              float(a.get("polarity", 0) or 0),
            "activity_ref_density":  float(a.get("activityrefdensity", 0) or 0),
            "word_count":            int(a.get("wordcount", 0) or 0),
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def fetch_articles_windowed(
    query:       str,
    start:       "str | datetime | pd.Timestamp",
    end:         "str | datetime | pd.Timestamp | None" = None,
    window_days: int = 7,
    max_per_window: int = 250,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Paginate fetch_articles() by splitting the date range into chunks.

    Because GDELT caps results at 250/request, this is the recommended
    approach for longer time windows (e.g. a full election campaign).

    Parameters
    ----------
    window_days     : Size of each sub-window in days.
    max_per_window  : Records to fetch per sub-window.
    verbose         : Print progress.
    """
    if end is None:
        end = pd.Timestamp.now(tz="UTC")

    start_ts = pd.Timestamp(start, tz="UTC") if isinstance(start, str) else pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
    end_ts   = pd.Timestamp(end,   tz="UTC") if isinstance(end,   str) else pd.Timestamp(end).tz_localize("UTC")   if pd.Timestamp(end).tzinfo   is None else pd.Timestamp(end)

    all_frames = []
    cursor = start_ts

    while cursor < end_ts:
        window_end = min(cursor + timedelta(days=window_days), end_ts)
        if verbose:
            print(f"  Fetching {cursor.date()} → {window_end.date()} …", end=" ")

        try:
            chunk = fetch_articles(query, start=cursor, end=window_end, max_records=max_per_window)
            if verbose:
                print(f"{len(chunk)} articles")
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


# ── tone aggregation (daily) ──────────────────────────────────────────────────

def aggregate_daily_tone(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level tone scores to daily statistics.

    Returns a DataFrame with columns:
        date, n_articles, mean_tone, median_tone, tone_std,
        mean_positive, mean_negative, mean_polarity
    """
    if articles_df.empty:
        return pd.DataFrame()

    df = articles_df.copy()
    df["day"] = df["date"].dt.normalize()

    agg = df.groupby("day").agg(
        n_articles    = ("url",            "count"),
        mean_tone     = ("tone",           "mean"),
        median_tone   = ("tone",           "median"),
        tone_std      = ("tone",           "std"),
        mean_positive = ("positive_score", "mean"),
        mean_negative = ("negative_score", "mean"),
        mean_polarity = ("polarity",       "mean"),
    ).reset_index()

    agg = agg.rename(columns={"day": "date"})
    agg["date"] = pd.to_datetime(agg["date"], utc=True)
    return agg.sort_values("date").reset_index(drop=True)


# ── query builders ────────────────────────────────────────────────────────────

def build_prediction_market_query(
    topic_terms: list[str],
    include_polymarket: bool = True,
) -> str:
    """
    Build a GDELT query string for prediction market coverage of a topic.

    The query targets articles that mention BOTH prediction market
    language AND the topic — this gets closer to measuring market-influenced
    coverage rather than all coverage of the topic.

    Keeps the query short to avoid GDELT URL-length / parsing limits.
    GDELT supports: AND (implicit), OR, NOT, quoted phrases, parentheses.

    Parameters
    ----------
    topic_terms         : List of topic keywords (e.g. ["Trump", "Biden", "election"])
    include_polymarket  : Whether to include "polymarket" as an explicit term.

    Example
    -------
    >>> q = build_prediction_market_query(["Trump", "Harris", "election"])
    >>> # → '"polymarket" OR "prediction market" (Trump OR Harris OR election)'
    """
    # Keep market clause concise; long OR chains break GDELT's parser
    if include_polymarket:
        market_clause = '"polymarket" OR "prediction market"'
    else:
        market_clause = '"prediction market" OR "betting odds"'

    # Cap to first 4 topic terms to avoid URL length issues
    capped_terms = topic_terms[:4]
    topic_clause = " OR ".join(capped_terms)

    return f"({market_clause}) ({topic_clause})"

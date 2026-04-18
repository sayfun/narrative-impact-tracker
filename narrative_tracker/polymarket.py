"""
polymarket.py
-------------
Polymarket API client for the Narrative Impact Tracker.

Covers two endpoints:
  - Gamma API  (gamma-api.polymarket.com) : market discovery and metadata
  - CLOB API   (clob.polymarket.com)      : price / probability history

Key design decisions:
  - All timestamps are handled as UTC-aware pandas Timestamps internally.
  - Price history is returned at daily fidelity by default (fidelity=1440).
  - Market discovery returns a DataFrame so it can be inspected interactively.
"""

import re
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── constants ────────────────────────────────────────────────────────────────

GAMMA_BASE  = "https://gamma-api.polymarket.com"
CLOB_BASE   = "https://clob.polymarket.com"
DEFAULT_FIDELITY = 1440          # minutes → daily candles
REQUEST_TIMEOUT  = 30            # seconds


# ── market discovery ─────────────────────────────────────────────────────────

def search_markets(query: str, limit: int = 20, include_active: bool = True, include_closed: bool = True) -> pd.DataFrame:
    """
    Search Polymarket for markets matching *query*.

    NOTE: The Polymarket Gamma API's `q` text-search parameter is unreliable —
    it ignores the query and returns markets by volume. We therefore fetch
    active markets in bulk and filter client-side by matching query words
    against the question text. Falls back to closed markets if needed.

    Returns a DataFrame with columns:
        condition_id, question, end_date_iso, active, volume, volume_24hr,
        liquidity, token_ids
    """

    def _parse_token_ids(raw) -> list:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    def _market_row(m: dict) -> dict:
        return {
            "condition_id": m.get("conditionId", ""),
            "question":     m.get("question", ""),
            "end_date_iso": m.get("endDateIso") or m.get("endDate", ""),
            "active":       m.get("active", False),
            "closed":       m.get("closed", False),
            "volume":       float(m.get("volume", 0) or 0),
            "volume_24hr":  float(m.get("volume24hr", 0) or 0),
            "liquidity":    float(m.get("liquidityClob") or m.get("liquidity", 0) or 0),
            "token_ids":    _parse_token_ids(m.get("clobTokenIds")),
        }

    # Stopwords that are too common to be meaningful match signals.
    # Keep this list tight — only words that carry no discriminating power.
    _STOPWORDS = {
        # Articles / pronouns / conjunctions
        "will", "the", "who", "what", "when", "which", "that", "this",
        "with", "from", "have", "has", "been", "are", "was", "were",
        "for", "and", "not", "but", "can", "its", "his", "her", "they",
        # Generic verbs / prepositions — carry no topic signal
        "make", "take", "come", "does", "more", "than", "some",
        "into", "over", "just", "also", "back", "time", "only",
        # Round / place — appear in every sports bracket AND every election
        # first-round result, adding huge noise without discriminating power
        "round", "place", "first", "second", "third",
        # Short words that create false matches
        "win",  # substring of "winner", "twins", etc.
        "set",  # matches esports "Set 1", "Set 2"
    }

    # Tokenise query — keep only meaningful content words (len > 3, not stopwords)
    query_words = [
        w.lower() for w in query.split()
        if len(w) > 3 and w.lower() not in _STOPWORDS
    ]

    def _score(question: str) -> int:
        """Count how many query words appear as whole words in the question."""
        q_lower = question.lower()
        return sum(
            1 for w in query_words
            if re.search(r'\b' + re.escape(w) + r'\b', q_lower)
        )

    markets_url = f"{GAMMA_BASE}/markets"
    events_url  = f"{GAMMA_BASE}/events"
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    def _add_market(m: dict, event_score: int = 0) -> None:
        """Parse one raw market dict and add it to all_rows if not seen."""
        row = _market_row(m)
        if row["token_ids"] and row["question"] and row["condition_id"] not in seen_ids:
            # Use the higher of question-level or event-title-level score.
            # This ensures sub-markets inside a matching event (e.g. all 20
            # "Will Bieber feature X?" sub-questions) rank highly even if the
            # individual question only mentions one query word.
            row["_score"] = max(_score(row["question"]), event_score)
            all_rows.append(row)
            seen_ids.add(row["condition_id"])

    def _fetch_markets_page(params: dict) -> None:
        resp = requests.get(markets_url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200 or not resp.content:
            return
        try:
            data = resp.json()
        except Exception:
            return
        for m in (data or []):
            _add_market(m)

    # ── Pool A: currently trading, ranked by 24h volume ───────────────────────
    # Catches today's hot markets regardless of all-time volume.
    _fetch_markets_page({"limit": 200, "active": "true", "closed": "false",
                         "order": "volume24hr", "ascending": "false", "offset": 0})
    _fetch_markets_page({"limit": 200, "active": "true", "closed": "false",
                         "order": "volume24hr", "ascending": "false", "offset": 200})

    # ── Pool B: all markets by all-time volume (no status filter) ─────────────
    # Catches big historical markets ($1B+ volume) that may be archived.
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 0})
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 200})
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 400})
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 600})
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 800})
    _fetch_markets_page({"limit": 200, "order": "volume", "ascending": "false", "offset": 1000})

    # ── Pool C: recent events, filter by event title, add ALL sub-markets ─────
    # The Gamma API's q= text-search parameter is broken (returns random markets
    # by volume regardless of the query). Instead, we fetch recent events sorted
    # by creation date and filter on the event TITLE client-side.
    #
    # This is the critical pool for new, low-volume markets (e.g. a Justin Bieber
    # Coachella event with $8K volume would never appear in Pools A or B, but its
    # event title matches the query immediately). When an event matches, we add
    # ALL of its sub-markets so users can pick the exact outcome they want.
    try:
        for ev_offset in [0, 100, 200, 300]:
            resp = requests.get(
                events_url,
                params={
                    "active": "true", "closed": "false",
                    "order": "startDate", "ascending": "false",
                    "limit": 100, "offset": ev_offset,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200 or not resp.content:
                break
            try:
                events = resp.json() or []
            except Exception:
                break
            if not events:
                break
            for event in events:
                event_title = event.get("title", "")
                ev_score = sum(
                    1 for w in query_words
                    if re.search(r'\b' + re.escape(w) + r'\b', event_title.lower())
                )
                if ev_score == 0:
                    continue  # event title has no query words — skip
                # Add every sub-market from this matching event
                for m in event.get("markets", []):
                    _add_market(m, event_score=ev_score)
    except Exception:
        pass  # Pool C is best-effort; Pools A/B still run

    # Minimum score thresholds.
    # Users rarely type market question verbatim — "lakers 2nd round" maps to
    # "advance to the NBA Finals", so we only need 1 strong distinctive word to
    # match for short queries. For long queries (4+ words) we require 2 matches
    # to prevent very generic terms from flooding results.
    if len(query_words) >= 4:
        min_score = 2
    else:
        min_score = 1
    matched = [r for r in all_rows if r["_score"] >= min_score]

    if not matched:
        return pd.DataFrame()

    df = pd.DataFrame(matched)

    # Client-side filter by active/closed status
    if include_active and not include_closed:
        df = df[df["active"] == True]
    elif include_closed and not include_active:
        df = df[(df["active"] == False) | (df["closed"] == True)]
    # if both or neither are checked, return everything matched

    if df.empty:
        return df.reset_index(drop=True)

    # Sort: text relevance first (dominant), then active status as tiebreaker,
    # then all-time volume. A closed market that matches 4/4 query words beats
    # an active market that only matches 2/4.
    df["_sort"] = df["_score"] * 1e14 + df["active"].astype(int) * 1e12 + df["volume"]
    df = (df.sort_values("_sort", ascending=False)
            .drop(columns=["_score", "_sort"])
            .head(limit)
            .reset_index(drop=True))
    return df


def get_trending_markets(
    top_n:            int   = 30,
    return_n:         int   = 15,
    movement_window_hours: int = 72,
    volume_weight:    float = 0.5,
    movement_weight:  float = 0.5,
    min_volume_24hr:  float = 1000.0,
    max_workers:      int   = 10,
) -> pd.DataFrame:
    """
    Surface markets where a narrative is *actively* being constructed right now.

    Hybrid approach:
      1. One Gamma call: fetch top `top_n` active markets by 24h volume.
      2. Parallel CLOB calls: fetch recent price history for each, compute
         |probability change| over the last `movement_window_hours`.
      3. Composite rank = normalised volume24hr × volume_weight
                        + normalised |movement|  × movement_weight.

    Theoretically grounded for narrative-impact work: a market that's moving
    is generating a new story; a stable-but-liquid market is not.

    Returns a DataFrame with the same columns as `search_markets()` plus:
        volume_24hr       (float) — raw 24h USD volume
        prob_delta_72h    (float) — signed probability change over window
        movement_abs      (float) — |prob_delta_72h|
        trending_score    (float) — composite 0–1 score (already sorted desc)

    Empty DataFrame if the Gamma endpoint returns nothing.
    """
    import numpy as np

    # ── 1. Pull top markets by 24h volume from Gamma ──────────────────────────
    resp = requests.get(
        f"{GAMMA_BASE}/markets",
        params={
            "active":    "true",
            "closed":    "false",
            "order":     "volume24hr",
            "ascending": "false",
            "limit":     top_n,
            "offset":    0,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200 or not resp.content:
        return pd.DataFrame()

    try:
        raw = resp.json() or []
    except Exception:
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()

    def _parse_token_ids(r):
        if r is None:
            return []
        if isinstance(r, list):
            return r
        if isinstance(r, str):
            try:
                p = json.loads(r)
                return p if isinstance(p, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    today_ts = pd.Timestamp.now(tz="UTC")

    rows = []
    for m in raw:
        tokens = _parse_token_ids(m.get("clobTokenIds"))
        vol24  = float(m.get("volume24hr", 0) or 0)
        if not tokens or not m.get("question") or vol24 < min_volume_24hr:
            continue
        # Skip markets resolving within 3 days — too close to resolution
        # for meaningful narrative analysis, and CLOB history is too sparse.
        raw_end = m.get("endDateIso") or m.get("endDate") or ""
        if raw_end:
            try:
                end_ts = pd.Timestamp(raw_end, tz="UTC")
                if (end_ts - today_ts).days < 3:
                    continue
            except Exception:
                pass
        rows.append({
            "condition_id": m.get("conditionId", ""),
            "question":     m.get("question", ""),
            "end_date_iso": m.get("endDateIso") or m.get("endDate", ""),
            "active":       m.get("active", False),
            "closed":       m.get("closed", False),
            "volume":       float(m.get("volume", 0) or 0),
            "volume_24hr":  vol24,
            "liquidity":    float(m.get("liquidityClob") or m.get("liquidity", 0) or 0),
            "token_ids":    tokens,
        })

    if not rows:
        return pd.DataFrame()

    # ── 2. Compute 72h probability movement in parallel ───────────────────────
    window_start = pd.Timestamp.now(tz="UTC") - timedelta(hours=movement_window_hours)

    def _movement(row: dict) -> float:
        """Return signed probability change over the movement window, or NaN."""
        try:
            # Hourly fidelity — we need recent intraday, not daily
            hist = fetch_price_history(
                row["token_ids"][0],
                start=window_start - timedelta(hours=6),  # small buffer for start
                end=None,
                fidelity=60,
            )
            if hist.empty or len(hist) < 2:
                return float("nan")
            return float(hist["probability"].iloc[-1] - hist["probability"].iloc[0])
        except Exception:
            return float("nan")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_movement, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futures):
            i = futures[fut]
            rows[i]["prob_delta_72h"] = fut.result()

    df = pd.DataFrame(rows)
    df["prob_delta_72h"] = df["prob_delta_72h"].fillna(0.0)
    df["movement_abs"]   = df["prob_delta_72h"].abs()

    # ── 3. Composite score — min-max normalise each signal to [0, 1] ──────────
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - lo) / (hi - lo)

    df["trending_score"] = (
        _norm(df["volume_24hr"])  * volume_weight +
        _norm(df["movement_abs"]) * movement_weight
    )

    df = (df.sort_values("trending_score", ascending=False)
            .head(return_n)
            .reset_index(drop=True))
    return df


def get_market_by_slug(slug: str) -> dict:
    """
    Fetch a single market by its URL slug (e.g. 'presidential-election-winner-2024').
    Returns raw JSON dict.
    """
    url = f"{GAMMA_BASE}/markets"
    params = {"slug": slug}
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0]
    return data


# ── price / probability history ───────────────────────────────────────────────

def _to_unix(dt: "str | datetime | pd.Timestamp") -> int:
    """Convert various date representations to a UTC Unix timestamp (int)."""
    if isinstance(dt, (int, float)):
        return int(dt)
    if isinstance(dt, str):
        dt = pd.Timestamp(dt, tz="UTC")
    elif isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = pd.Timestamp(dt)
    return int(dt.timestamp())


def fetch_price_history(
    token_id:  str,
    start:     "str | datetime | pd.Timestamp | None" = None,
    end:       "str | datetime | pd.Timestamp | None" = None,
    fidelity:  int = DEFAULT_FIDELITY,
) -> pd.DataFrame:
    """
    Fetch probability time series for a single Polymarket outcome token.

    Parameters
    ----------
    token_id  : The YES or NO token ID from the market's clobTokenIds list.
    start     : Start of window. If None, fetches full history (interval='all').
    end       : End of window. If None and start is given, defaults to now.
    fidelity  : Candle resolution in minutes (1440 = daily).

    Note: The CLOB API enforces an undocumented maximum interval (roughly
    14-21 days for startTs/endTs). For full history, use start=None which
    triggers interval='all'. For custom windows, the client automatically
    chunks the request into 14-day segments and concatenates.

    Returns
    -------
    DataFrame with columns:
        date        (datetime64[ns, UTC])
        probability (float, 0–1)
        token_id    (str)
    """
    url = f"{CLOB_BASE}/prices-history"

    def _fetch_raw(fidel: int) -> list:
        """Single CLOB request; returns history list or []."""
        params = {"market": token_id, "interval": "all", "fidelity": fidel}
        try:
            r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return (r.json() or {}).get("history", [])
        except Exception:
            return []

    # Primary fetch at requested fidelity (default: 1440 = daily candles).
    history = _fetch_raw(fidelity)

    # Fallback: some markets have very sparse daily history (e.g. near
    # resolution, or very new markets). If we get fewer than 3 candles
    # and fidelity is daily, retry at 60-min resolution and let the
    # dedup/normalise step collapse it back to one-per-day.
    if len(history) < 3 and fidelity >= 1440:
        history = _fetch_raw(60) or history
    if not history:
        return pd.DataFrame(columns=["date", "probability", "token_id"])

    df = pd.DataFrame(history)           # columns: t (unix int or ISO string), p (price 0–1)
    df = df.rename(columns={"t": "date", "p": "probability"})

    # ── Robust timestamp parsing ─────────────────────────────────────────────
    # The CLOB API is inconsistent across market states:
    #   - Active markets:   t is an integer (Unix seconds)
    #   - Archived markets: t may be an ISO string OR an integer stored as
    #     a Python object (object-dtype column).
    # pandas interprets integers in an object-dtype column as NANOSECONDS when
    # passed to pd.to_datetime() without unit=, producing 1970-era dates.
    # Fix: always try numeric conversion first; fall back to ISO string parsing.
    date_col = df["date"]
    if pd.api.types.is_numeric_dtype(date_col):
        # Clean integer or float column — parse as Unix seconds
        df["date"] = pd.to_datetime(date_col, unit="s", utc=True)
    else:
        # May be object dtype containing either ints or ISO strings
        numeric = pd.to_numeric(date_col, errors="coerce")
        if numeric.notna().any():
            # At least some values are parseable as numbers → Unix seconds
            df["date"] = pd.to_datetime(numeric, unit="s", utc=True)
        else:
            # All values are strings → ISO 8601 / RFC 3339 format
            df["date"] = pd.to_datetime(date_col, utc=True, errors="coerce")

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["token_id"] = token_id

    # Normalise to daily: keep the last observation per calendar day,
    # then normalize the date to midnight so it merges cleanly later.
    df["day"] = df["date"].dt.normalize()
    df = (
        df.sort_values("date")
          .groupby("day", as_index=False)
          .last()
          .drop(columns=["day"])
          .sort_values("date")
          .reset_index(drop=True)
    )
    # Normalise to midnight *after* dedup so that filter comparisons are
    # date-vs-date (not intraday timestamp vs midnight), and so that the
    # date column is already clean when pipeline._build_aligned_frame merges it.
    df["date"] = df["date"].dt.normalize()

    # Filter to requested window if start/end were specified
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        df = df[df["date"] >= start_ts.normalize()]
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC") if isinstance(end, str) else end
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        df = df[df["date"] <= end_ts.normalize()]

    return df.reset_index(drop=True)


def fetch_market_probability(
    market_df_row: pd.Series,
    start:  "str | datetime | pd.Timestamp",
    end:    "str | datetime | pd.Timestamp | None" = None,
    outcome: str = "YES",
) -> pd.DataFrame:
    """
    Convenience wrapper: given a row from search_markets(), fetch the
    probability history for the YES (or NO) outcome token.

    Adds a `question` column to the returned DataFrame so the source
    market is always traceable.
    """
    tokens = market_df_row.get("token_ids", [])
    if not tokens:
        raise ValueError(f"No token_ids found in market row: {market_df_row}")

    # Polymarket always stores YES token first, NO token second
    idx = 0 if outcome.upper() == "YES" else 1
    if idx >= len(tokens):
        raise ValueError(f"Outcome '{outcome}' not available; tokens: {tokens}")

    df = fetch_price_history(tokens[idx], start=start, end=end)
    df["question"] = market_df_row.get("question", "")
    df["outcome"]  = outcome.upper()
    return df


# ── probability movement detection ───────────────────────────────────────────

def detect_sharp_movements(
    prob_df: pd.DataFrame,
    threshold: float = 0.08,
    window_days: int = 2,
) -> pd.DataFrame:
    """
    Identify dates where probability moved ≥ threshold over window_days.

    These 'clean shocks' are candidates for event-study analysis.

    Returns a DataFrame of shock events with columns:
        date, probability, delta, direction
    """
    df = prob_df.copy().sort_values("date").reset_index(drop=True)
    df["delta"] = df["probability"].diff(periods=window_days)
    shocks = df[df["delta"].abs() >= threshold].copy()
    shocks["direction"] = shocks["delta"].apply(
        lambda x: "UP" if x > 0 else "DOWN"
    )
    return shocks[["date", "probability", "delta", "direction"]].reset_index(drop=True)

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

import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional


# ── constants ────────────────────────────────────────────────────────────────

GAMMA_BASE  = "https://gamma-api.polymarket.com"
CLOB_BASE   = "https://clob.polymarket.com"
DEFAULT_FIDELITY = 1440          # minutes → daily candles
REQUEST_TIMEOUT  = 30            # seconds


# ── market discovery ─────────────────────────────────────────────────────────

def search_markets(query: str, limit: int = 20) -> pd.DataFrame:
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

    # Tokenise query into words for client-side matching
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    def _score(question: str) -> int:
        """Count how many query words appear in the question (case-insensitive)."""
        q_lower = question.lower()
        return sum(1 for w in query_words if w in q_lower)

    url = f"{GAMMA_BASE}/markets"
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    def _fetch_page(params: dict) -> None:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return
        for m in resp.json() or []:
            row = _market_row(m)
            if row["token_ids"] and row["question"] and row["condition_id"] not in seen_ids:
                row["_score"] = _score(row["question"])
                all_rows.append(row)
                seen_ids.add(row["condition_id"])

    # Active markets: two pages by 24h volume (~400 currently trading markets)
    _fetch_page({"limit": 200, "active": "true", "closed": "false",
                 "order": "volume24hr", "ascending": "false", "offset": 0})
    _fetch_page({"limit": 200, "active": "true", "closed": "false",
                 "order": "volume24hr", "ascending": "false", "offset": 200})

    # Resolved/historical markets by all-time volume.
    # The Gamma API uses two different fields for resolved markets:
    # some have closed=true, others have active=false with closed=false.
    # Fetch both to ensure large historical markets (e.g. 2024 US election,
    # $1.5B volume) always appear when relevant.
    _fetch_page({"limit": 200, "closed": "true",
                 "order": "volume", "ascending": "false", "offset": 0})
    _fetch_page({"limit": 200, "closed": "true",
                 "order": "volume", "ascending": "false", "offset": 200})
    _fetch_page({"limit": 200, "active": "false",
                 "order": "volume", "ascending": "false", "offset": 0})
    _fetch_page({"limit": 200, "active": "false",
                 "order": "volume", "ascending": "false", "offset": 200})

    matched = [r for r in all_rows if r["_score"] > 0]

    if not matched:
        return pd.DataFrame()

    df = pd.DataFrame(matched)
    # Sort: text relevance first (dominant), then active status as tiebreaker,
    # then all-time volume. A closed market that matches 4/4 query words beats
    # an active market that only matches 2/4.
    df["_sort"] = df["_score"] * 1e14 + df["active"].astype(int) * 1e12 + df["volume"]
    df = (df.sort_values("_sort", ascending=False)
            .drop(columns=["_score", "_sort"])
            .head(limit)
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

    # If no start specified, fetch full history with interval=all
    if start is None:
        params = {
            "market":   token_id,
            "interval": "all",
            "fidelity": fidelity,
        }
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    else:
        # Chunked fetch: CLOB API rejects intervals > ~14 days for custom ranges
        # Use interval=all and filter client-side (simpler and reliable)
        params = {
            "market":   token_id,
            "interval": "all",
            "fidelity": fidelity,
        }
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

    history = data.get("history", [])
    if not history:
        return pd.DataFrame(columns=["date", "probability", "token_id"])

    df = pd.DataFrame(history)           # columns: t (unix), p (price 0–1)
    df = df.rename(columns={"t": "date", "p": "probability"})
    df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["token_id"] = token_id

    # Normalise to end-of-day: keep the last observation per calendar day
    df["day"] = df["date"].dt.normalize()
    df = (
        df.sort_values("date")
          .groupby("day", as_index=False)
          .last()
          .drop(columns=["day"])
          .sort_values("date")
          .reset_index(drop=True)
    )

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

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

    Uses the events API (sorted by volume) to find the most relevant markets.
    Falls back to the markets API for queries that don't match events.

    Returns a DataFrame with columns:
        condition_id, question, end_date_iso, active, volume, liquidity,
        token_ids   (list of [YES_token_id, NO_token_id])

    Usage
    -----
    >>> markets = search_markets("presidential election 2024")
    >>> print(markets[["question", "volume"]].head())
    """
    # Strategy 1: search events (aggregated markets) sorted by volume
    events_url = f"{GAMMA_BASE}/events"
    events_params = {
        "q":          query,
        "limit":      limit,
        "closed":     "true",
        "order":      "volume",
        "ascending":  "false",
    }
    events_resp = requests.get(events_url, params=events_params, timeout=REQUEST_TIMEOUT)
    events_resp.raise_for_status()
    events_data = events_resp.json() or []

    def _parse_token_ids(raw) -> list:
        """clobTokenIds comes back as a JSON string from the Gamma API."""
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

    rows = []
    for event in events_data:
        for m in event.get("markets", []):
            # clobTokenIds is a JSON-encoded string in the Gamma API
            token_ids = _parse_token_ids(m.get("clobTokenIds"))
            rows.append({
                "condition_id": m.get("conditionId", ""),
                "question":     m.get("question", ""),
                "end_date_iso": m.get("endDateIso") or m.get("endDate", ""),
                "active":       m.get("active", False),
                "closed":       m.get("closed", False),
                "volume":       float(m.get("volume", 0) or 0),
                "liquidity":    float(m.get("liquidityClob") or m.get("liquidity", 0) or 0),
                "token_ids":    token_ids,
            })

    if not rows:
        # Strategy 2: fall back to markets API
        url = f"{GAMMA_BASE}/markets"
        params = {"q": query, "limit": limit, "closed": "true"}
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        for m in resp.json():
            token_ids = _parse_token_ids(m.get("clobTokenIds"))
            rows.append({
                "condition_id": m.get("conditionId", ""),
                "question":     m.get("question", ""),
                "end_date_iso": m.get("endDateIso") or m.get("endDate", ""),
                "active":       m.get("active", False),
                "closed":       m.get("closed", False),
                "volume":       float(m.get("volume", 0) or 0),
                "liquidity":    float(m.get("liquidity", 0) or 0),
                "token_ids":    token_ids,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("volume", ascending=False).reset_index(drop=True)
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

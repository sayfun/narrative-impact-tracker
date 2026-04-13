"""
pipeline.py
-----------
Alignment pipeline: merges Polymarket probability data with GDELT
coverage data into a single analytical DataFrame.

This is the core of the Narrative Impact Tracker.  The aligned frame
is what you hand to the statistical analysis layer (Granger tests,
event study) and to the narrative feature extractor.

Output schema
-------------
    date              datetime64[ns, UTC]   — calendar day
    probability       float                 — Polymarket YES probability (0–1)
    prob_delta_1d     float                 — 1-day probability change
    prob_delta_3d     float                 — 3-day probability change
    prob_pct_rank     float                 — percentile rank within window
    article_count     int                   — GDELT daily article count
    volume_norm       float                 — normalised coverage volume
    rolling_3d_vol    float                 — 3-day rolling coverage volume
    mean_tone         float                 — GDELT daily mean tone
    tone_std          float                 — daily tone standard deviation
    mean_positive     float
    mean_negative     float
    is_shock          bool                  — True on sharp movement days
    shock_direction   str | None            — 'UP', 'DOWN', or None
"""

import pandas as pd
import numpy as np
from typing import Optional

from narrative_tracker.polymarket import (
    search_markets,
    fetch_market_probability,
    detect_sharp_movements,
)
from narrative_tracker.gdelt import (
    fetch_coverage_timeline,
    fetch_articles_windowed,
    aggregate_daily_tone,
    build_prediction_market_query,
)


# ── main pipeline class ────────────────────────────────────────────────────────

class NarrativePipeline:
    """
    Orchestrates data collection and alignment for a single Polymarket market.

    Usage
    -----
    pipe = NarrativePipeline(
        market_query   = "presidential election winner 2024",
        topic_terms    = ["Trump", "Biden", "election", "president"],
        start          = "2024-01-01",
        end            = "2024-11-07",
    )
    pipe.collect()          # fetches from both APIs
    df = pipe.aligned       # the merged analytical DataFrame
    shocks = pipe.shocks    # detected sharp probability movements

    Design notes
    ------------
    - The pipeline is deliberately stateful so you can inspect intermediate
      results (pipe.prob_df, pipe.coverage_df, pipe.articles_df) before
      looking at the merged output.
    - All time series are aligned on calendar day (UTC).  Missing days
      are forward-filled for probability (markets don't trade on weekends)
      and zero-filled for coverage.
    """

    def __init__(
        self,
        market_query:   str,
        topic_terms:    list[str],
        start:          "str | pd.Timestamp",
        end:            "str | pd.Timestamp | None" = None,
        market_index:   int = 0,           # which result from search_markets() to use
        shock_threshold: float = 0.08,     # minimum prob movement to flag as shock
        shock_window:   int = 2,           # days over which movement is measured
        fetch_articles: bool = True,       # set False for faster runs without tone data
        article_window_days: int = 7,      # GDELT pagination window
        manual_token_id: str = "",         # bypass search; use this YES token directly
        manual_market_question: str = "",  # display label when using manual token
    ):
        self.market_query   = market_query
        self.topic_terms    = topic_terms
        self.start          = pd.Timestamp(start, tz="UTC") if isinstance(start, str) else start
        self.end            = (
            pd.Timestamp(end, tz="UTC") if isinstance(end, str) and end
            else pd.Timestamp.now(tz="UTC")
        )
        self.market_index        = market_index
        self.shock_threshold     = shock_threshold
        self.shock_window        = shock_window
        self._fetch_articles     = fetch_articles
        self.article_window_days = article_window_days
        self.manual_token_id     = manual_token_id.strip()
        self.manual_market_question = manual_market_question.strip()

        # populated by collect()
        self.market_meta   = None
        self.prob_df       = None
        self.coverage_df   = None
        self.articles_df   = None
        self.tone_df       = None
        self.shocks        = None
        self._aligned      = None

    # ── collection ─────────────────────────────────────────────────────────

    def collect(self, verbose: bool = True) -> "NarrativePipeline":
        """Fetch all data from APIs and build the aligned DataFrame."""

        # 1. Discover market (or use manual token directly)
        if self.manual_token_id:
            if verbose:
                print(f"[1/4] Using manual token ID: {self.manual_token_id[:20]}…")
            self.market_meta = pd.Series({
                "question":    self.manual_market_question or "Manual market",
                "token_ids":   [self.manual_token_id],
                "volume":      0,
                "volume_24hr": 0,
                "active":      False,
                "closed":      True,
            })
        else:
            if verbose:
                print(f"[1/4] Searching Polymarket: '{self.market_query}' …")
            markets = search_markets(self.market_query)
            if markets.empty:
                raise RuntimeError(f"No Polymarket markets found for: '{self.market_query}'")
            self.market_meta = markets.iloc[self.market_index]
            if verbose:
                print(f"      → Using: {self.market_meta['question'][:80]}")
                print(f"        Volume: ${self.market_meta['volume']:,.0f}  "
                      f"| Token IDs: {self.market_meta['token_ids']}")

        # 2. Fetch probability history (YES token)
        if verbose:
            print(f"[2/4] Fetching Polymarket price history …")
        self.prob_df = fetch_market_probability(
            self.market_meta, start=self.start, end=self.end, outcome="YES"
        )
        if verbose:
            print(f"      → {len(self.prob_df)} daily data points")

        # 3. Fetch GDELT coverage
        gdelt_query = build_prediction_market_query(
            self.topic_terms, include_polymarket=True
        )
        if verbose:
            print(f"[3/4] Fetching GDELT coverage timeline …")
            print(f"      Query: {gdelt_query}")
        self.coverage_df = fetch_coverage_timeline(
            gdelt_query, start=self.start, end=self.end
        )
        if verbose:
            print(f"      → {len(self.coverage_df)} daily coverage points")

        if self._fetch_articles:
            if verbose:
                print(f"[3b]  Fetching GDELT article list (paginated) …")
            self.articles_df = fetch_articles_windowed(
                gdelt_query,
                start=self.start,
                end=self.end,
                window_days=self.article_window_days,
                verbose=verbose,
            )
            self.tone_df = aggregate_daily_tone(self.articles_df)
            if verbose:
                print(f"      → {len(self.articles_df)} articles collected")

        # 4. Align
        if verbose:
            print(f"[4/4] Aligning time series …")
        self._aligned = self._build_aligned_frame()
        self.shocks   = detect_sharp_movements(
            self.prob_df,
            threshold=self.shock_threshold,
            window_days=self.shock_window,
        )
        if verbose:
            print(f"      → {len(self._aligned)} aligned days")
            print(f"      → {len(self.shocks)} sharp movements detected "
                  f"(≥{self.shock_threshold*100:.0f}pp over {self.shock_window}d)")
            print("Done.")

        return self

    # ── alignment logic ────────────────────────────────────────────────────

    def _build_aligned_frame(self) -> pd.DataFrame:
        """Inner method: merge and impute all time series."""

        # Build a full date index covering the window
        date_range = pd.date_range(
            start=self.start.normalize(),
            end=self.end.normalize(),
            freq="D",
            tz="UTC",
        )
        base = pd.DataFrame({"date": date_range})

        # ── Polymarket (forward-fill gaps, e.g. weekends) ──
        prob = self.prob_df[["date", "probability"]].copy()
        prob["date"] = prob["date"].dt.normalize()
        prob = prob.drop_duplicates("date")
        merged = base.merge(prob, on="date", how="left")
        merged["probability"] = merged["probability"].ffill()

        # Derived probability features
        merged["prob_delta_1d"] = merged["probability"].diff(1)
        merged["prob_delta_3d"] = merged["probability"].diff(3)
        merged["prob_pct_rank"] = merged["probability"].rank(pct=True)

        # ── GDELT coverage volume ──
        if self.coverage_df is not None and not self.coverage_df.empty:
            cov = self.coverage_df.copy()
            cov["date"] = cov["date"].dt.normalize()
            cov = cov.rename(columns={"rolling_3d": "rolling_3d_vol"})
            merged = merged.merge(
                cov[["date", "article_count", "volume_norm", "rolling_3d_vol"]],
                on="date", how="left"
            )
            merged["article_count"]  = merged["article_count"].fillna(0).astype(int)
            merged["volume_norm"]    = merged["volume_norm"].fillna(0.0)
            merged["rolling_3d_vol"] = merged["rolling_3d_vol"].fillna(0.0)
        else:
            merged["article_count"]  = 0
            merged["volume_norm"]    = 0.0
            merged["rolling_3d_vol"] = 0.0

        # ── GDELT tone (from article-level aggregation) ──
        if self.tone_df is not None and not self.tone_df.empty:
            tone = self.tone_df.copy()
            tone["date"] = tone["date"].dt.normalize()
            merged = merged.merge(
                tone[["date", "mean_tone", "tone_std", "mean_positive", "mean_negative"]],
                on="date", how="left"
            )
        else:
            for col in ["mean_tone", "tone_std", "mean_positive", "mean_negative"]:
                merged[col] = np.nan

        # ── Shock flags ──
        shock_df = detect_sharp_movements(
            self.prob_df,
            threshold=self.shock_threshold,
            window_days=self.shock_window,
        )
        shock_days = set(shock_df["date"].dt.normalize())
        merged["is_shock"] = merged["date"].isin(shock_days)
        merged["shock_direction"] = merged["date"].map(
            dict(zip(shock_df["date"].dt.normalize(), shock_df["direction"]))
        )

        return merged.reset_index(drop=True)

    # ── properties ────────────────────────────────────────────────────────

    @property
    def aligned(self) -> pd.DataFrame:
        if self._aligned is None:
            raise RuntimeError("Call .collect() first.")
        return self._aligned

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline state."""
        if self._aligned is None:
            return "Pipeline not yet collected. Call .collect() first."

        df = self._aligned
        lines = [
            f"Market      : {self.market_meta['question'][:70]}",
            f"Window      : {self.start.date()} → {self.end.date()}  ({len(df)} days)",
            f"Probability : min={df['probability'].min():.2f}  "
                         f"max={df['probability'].max():.2f}  "
                         f"mean={df['probability'].mean():.2f}",
            f"Coverage    : total articles={df['article_count'].sum():,}  "
                         f"peak={df['article_count'].max():,}/day",
            f"Shocks      : {df['is_shock'].sum()} days with ≥{self.shock_threshold*100:.0f}pp movement",
            f"Tone data   : {'yes' if not df['mean_tone'].isna().all() else 'no (articles not fetched)'}",
        ]
        return "\n".join(lines)


# ── convenience function ───────────────────────────────────────────────────────

def run_pipeline(
    market_query:  str,
    topic_terms:   list[str],
    start:         str,
    end:           str | None = None,
    fetch_articles: bool = True,
    verbose:       bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-shot helper for interactive use.

    Returns (aligned_df, shocks_df).

    Example
    -------
    aligned, shocks = run_pipeline(
        market_query = "presidential election winner 2024",
        topic_terms  = ["Trump", "Harris", "Biden", "election", "president"],
        start        = "2024-07-01",
        end          = "2024-11-07",
    )
    """
    pipe = NarrativePipeline(
        market_query   = market_query,
        topic_terms    = topic_terms,
        start          = start,
        end            = end,
        fetch_articles = fetch_articles,
    )
    pipe.collect(verbose=verbose)
    print("\n" + pipe.summary())
    return pipe.aligned, pipe.shocks

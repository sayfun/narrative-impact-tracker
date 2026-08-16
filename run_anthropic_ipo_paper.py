"""
Run the Narrative Impact Tracker on the Anthropic IPO by October 31, 2026 market
for the Prototypes for Humanity short paper.

Uses:
  - Polymarket manual token ID (market has fallen out of search top-3)
  - MediaCloud API v4/v5 direct via the current `mediacloud` pip package
    (the tracker's built-in mediacloud_client.py targets a retired v2 endpoint)
  - GDELT as coverage volume signal

Outputs land in ~/Research Articles/Prototypes_Anthropic_Case/local_full_run/

Usage:
    python3 run_anthropic_ipo_paper.py

Requires (one-time):
    pip3 install mediacloud
"""
from pathlib import Path
import json
import os
import datetime as dt
import pandas as pd

from narrative_tracker.pipeline import NarrativePipeline
from narrative_tracker.analysis import run_full_analysis
from narrative_tracker.features import (
    extract_features_batch,
    aggregate_daily_features,
    compute_eai,
)

# ── config ────────────────────────────────────────────────────────────────────

YES_TOKEN = "35480485928147694422522317298160081194603766865102488154902775756915955129166"
MARKET_LABEL = "Will Anthropic IPO by October 31, 2026?"
TOPIC_TERMS = ["Anthropic", "IPO", "valuation"]
START = "2026-06-01"
END = "2026-07-27"

MC_API_KEY = os.getenv("MEDIACLOUD_API_KEY", "")
if not MC_API_KEY:
    raise RuntimeError("Set the MEDIACLOUD_API_KEY environment variable before running.")

# MediaCloud's US National Newspapers collection (broad English-language coverage)
US_NATIONAL_COLLECTION = 34412234

OUT = Path.home() / "Research Articles" / "Prototypes_Anthropic_Case" / "local_full_run"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Polymarket + GDELT ─────────────────────────────────────────────────────

print(f"Running Narrative Impact Tracker")
print(f"  Market : {MARKET_LABEL}")
print(f"  Window : {START} → {END}")
print(f"  Topics : {', '.join(TOPIC_TERMS)}")
print(f"  Output : {OUT}\n")

pipe = NarrativePipeline(
    market_query=MARKET_LABEL,
    manual_token_id=YES_TOKEN,
    topic_terms=TOPIC_TERMS,
    start=START,
    end=END,
)
pipe.collect(verbose=True)

aligned = pipe.aligned.copy()


# ── 2. MediaCloud v4/v5 direct ────────────────────────────────────────────────

print("\nFetching MediaCloud stories via current API (v4/v5)…")

try:
    import mediacloud.api as mc_api
except ImportError:
    print("The `mediacloud` package is not installed. Run: pip3 install mediacloud")
    mc_articles = None
else:
    # Build a Solr-style query. The old client wrapped topics with "(polymarket OR
    # prediction market) AND (...)" — that filter effectively excluded most
    # coverage. For this case we drop the market-clause and search directly
    # on the topic terms.
    q = " AND ".join(TOPIC_TERMS)   # "Anthropic AND IPO AND valuation"

    start_date = dt.date.fromisoformat(START)
    end_date = dt.date.fromisoformat(END)

    mc_search = mc_api.SearchApi(MC_API_KEY)
    all_stories = []
    pagination_token = None
    page_num = 0
    while True:
        page_num += 1
        try:
            page, pagination_token = mc_search.story_list(
                q,
                start_date=start_date,
                end_date=end_date,
                collection_ids=[US_NATIONAL_COLLECTION],
                pagination_token=pagination_token,
            )
        except Exception as e:
            print(f"  MediaCloud error on page {page_num}: {e}")
            break
        all_stories += page
        print(f"  page {page_num}: {len(page)} stories (running total {len(all_stories)})")
        if pagination_token is None or page_num >= 20:  # safety cap
            break

    if not all_stories:
        print("MediaCloud returned no articles.")
        mc_articles = None
    else:
        mc_articles = pd.DataFrame(all_stories)
        # Normalise column names for the feature extractor
        if "publish_date" in mc_articles.columns:
            mc_articles["date"] = pd.to_datetime(
                mc_articles["publish_date"], utc=True, errors="coerce"
            )
        elif "date" not in mc_articles.columns:
            mc_articles["date"] = pd.NaT
        for col in ("title", "url", "text"):
            if col not in mc_articles.columns:
                mc_articles[col] = ""
        # If text is missing (v4/v5 story_list returns metadata by default),
        # fall back to title for feature extraction.
        if mc_articles["text"].astype(str).str.strip().eq("").all():
            print("  Note: story_list didn't return full text — using titles only.")
            mc_articles["text"] = mc_articles["title"]

        mc_articles.to_csv(OUT / "articles_mediacloud.csv", index=False)
        print(f"  Saved articles_mediacloud.csv  ({len(mc_articles)} rows)")


# ── 3. Feature extraction (ERS / PCF / NCS / EAI) ────────────────────────────

if mc_articles is not None and not mc_articles.empty:
    print("\nExtracting ERS / PCF / NCS features…")
    enriched = extract_features_batch(
        mc_articles, text_col="text", title_col="title", verbose=True
    )

    daily = aggregate_daily_features(enriched)
    daily = compute_eai(daily)

    if daily is not None and not daily.empty:
        feat_cols = [c for c in daily.columns if c != "date"]
        daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
        aligned["date"] = pd.to_datetime(aligned["date"], utc=True).dt.normalize()
        aligned = aligned.merge(daily[["date"] + feat_cols], on="date", how="left")

    enriched.to_csv(OUT / "articles_features.csv", index=False)
    print(f"Saved: articles_features.csv")


# ── 4. Save aligned frame + shocks ────────────────────────────────────────────

aligned.to_csv(OUT / "aligned_frame.csv", index=False)
print(f"\nSaved: aligned_frame.csv ({len(aligned)} rows, {len(aligned.columns)} cols)")

if hasattr(pipe, "shocks") and pipe.shocks is not None and len(pipe.shocks):
    pipe.shocks.to_csv(OUT / "shocks.csv", index=False)
    print(f"Saved: shocks.csv ({len(pipe.shocks)} shocks)")


# ── 5. Statistical analysis on enriched frame ────────────────────────────────

print("\nRunning statistical analysis on enriched frame…")
analysis = run_full_analysis(aligned, pipe.shocks)

def _safe(o):
    if hasattr(o, "to_dict"): return o.to_dict()
    if hasattr(o, "tolist"): return o.tolist()
    if isinstance(o, dict): return {k: _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_safe(x) for x in o]
    if isinstance(o, (str, int, float, bool)) or o is None: return o
    return str(o)

with open(OUT / "analysis_summary.json", "w") as f:
    json.dump(_safe(analysis), f, indent=2, default=str)
print("Saved: analysis_summary.json")


# ── 6. Print headline metrics ─────────────────────────────────────────────────

print()
print("=" * 60)
print(pipe.summary())
print("=" * 60)

xc = analysis.get("xcorr", {})
gr = analysis.get("granger", {})
print(f"\nPeak cross-correlation r = {xc.get('peak_corr', 0):.3f} at lag={xc.get('peak_lag', 0)}d")
gi = gr.get('interpretation', '')
if gi:
    print(f"Granger: {gi.splitlines()[0]}")

print("\nDone.")

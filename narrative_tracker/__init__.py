"""
narrative-impact-tracker
========================
A research tool for empirically measuring the influence of prediction
market probability movements on media narratives.

Theoretical grounding
---------------------
Implements the three-mode enrollment framework (financial, discursive,
epistemic) and operationalises three core narrative variables:

  ERS  — Epistemic Register Score: certainty vs hedging language
  PCF  — Probability Citation Frequency: explicit market uptake
  NCS  — Narrative Closure Score: possibility-space foreclosure language

These map onto the secondary narrativisation process by which bare
probability numbers are transformed into character-driven stories.

Quick start
-----------
From the command line:
    narrative-tracker run --market "presidential election winner 2024" \\
                          --topic Trump Harris election \\
                          --start 2024-07-15 --end 2024-11-07

From Python:
    from narrative_tracker.pipeline import NarrativePipeline
    pipe = NarrativePipeline(
        market_query="presidential election winner 2024",
        topic_terms=["Trump", "Harris", "election"],
        start="2024-07-15",
        end="2024-11-07",
    )
    pipe.collect()
    print(pipe.summary())
"""

__version__ = "0.1.0"
__author__  = "Narrative Impact Tracker"

from narrative_tracker.pipeline import NarrativePipeline
from narrative_tracker.polymarket import search_markets, fetch_price_history
from narrative_tracker.gdelt import fetch_coverage_timeline, fetch_articles

__all__ = [
    "NarrativePipeline",
    "search_markets",
    "fetch_price_history",
    "fetch_coverage_timeline",
    "fetch_articles",
]

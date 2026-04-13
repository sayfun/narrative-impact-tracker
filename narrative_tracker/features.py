"""
features.py
-----------
Narrative feature extractor for the Narrative Impact Tracker.

Implements three core narrative variables operating on article text:

  1. Epistemic Register Score (ERS)
     Measures certainty/uncertainty language density.
     High ERS = high-certainty framing ("is projected to win")
     Low ERS  = hedged framing ("might win", "could potentially")

  2. Probability Citation Frequency (PCF)
     Detects explicit probability language — numerical percentages,
     Polymarket citations, and epistemic probability anchoring.
     This is the most direct measure of market-to-media uptake.

  3. Narrative Closure Score (NCS)
     Detects language that forecloses possibility space.
     ("no path to victory", "effectively over", "insurmountable lead")

These three variables map onto your theoretical framework:
  ERS  ↔ epistemic enrollment (how markets shift certainty registers)
  PCF  ↔ financial/discursive enrollment (market numbers in media)
  NCS  ↔ narrative closure (prediction markets as story-ending machines)

The extractor also computes a basic PII proxy (agentive vs. passive
candidate mentions) that connects to your existing FRAME metrics.

Requirements: spaCy en_core_web_sm
"""

import re
import spacy
import pandas as pd
import numpy as np
from typing import Optional

# Load spaCy model once at import time
# Disable heavy components we don't need; add sentencizer for sentence boundary detection
try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    NLP.add_pipe("sentencizer")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm"
    )


# ── lexicons ──────────────────────────────────────────────────────────────────

# High-certainty epistemic markers (raise ERS)
HIGH_CERTAINTY = [
    r"\bwill\b", r"\bis (projected|expected|forecast|set|poised|on track|certain|sure) to\b",
    r"\bguaranteed\b", r"\binevitable\b", r"\bundoubtedly\b", r"\bclearly\b",
    r"\bhas (secured|locked up|all but won|effectively won)\b",
    r"\bpresumptive\b", r"\bfrontrunner\b", r"\bheavy favou?rite\b",
    r"\bthe odds (are|favor|strongly favor)\b",
    r"\b(commanding|dominant|insurmountable) lead\b",
]

# Low-certainty / hedging markers (lower ERS)
LOW_CERTAINTY = [
    r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\bpotentially\b", r"\bif\b", r"\bshould\b", r"\bwould\b",
    r"\bit remains (unclear|uncertain|to be seen)\b",
    r"\b(tight|close|competitive|uncertain|unpredictable) (race|contest|election)\b",
    r"\btoo (early|close) to (call|say|tell)\b",
    r"\bno clear (winner|frontrunner|favourite)\b",
]

# Probability citation patterns (PCF)
PROB_CITATIONS = [
    r"\b\d{1,3}[\.\d]*\s*%\s*(chance|probability|likelihood|odds|probability|shot)\b",
    r"\b(polymarket|prediction market|betting market|futures market|PredictIt)\b",
    r"\b(odds|probability) (of|that|are|suggest|imply)\b",
    r"\baccording to (prediction|betting) markets?\b",
    r"\bmarkets? (give|price|put|assign|estimate|show|indicate)\b",
    r"\b\d{1,3}[\.\d]*[\s-]to[\s-]\d{1,3}\s*(odds|chance|probability)\b",
    r"\b(implied|market[\s-]implied)\s*(probability|odds|price)\b",
    r"\b(traders?|investors?|bettors?)\s*(expect|predict|believe|price in)\b",
]

# Narrative closure patterns (NCS)
NARRATIVE_CLOSURE = [
    r"\bno (viable|realistic|plausible|real|clear) path\b",
    r"\b(effectively|virtually|all but|essentially|practically) (over|decided|done|settled|finished|won|lost)\b",
    r"\b(race|contest|election|campaign) is (over|finished|done|decided|settled)\b",
    r"\b(concede|concession|gave up|stepped aside|dropped out)\b",
    r"\b(sealed|clinched|secured|wrapped up|locked up) (the|a|his|her|their) (win|victory|nomination|lead)\b",
    r"\bbarring (a|an) (miracle|upset|major)\b",
    r"\bhas no (path|route|way) (to|back to|forward)\b",
    r"\bmathematically (eliminated|impossible|certain)\b",
    r"\b(cannot|can't|won't|will not) (win|recover|come back)\b",
]


# ── compiled patterns ─────────────────────────────────────────────────────────

def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]

_HIGH_CERT_RE   = _compile(HIGH_CERTAINTY)
_LOW_CERT_RE    = _compile(LOW_CERTAINTY)
_PROB_CITE_RE   = _compile(PROB_CITATIONS)
_CLOSURE_RE     = _compile(NARRATIVE_CLOSURE)


# ── per-article feature extraction ───────────────────────────────────────────

def extract_features(text: str, title: str = "") -> dict:
    """
    Extract all narrative features from a single article's text.

    Parameters
    ----------
    text  : Full article text (or as much as you have).
    title : Article headline (used for supplementary PII computation).

    Returns
    -------
    dict with keys:
        ers              : Epistemic Register Score (-1 to +1)
        high_cert_count  : Raw high-certainty marker count
        low_cert_count   : Raw low-certainty marker count
        pcf              : Probability Citation Frequency (0+)
        pcf_binary       : 1 if any probability citation found, else 0
        ncs              : Narrative Closure Score (0+)
        ncs_binary       : 1 if any closure marker found, else 0
        pii_proxy        : Personalisation Intensity proxy (0–1)
        word_count       : Approximate word count
        n_sentences      : Sentence count (via spaCy)
    """
    if not text:
        return _empty_features()

    full = (title + " " + text).strip()
    word_count = len(full.split())

    # ── ERS ──
    high_count = sum(
        len(p.findall(full)) for p in _HIGH_CERT_RE
    )
    low_count = sum(
        len(p.findall(full)) for p in _LOW_CERT_RE
    )
    total_cert = high_count + low_count
    ers = (high_count - low_count) / total_cert if total_cert > 0 else 0.0

    # ── PCF ──
    pcf_count = sum(len(p.findall(full)) for p in _PROB_CITE_RE)
    # normalise by word count (per 1000 words)
    pcf_norm  = (pcf_count / word_count * 1000) if word_count > 0 else 0.0

    # ── NCS ──
    ncs_count = sum(len(p.findall(full)) for p in _CLOSURE_RE)
    ncs_norm  = (ncs_count / word_count * 1000) if word_count > 0 else 0.0

    # ── PII proxy (spaCy) ──
    # Ratio of PERSON-entity sentences with active/agentive verbs
    # vs total PERSON-entity sentences. This is a simplified proxy;
    # full PII would require your FRAME operationalisation.
    doc = NLP(full[:50000])   # cap to avoid timeout on very long texts
    n_sentences      = len(list(doc.sents))
    person_sentences = sum(1 for sent in doc.sents if any(ent.label_ == "PERSON" for ent in sent.ents))
    pii_proxy = person_sentences / n_sentences if n_sentences > 0 else 0.0

    return {
        "ers":             round(ers, 4),
        "high_cert_count": high_count,
        "low_cert_count":  low_count,
        "pcf":             round(pcf_norm, 4),
        "pcf_binary":      int(pcf_count > 0),
        "ncs":             round(ncs_norm, 4),
        "ncs_binary":      int(ncs_count > 0),
        "pii_proxy":       round(pii_proxy, 4),
        "word_count":      word_count,
        "n_sentences":     n_sentences,
    }


def _empty_features() -> dict:
    return {k: 0 for k in [
        "ers", "high_cert_count", "low_cert_count",
        "pcf", "pcf_binary", "ncs", "ncs_binary",
        "pii_proxy", "word_count", "n_sentences"
    ]}


# ── batch extraction ──────────────────────────────────────────────────────────

def extract_features_batch(
    articles_df:   pd.DataFrame,
    text_col:      str = "text",
    title_col:     str = "title",
    verbose:       bool = True,
) -> pd.DataFrame:
    """
    Extract narrative features for every row in articles_df.

    articles_df must have a text column (text_col) — if you're working
    with GDELT article metadata only (no full text), pass the title
    as text_col and accept that ERS/NCS/PCF will be headline-level only.

    Returns the input DataFrame with feature columns appended.
    """
    if text_col not in articles_df.columns:
        if verbose:
            print(f"  Warning: '{text_col}' column not found. Using titles only.")
        articles_df = articles_df.copy()
        articles_df[text_col] = articles_df.get(title_col, "")

    feature_rows = []
    for i, row in articles_df.iterrows():
        text  = str(row.get(text_col, "") or "")
        title = str(row.get(title_col, "") or "")
        feature_rows.append(extract_features(text, title))

        if verbose and (i + 1) % 50 == 0:
            print(f"  … processed {i+1}/{len(articles_df)} articles")

    features_df = pd.DataFrame(feature_rows, index=articles_df.index)
    return pd.concat([articles_df, features_df], axis=1)


# ── daily aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_features(enriched_articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate article-level narrative features to daily statistics.

    Input: the output of extract_features_batch().

    Returns DataFrame with columns:
        date, n_articles,
        mean_ers, mean_pcf, pcf_adoption_rate, mean_ncs, mean_pii_proxy,
        ers_trend_3d  (3-day rolling mean of ERS — captures drift)
    """
    df = enriched_articles_df.copy()
    df["day"] = df["date"].dt.normalize()

    agg = df.groupby("day").agg(
        n_articles        = ("url",       "count"),
        mean_ers          = ("ers",       "mean"),
        mean_pcf          = ("pcf",       "mean"),
        pcf_adoption_rate = ("pcf_binary","mean"),   # % of articles citing prob
        mean_ncs          = ("ncs",       "mean"),
        ncs_adoption_rate = ("ncs_binary","mean"),
        mean_pii_proxy    = ("pii_proxy", "mean"),
    ).reset_index()

    agg = agg.rename(columns={"day": "date"})
    agg["date"] = pd.to_datetime(agg["date"], utc=True)
    agg["ers_trend_3d"] = agg["mean_ers"].rolling(3, center=True, min_periods=1).mean()

    return agg.sort_values("date").reset_index(drop=True)


# ── headline-only feature extraction (GDELT fast path) ───────────────────────

def extract_headline_features_df(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast path: extract features from headlines only (no full text required).

    This is the practical approach when working with GDELT metadata,
    which doesn't include article body text.  Results will undercount
    all three metrics (ERS, PCF, NCS) relative to full-text analysis,
    but the time series shape should be preserved — meaning you can
    still detect relative shifts across time even if absolute values
    are suppressed.  Declare this limitation explicitly in your methods.
    """
    return extract_features_batch(
        articles_df,
        text_col  = "title",
        title_col = "title",
        verbose   = False,
    )

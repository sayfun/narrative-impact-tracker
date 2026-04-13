"""
app.py
------
Streamlit web interface for the Narrative Impact Tracker.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push this repo to GitHub → connect at share.streamlit.io → done.

The app wraps the same NarrativePipeline, analysis, and feature
extractor as the CLI, but presents results as interactive Plotly charts
and lets users share analyses via URL query params.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta
import io

# ── page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Narrative Impact Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Narrative Impact Tracker** — empirically measures prediction "
            "market probability movements' influence on media narratives.\n\n"
            "Implements ERS, PCF, NCS metrics derived from the FRAME framework."
        ),
    },
)

# ── colour palette (matches CLI report for consistency) ───────────────────────

C = {
    "prob":     "#2166ac",
    "coverage": "#d6604d",
    "ers":      "#4dac26",
    "pcf":      "#b8860b",
    "ncs":      "#8b008b",
    "shock_up": "#1a9641",
    "shock_dn": "#d73027",
    "neutral":  "#aaaaaa",
    "bg":       "#fafafa",
}


# ── cached pipeline runner ────────────────────────────────────────────────────

@st.cache_data(
    show_spinner=False,
    ttl=3600,   # cache results for 1 hour — GDELT + Polymarket don't change fast
)
def run_pipeline_cached(
    market_query: str,
    topic_terms_tuple: tuple,   # tuples are hashable; lists are not
    start: str,
    end: str,
    shock_threshold: float,
    fetch_articles: bool,
    market_index: int = 0,
    mc_api_key: str = "",
    manual_token_id: str = "",
    manual_market_question: str = "",
):
    """
    Cached wrapper around NarrativePipeline.collect().

    Returns a dict of serialisable objects (DataFrames, dicts) so Streamlit
    can cache and hash them cleanly.

    Cache key is the full set of parameters — changing any input triggers
    a fresh API fetch.
    """
    from narrative_tracker.pipeline import NarrativePipeline
    from narrative_tracker.analysis import run_full_analysis
    from narrative_tracker.features import (
        extract_headline_features_df,
        extract_features_batch,
        aggregate_daily_features,
        compute_eai,
    )

    pipe = NarrativePipeline(
        market_query    = market_query,
        topic_terms     = list(topic_terms_tuple),
        start           = start,
        end             = end,
        shock_threshold = shock_threshold,
        fetch_articles  = fetch_articles,
        market_index    = market_index,
        manual_token_id = manual_token_id,
        manual_market_question = manual_market_question,
    )
    pipe.collect(verbose=False)

    features_daily   = None
    fulltext_source  = None   # "gdelt_headlines" | "mediacloud_fulltext"
    aligned_for_analysis = pipe.aligned.copy()

    # ── Attempt MediaCloud full-text enrichment ──
    mc_articles = None
    if mc_api_key and mc_api_key.strip() and fetch_articles:
        try:
            from narrative_tracker.mediacloud_client import (
                search_stories_windowed,
                build_mediacloud_query,
            )
            mc_query = build_mediacloud_query(list(topic_terms_tuple))
            mc_articles = search_stories_windowed(
                mc_query, start=start, end=end,
                api_key=mc_api_key.strip(),
                window_days=14, max_per_window=200,
                include_text=True, verbose=False,
            )
        except Exception:
            mc_articles = None   # fall through to GDELT headlines

    # ── Feature extraction (full-text if MediaCloud succeeded, else headlines) ──
    articles_for_features = mc_articles if (mc_articles is not None and not mc_articles.empty) else pipe.articles_df

    if articles_for_features is not None and not articles_for_features.empty:
        if mc_articles is not None and not mc_articles.empty:
            # Full-text path: use the text column for proper feature extraction
            enriched = extract_features_batch(
                articles_for_features,
                text_col="text", title_col="title", verbose=False,
            )
            fulltext_source = "mediacloud_fulltext"
        else:
            enriched = extract_headline_features_df(articles_for_features)
            fulltext_source = "gdelt_headlines"

        features_daily = aggregate_daily_features(enriched)
        features_daily = compute_eai(features_daily)

        if features_daily is not None and not features_daily.empty:
            feat_cols  = [c for c in features_daily.columns if c != "date"]
            feat_merge = features_daily[["date"] + feat_cols].copy()
            feat_merge["date"] = pd.to_datetime(feat_merge["date"], utc=True).dt.normalize()
            aligned_for_analysis["date"] = pd.to_datetime(
                aligned_for_analysis["date"], utc=True
            ).dt.normalize()
            aligned_for_analysis = aligned_for_analysis.merge(
                feat_merge, on="date", how="left"
            )

    analysis = run_full_analysis(aligned_for_analysis, pipe.shocks, verbose=False)

    return {
        "market_question":  pipe.market_meta["question"],
        "aligned":          aligned_for_analysis,
        "shocks":           pipe.shocks,
        "articles":         pipe.articles_df,
        "mc_articles":      mc_articles,
        "features_daily":   features_daily,
        "fulltext_source":  fulltext_source,
        "analysis":         analysis,
        "summary":          pipe.summary(),
    }


# ── plotly chart builders ─────────────────────────────────────────────────────

def fig_probability_coverage(aligned: pd.DataFrame, shocks: pd.DataFrame, title: str):
    """Dual-panel: probability series (top) + coverage intensity (bottom)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        subplot_titles=("Polymarket WIN probability", "GDELT coverage intensity"),
    )

    dates = pd.to_datetime(aligned["date"])
    prob  = aligned["probability"]

    # ── Panel 1: probability ──
    fig.add_trace(
        go.Scatter(
            x=dates, y=prob,
            mode="lines",
            name="WIN probability",
            line=dict(color=C["prob"], width=2.5),
            hovertemplate="%{x|%b %d}: <b>%{y:.1%}</b><extra></extra>",
        ),
        row=1, col=1,
    )
    # Fill above/below 50%
    fig.add_trace(
        go.Scatter(
            x=dates, y=prob,
            fill="tonexty", mode="none",
            fillcolor="rgba(33,102,172,0.08)",
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=0.5, line_dash="dot", line_color=C["neutral"],
        line_width=1, row=1, col=1,
        annotation_text="50%", annotation_position="right",
    )

    # Shock markers
    for _, s in shocks.iterrows():
        colour = C["shock_up"] if s["direction"] == "UP" else C["shock_dn"]
        label  = f"↑ +{s['delta']:.1%}" if s["direction"] == "UP" else f"↓ {s['delta']:.1%}"
        fig.add_vline(
            x=pd.Timestamp(s["date"]).timestamp() * 1000,
            line_color=colour, line_width=1.5, line_dash="dot",
            row=1, col=1,
            annotation_text=label,
            annotation_font_size=9,
            annotation_font_color=colour,
        )

    # ── Panel 2: coverage ──
    vol_col = "volume_intensity" if "volume_intensity" in aligned.columns else "volume_norm"
    vol = aligned[vol_col].fillna(0)

    fig.add_trace(
        go.Scatter(
            x=dates, y=vol,
            mode="lines",
            name="Coverage intensity",
            fill="tozeroy",
            line=dict(color=C["coverage"], width=1.5),
            fillcolor="rgba(214,96,77,0.18)",
            hovertemplate="%{x|%b %d}: <b>%{y:.4f}</b><extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(tickformat=".0%", row=1, col=1, title_text="Probability")
    fig.update_yaxes(row=2, col=1, title_text="GDELT intensity")
    fig.update_layout(
        title=dict(text=title[:100], font_size=14),
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=60, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def fig_narrative_features(features_daily: pd.DataFrame):
    """ERS, PCF adoption rate, NCS in a three-panel chart."""
    cols = [
        ("mean_ers",          "Epistemic Register Score (ERS)",    C["ers"],
         "Positive = high-certainty framing; negative = hedged language"),
        ("pcf_adoption_rate", "PCF Adoption Rate",                  C["pcf"],
         "% of articles citing explicit probability language"),
        ("mean_ncs",          "Narrative Closure Score (NCS)",      C["ncs"],
         "Density of possibility-foreclosure language"),
    ]
    present = [(col, label, colour, note)
               for col, label, colour, note in cols
               if col in features_daily.columns]
    if not present:
        return None

    fig = make_subplots(
        rows=len(present), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[label for _, label, _, _ in present],
    )
    dates = pd.to_datetime(features_daily["date"])

    for i, (col, label, colour, note) in enumerate(present, start=1):
        vals = features_daily[col].ffill()
        fig.add_trace(
            go.Scatter(
                x=dates, y=vals,
                mode="lines",
                name=label,
                line=dict(color=colour, width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(colour.lstrip('#')[j:j+2], 16) for j in (0,2,4)) + [0.12])}",
                hovertemplate=f"{label}: <b>%{{y:.4f}}</b><extra></extra>",
            ),
            row=i, col=1,
        )
        if col == "mean_ers":
            fig.add_hline(
                y=0, line_dash="dot", line_color=C["neutral"],
                line_width=1, row=i, col=1,
            )

    fig.update_layout(
        height=180 * len(present),
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def fig_cross_correlation(xcorr: dict):
    """Bar chart of cross-correlations at each lag."""
    lags  = xcorr["lags"]
    corrs = xcorr["correlations"]
    peak  = xcorr["peak_lag"]

    colours = [C["shock_dn"] if l < 0 else C["shock_up"] if l > 0 else C["neutral"]
               for l in lags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags, y=corrs,
        marker_color=colours,
        opacity=0.8,
        hovertemplate="Lag %{x}d: r = <b>%{y:.3f}</b><extra></extra>",
        name="Cross-correlation",
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.add_vline(x=peak, line_color="#555", line_dash="dash", line_width=1.5,
                  annotation_text=f"Peak: lag={peak}d", annotation_position="top")
    fig.add_hline(y=0, line_color="black", line_width=0.5)

    fig.update_layout(
        title=f"Cross-Correlation: Probability → Coverage  (peak lag={peak}d, r={xcorr['peak_corr']:.3f})",
        xaxis_title="Lag (days)   ← narrative leads | probability leads →",
        yaxis_title="r",
        height=320,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.add_annotation(
        x=0.01, y=0.97, xref="paper", yref="paper",
        text="<span style='color:#d73027'>■</span> Coverage leads prob  "
             "<span style='color:#1a9641'>■</span> Probability leads coverage",
        showarrow=False, font_size=11, align="left",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", dtick=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def fig_event_study(events_result: dict, aligned: pd.DataFrame):
    """Scatter plot: each shock event, coloured by coverage change."""
    events = events_result.get("events", [])
    if not events:
        return None

    rows = []
    for e in events:
        vol_change = (e["variables"].get("volume_norm") or {}).get("change")
        tone_change = (e["variables"].get("mean_tone") or {}).get("change")
        rows.append({
            "date":       e["date"],
            "direction":  e["direction"],
            "delta":      e["delta"],
            "probability":e["probability"],
            "vol_change": vol_change,
            "tone_change":tone_change,
        })
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for direction, colour in [("UP", C["shock_up"]), ("DOWN", C["shock_dn"])]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["delta"],
            y=sub["vol_change"] if "vol_change" in sub else [0]*len(sub),
            mode="markers+text",
            marker=dict(color=colour, size=10, opacity=0.8),
            text=sub["date"],
            textposition="top center",
            textfont=dict(size=9),
            name=f"Shock {direction}",
            hovertemplate=(
                "Date: %{text}<br>"
                "Δ probability: %{x:+.3f}<br>"
                "Δ coverage: %{y:+.4f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line_color=C["neutral"], line_dash="dot", line_width=1)
    fig.add_vline(x=0, line_color=C["neutral"], line_dash="dot", line_width=1)
    fig.update_layout(
        title="Event Study: Probability Shock vs. Coverage Change (±5-day window)",
        xaxis_title="Δ probability at shock",
        yaxis_title="Δ coverage volume (post − pre mean)",
        height=360,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", tickformat=".0%")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


def fig_eai(aligned: pd.DataFrame, features_daily: pd.DataFrame, shocks: pd.DataFrame):
    """
    Epistemic Authority Index vs. Probability — dual-panel.

    Top: Probability (blue) and smoothed EAI (gold) on a shared 0–1 axis.
         Convergence = EAI rising as probability hardens = market-to-narrative
         influence in action. Divergence after shocks = lag structure.

    Bottom: EAI component breakdown — ERS (green), PCF (amber), NCS (purple)
            stacked contributions that sum to the total EAI.
    """
    if features_daily is None or features_daily.empty or "eai" not in features_daily.columns:
        return None

    # Merge EAI onto the aligned date index for consistent x-axis
    feat = features_daily[["date", "eai_smooth", "eai_ers_contrib", "eai_pcf_contrib", "eai_ncs_contrib"]].copy()
    feat["date"] = pd.to_datetime(feat["date"], utc=True).dt.normalize()
    al = aligned.copy()
    al["date"] = pd.to_datetime(al["date"], utc=True).dt.normalize()
    merged = al[["date", "probability"]].merge(feat, on="date", how="left")

    dates = pd.to_datetime(merged["date"])
    prob  = merged["probability"]
    eai   = merged["eai_smooth"].fillna(method=None)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.40],
        vertical_spacing=0.06,
        subplot_titles=(
            "Probability vs. Epistemic Authority Index (EAI)",
            "EAI component breakdown",
        ),
    )

    # ── Panel 1: Probability ──
    fig.add_trace(
        go.Scatter(
            x=dates, y=prob,
            mode="lines", name="WIN probability",
            line=dict(color=C["prob"], width=2.5),
            hovertemplate="%{x|%b %d}: prob <b>%{y:.1%}</b><extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Panel 1: EAI (same axis, 0–1) ──
    fig.add_trace(
        go.Scatter(
            x=dates, y=eai,
            mode="lines", name="EAI (smoothed)",
            line=dict(color="#e6a817", width=2.5, dash="solid"),
            fill="tozeroy",
            fillcolor="rgba(230,168,23,0.10)",
            hovertemplate="%{x|%b %d}: EAI <b>%{y:.3f}</b><extra></extra>",
        ),
        row=1, col=1,
    )

    # Shock markers on panel 1
    for _, s in shocks.iterrows():
        colour = C["shock_up"] if s["direction"] == "UP" else C["shock_dn"]
        fig.add_vline(
            x=pd.Timestamp(s["date"]).timestamp() * 1000,
            line_color=colour, line_width=1.2, line_dash="dot",
            row=1, col=1,
        )

    # ── Panel 2: EAI components (stacked area) ──
    component_cols = [
        ("eai_pcf_contrib", "PCF (market citation)", C["pcf"]),
        ("eai_ers_contrib", "ERS (certainty register)", C["ers"]),
        ("eai_ncs_contrib", "NCS (narrative closure)", C["ncs"]),
    ]
    for col, label, colour in component_cols:
        if col in merged.columns:
            vals = merged[col].fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=dates, y=vals,
                    mode="lines", name=label,
                    stackgroup="eai_components",
                    line=dict(color=colour, width=1),
                    fillcolor=colour.replace("#", "rgba(") if False else colour,
                    hovertemplate=f"{label}: <b>%{{y:.3f}}</b><extra></extra>",
                ),
                row=2, col=1,
            )

    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=1, col=1, title_text="0 – 1 scale")
    fig.update_yaxes(row=2, col=1, title_text="EAI contribution")
    fig.update_layout(
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0, font_size=11),
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    return fig


# ── HTML report export (reuses CLI report generator) ─────────────────────────

def build_html_report(result: dict, gdelt_query: str, start: str, end: str) -> bytes:
    from narrative_tracker.report import generate_report
    html = generate_report(
        aligned_df       = result["aligned"],
        shocks_df        = result["shocks"],
        analysis_results = result["analysis"],
        market_question  = result["market_question"],
        gdelt_query      = gdelt_query,
        start            = start,
        end              = end,
        features_daily   = result["features_daily"],
    )
    return html.encode("utf-8")


# ── market search helper (cached separately so it's fast) ────────────────────

@st.cache_data(show_spinner=False, ttl=300, max_entries=50)
def search_markets_cached(query: str, include_active: bool = True, include_closed: bool = True, _v: int = 3):
    # _v is a version bump — increment to bust the cache after scoring fixes
    from narrative_tracker.polymarket import search_markets
    df = search_markets(query, limit=15, include_active=include_active, include_closed=include_closed)
    if df.empty:
        return df
    valid = df[df["token_ids"].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    return valid


# ── sidebar (inputs) ──────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image(
        "https://em-content.zobj.net/source/apple/391/bar-chart_1f4ca.png",
        width=40,
    )
    st.sidebar.title("Narrative Impact Tracker")
    st.sidebar.caption(
        "Empirically measures prediction market influence on media narratives."
    )
    st.sidebar.divider()

    # ── Curated historical markets (archived from API) ────────────────────────
    HISTORICAL_MARKETS = {
        "— pick a featured market —": None,
        "🇺🇸 Presidential Election Winner 2024 (Trump vs Harris)": {
            "token_id": "21742633143463906290569050155826389240456629048843189025793797045763932443804",
            "question": "Presidential Election Winner 2024",
            "suggested_terms": "Trump, Harris, election, president",
            "suggested_start": "2024-07-15",
            "suggested_end":   "2024-11-07",
        },
        "🇺🇸 Will Trump win the 2024 presidential election?": {
            "token_id": "69236923620077691027083946871148767382819025171534185409556716274216206771",
            "question": "Will Trump win the 2024 presidential election?",
            "suggested_terms": "Trump, election, Republican, MAGA",
            "suggested_start": "2024-07-15",
            "suggested_end":   "2024-11-07",
        },
        "🇺🇦 Ukraine ceasefire 2025–2026": {
            "token_id": "76885168648776633882454084559225614067599803855804481698042219701097697688",
            "question": "Ukraine ceasefire 2025–2026",
            "suggested_terms": "Ukraine, ceasefire, peace, war, Russia",
            "suggested_start": "2025-01-01",
            "suggested_end":   "2026-04-01",
        },
        "🇮🇷 US attack on Iran 2026": {
            "token_id": "55527562813268502763749084855188242669753310943082818604279838396371048583",
            "question": "US attack on Iran 2026",
            "suggested_terms": "Iran, attack, military, strike, nuclear",
            "suggested_start": "2025-10-01",
            "suggested_end":   "2026-04-13",
        },
    }

    # ── Step 1: search query ──
    st.sidebar.subheader("1 — Find a market")

    # Featured historical markets shortcut
    featured = st.sidebar.selectbox(
        "Featured historical markets",
        options=list(HISTORICAL_MARKETS.keys()),
        index=0,
        help="Pre-loaded markets that are no longer searchable via the Polymarket API.",
    )
    featured_data = HISTORICAL_MARKETS[featured]

    search_query = st.sidebar.text_input(
        "Or search Polymarket",
        value="" if featured_data else "presidential election winner 2024",
        help="Type any topic — e.g. 'US attack Iran', 'UK election', 'Fed rate cut'",
        disabled=bool(featured_data),
    )
    _fc1, _fc2 = st.sidebar.columns(2)
    include_active = _fc1.checkbox("Active", value=True,  help="Include currently trading markets", disabled=bool(featured_data))
    include_closed = _fc2.checkbox("Closed", value=False, help="Include resolved/closed markets", disabled=bool(featured_data))

    # ── Step 2: pick market ──
    market_index    = 0
    market_question = ""
    token_ids       = []
    manual_token    = ""

    if featured_data:
        # Featured historical market selected — use its token directly
        market_question = featured_data["question"]
        manual_token    = featured_data["token_id"]
        st.sidebar.success(f"✓ {market_question}")
    elif search_query:
        with st.sidebar:
            with st.spinner("Searching markets…"):
                try:
                    markets_df = search_markets_cached(search_query, include_active=include_active, include_closed=include_closed, _v=3)
                except Exception:
                    markets_df = None

        if markets_df is not None and not markets_df.empty:
            def _option_label(row):
                vol  = row.get("volume_24hr", row.get("volume", 0)) or 0
                tag  = "ACTIVE" if row.get("active") else "closed"
                q    = row.get("question", "")
                q_display = (q[:62] + "…") if len(q) > 62 else q
                return f"{q_display}  [{tag}, ${vol:,.0f}/24h]"

            options = [_option_label(row.to_dict()) for _, row in markets_df.iterrows()]
            chosen = st.sidebar.selectbox(
                "Select market",
                options=options,
                index=0,
                help="Matched Polymarket markets, ranked by text relevance then 24h trading volume.",
            )
            market_index    = options.index(chosen)
            selected_row    = markets_df.iloc[market_index]
            market_question = selected_row["question"]
            token_ids       = selected_row["token_ids"]
        elif markets_df is not None and markets_df.empty:
            st.sidebar.warning(
                f"No markets found for **\"{search_query}\"**. "
                "Try different keywords or use the featured markets above."
            )

    # ── Manual token ID fallback (for archived/historical markets) ──
    with st.sidebar.expander("Use token ID directly (for archived markets)"):
        st.markdown(
            "For historical markets not found by search (e.g. 2024 US election), "
            "paste the YES token ID from the Polymarket market URL or CLOB API.\n\n"
            "**How to find it:** Go to the market on polymarket.com → open browser devtools "
            "→ Network tab → look for the CLOB API call → copy the first value from `clobTokenIds`.\n\n"
            "Example (Trump 2024 winner): `21742633143463906290569050155826389240456629048843189025793797045763932443804`"
        )
        manual_token = st.text_input(
            "YES token ID",
            value="",
            placeholder="Paste token ID here…",
        )
        manual_question = st.text_input(
            "Market label (for display)",
            value="",
            placeholder="e.g. Presidential Election Winner 2024",
        )
        if manual_token and manual_question:
            market_question = manual_question
            token_ids       = [manual_token.strip()]
            st.success(f"Using manual token: {manual_token[:20]}…")

    # ── Step 3: topic terms for GDELT ──
    st.sidebar.divider()
    st.sidebar.subheader("2 — Coverage query")
    default_terms = featured_data["suggested_terms"] if featured_data else "Trump, Harris, election, president"
    topics_raw = st.sidebar.text_input(
        "Topic terms (comma-separated)",
        value=default_terms,
        help="Keywords for the GDELT media coverage search. 2–4 terms work best.",
    )
    topic_terms = [t.strip() for t in topics_raw.split(",") if t.strip()]

    # ── Step 4: date range ──
    st.sidebar.divider()
    st.sidebar.subheader("3 — Date range")
    col1, col2 = st.sidebar.columns(2)
    _default_start = date.fromisoformat(featured_data["suggested_start"]) if featured_data else date(2024, 7, 15)
    _default_end   = date.fromisoformat(featured_data["suggested_end"])   if featured_data else date(2024, 11, 7)
    start_date = col1.date_input("Start", value=_default_start)
    end_date   = col2.date_input("End",   value=_default_end)

    # ── Advanced ──
    st.sidebar.divider()
    with st.sidebar.expander("Advanced settings"):
        shock_threshold = st.slider(
            "Shock threshold (pp)",
            min_value=3, max_value=20, value=8,
            help="Minimum probability movement (percentage points over 2 days) to flag as a shock event.",
        )
        fetch_articles = st.checkbox(
            "Fetch article tone data",
            value=False,
            help="Fetches GDELT article metadata for tone + narrative features. Slower (~2 min for long windows).",
        )
        st.divider()
        # Use key from Streamlit secrets if available (set in the Streamlit Cloud
        # dashboard under Settings → Secrets as: MEDIACLOUD_API_KEY = "your-key")
        _server_mc_key = st.secrets.get("MEDIACLOUD_API_KEY", "") if hasattr(st, "secrets") else ""
        if _server_mc_key:
            st.caption("✓ MediaCloud full-text enabled")
            mc_api_key = _server_mc_key
        else:
            mc_api_key = st.text_input(
                "MediaCloud API key (optional)",
                value="",
                type="password",
                help=(
                    "Enables full-text article extraction for better ERS/PCF/NCS accuracy. "
                    "Free academic key at search.mediacloud.org/register"
                ),
            )

    run = st.sidebar.button(
        "Run analysis",
        type="primary",
        use_container_width=True,
        disabled=(not market_question or not topic_terms),
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "Data: [Polymarket CLOB API](https://clob.polymarket.com) · "
        "[GDELT v2](https://api.gdeltproject.org/api/v2/doc/doc)\n\n"
        "Both APIs are public and free."
    )

    return {
        "market_query":     search_query,
        "market_question":  market_question,
        "topic_terms":      topic_terms,
        "start":            start_date.isoformat(),
        "end":              end_date.isoformat(),
        "shock_threshold":  shock_threshold / 100,
        "fetch_articles":   fetch_articles,
        "market_index":     market_index,
        "manual_token_id":  manual_token if manual_token else "",
        "mc_api_key":       mc_api_key,
        "run":              run,
    }


# ── main results render ───────────────────────────────────────────────────────

def render_results(result: dict, inputs: dict):
    aligned  = result["aligned"]
    shocks   = result["shocks"]
    analysis = result["analysis"]
    features = result["features_daily"]
    question = result["market_question"]

    granger = analysis.get("granger", {})
    xcorr   = analysis.get("xcorr",   {})
    events  = analysis.get("events",  {})

    # ── header ──
    st.title("Narrative Impact Tracker")
    st.caption(
        f"Market: **{question}** · "
        f"Window: {inputs['start']} → {inputs['end']} · "
        f"{len(aligned)} days"
    )

    # ── summary KPIs ──
    import math
    prob_min  = aligned["probability"].min()
    prob_max  = aligned["probability"].max()
    prob_mean = aligned["probability"].mean()
    n_shocks  = len(shocks)
    peak_lag  = xcorr.get("peak_lag", "–")
    peak_corr = xcorr.get("peak_corr", None)

    def fmt_pct(v):
        return f"{v:.1%}" if (v is not None and not (isinstance(v, float) and math.isnan(v))) else "no data"

    # Warn if no probability data was returned for this window
    if aligned["probability"].isna().all():
        st.warning(
            "⚠️ **No probability data found for this date window.** "
            "This usually means the matched Polymarket market closed before your start date. "
            "Check the market name below and try an earlier date range, or refine your search query.",
            icon="⚠️",
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Probability range", f"{fmt_pct(prob_min)} – {fmt_pct(prob_max)}")
    k2.metric("Mean probability",  fmt_pct(prob_mean))
    k3.metric("Sharp movements",   n_shocks,   help="Probability events ≥ shock threshold")
    k4.metric("Peak X-corr lag",
              f"{peak_lag}d" if isinstance(peak_lag, int) else "–",
              help="Lag at which probability-coverage correlation peaks")
    k5.metric("Peak r",
              f"{peak_corr:.3f}" if peak_corr is not None else "–")

    st.divider()

    # ── Section 1: Timeline ──
    st.subheader("1 — Probability & Coverage Timeline")
    st.caption(
        "Polymarket WIN probability (blue) and GDELT prediction-market coverage intensity (red). "
        "Dotted vertical lines = sharp probability movements."
    )
    fig1 = fig_probability_coverage(aligned, shocks, question)
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("Methodological note: GDELT coverage metric"):
        st.markdown(
            "GDELT's *TimelineVol* returns a **relative volume intensity** — share of global "
            "GDELT traffic — not raw article counts. Suitable for detecting relative shifts; "
            "do not interpret absolute values as coverage totals. "
            "Coverage query: `" + inputs.get("gdelt_query", "") + "`"
        )

    st.divider()

    fulltext_source = result.get("fulltext_source")

    # ── Section 2: Narrative features ──
    if features is not None and not features.empty:
        st.subheader("2 — Narrative Feature Time Series")
        source_note = (
            "📄 **Full-text analysis via MediaCloud** — ERS, PCF, NCS extracted from full article text."
            if fulltext_source == "mediacloud_fulltext"
            else "Headline-level analysis (GDELT). Absolute values suppressed; time-series patterns preserved."
        )
        st.caption(source_note)
        fig2 = fig_narrative_features(features)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Variable definitions"):
            st.markdown("""
| Variable | Full name | What it measures |
|----------|-----------|-----------------|
| **ERS** | Epistemic Register Score | Ratio of high-certainty to hedging language. Positive = "will win"; negative = "might win" |
| **PCF** | Probability Citation Frequency | Density of explicit probability language and Polymarket mentions per 1000 words |
| **NCS** | Narrative Closure Score | Density of possibility-foreclosing language ("no path to victory", "effectively over") |
| **PII** | Personalisation Intensity Index | Fraction of sentences containing named persons (proxy for character-driven framing) |
""")
        st.divider()

    # ── Section 2b: EAI ──
    if features is not None and not features.empty and "eai" in features.columns:
        mc_count = len(result.get("mc_articles") or [])
        st.subheader("2b — Epistemic Authority Index (EAI)")

        eai_caption = (
            f"EAI computed from **{mc_count} full-text MediaCloud articles**. "
            "Convergence between EAI (gold) and probability (blue) = market framing adopted by media."
            if fulltext_source == "mediacloud_fulltext"
            else
            "EAI computed from GDELT headlines. Provide a MediaCloud key for full-text EAI. "
            "Convergence between EAI (gold) and probability (blue) = market framing adopted by media."
        )
        st.caption(eai_caption)

        fig_e = fig_eai(aligned, features, shocks)
        if fig_e:
            st.plotly_chart(fig_e, use_container_width=True)

        # Current EAI value
        latest_eai = features["eai"].dropna().iloc[-1] if not features["eai"].dropna().empty else None
        peak_eai   = features["eai"].max()
        eai_c1, eai_c2 = st.columns(2)
        if latest_eai is not None:
            eai_c1.metric("Latest EAI", f"{latest_eai:.3f}", help="0 = purely hedged/agnostic framing; 1 = fully probability-anchored, declarative")
        if peak_eai:
            eai_c2.metric("Peak EAI", f"{peak_eai:.3f}")

        with st.expander("How EAI is calculated"):
            st.markdown("""
**EAI = 0.40 × PCF + 0.35 × ERS + 0.25 × NCS**

| Component | Weight | What it captures |
|-----------|--------|-----------------|
| PCF (Probability Citation Frequency) | 40% | Most direct measure: market numbers quoted as fact |
| ERS (Epistemic Register Score) | 35% | Language shift from hedged ("might") to declarative ("will") |
| NCS (Narrative Closure Score) | 25% | Possibility-foreclosing language as probability hardens |

Each component is normalised to 0–1 before weighting. EAI = 0 means purely hedged, uncertain framing; EAI = 1 means fully probability-anchored, declarative coverage.

*The convergence pattern — EAI rising toward the probability line after a sharp movement — is the visual test of market-to-narrative influence. Lag between the probability spike and EAI response is interpretable as the media uptake delay.*
""")
        st.divider()

    elif features is None and inputs.get("mc_api_key"):
        st.info("Enable 'Fetch article tone data' to compute the Epistemic Authority Index.", icon="ℹ️")

    # ── Section 3: Cross-correlation ──
    st.subheader("3 — Cross-Correlation Analysis")
    if xcorr.get("lags"):
        fig3 = fig_cross_correlation(xcorr)
        st.plotly_chart(fig3, use_container_width=True)
        st.info(xcorr.get("interpretation", ""))

    with st.expander("Why cross-correlation isn't causality"):
        st.markdown(
            "Cross-correlation detects co-movement at different lags, not structural causation. "
            "A positive peak lag (probability leads coverage) is *consistent with* market-to-narrative "
            "influence, but both series may be responding to the same underlying event "
            "(debate, polling release) with different lag structures. "
            "See the Granger tests and event study below for stronger identification."
        )

    st.divider()

    # ── Section 4: Granger causality ──
    st.subheader("4 — Granger Causality Tests")
    summary_df = granger.get("summary_table", pd.DataFrame())

    if not summary_df.empty:
        # Style the significance column
        def style_sig(val):
            if val == "Yes":
                return "color: #1a9641; font-weight: bold"
            return "color: #aaa"

        st.dataframe(
            summary_df.style.map(style_sig, subset=["forward_sig"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(granger.get("interpretation", ""))
    else:
        st.warning("Insufficient data for Granger tests (need ≥ 20 observations after differencing).")

    # ADF stationarity
    with st.expander("Stationarity tests (ADF)"):
        adf_rows = []
        for k, v in granger.get("stationarity", {}).items():
            if isinstance(v, dict) and "p_value" in v:
                adf_rows.append({
                    "Series":       v["name"],
                    "ADF stat":     round(v["adf_stat"], 4),
                    "p-value":      round(v["p_value"], 4),
                    "Stationary":   "✓" if v["is_stationary"] else "✗",
                })
        if adf_rows:
            st.dataframe(pd.DataFrame(adf_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Non-stationary series (✗) were first-differenced before Granger tests. "
            "ADF null hypothesis: series has a unit root (non-stationary)."
        )

    with st.expander("Granger causality ≠ structural causality"):
        st.markdown(
            "A significant forward Granger result (p < 0.05) means lagged probability movements "
            "improve out-of-sample prediction of the narrative variable — but this does **not** "
            "identify a structural causal mechanism. For publication, complement with:\n"
            "- Manual event coding (clean vs. confounded shocks)\n"
            "- Robustness check across different shock thresholds\n"
            "- Qualitative close-reading of high-impact event windows"
        )

    st.divider()

    # ── Section 5: Event study ──
    st.subheader("5 — Event Study")
    agg_table = events.get("aggregate_table", pd.DataFrame())

    if not agg_table.empty:
        st.dataframe(agg_table, use_container_width=True, hide_index=True)

        fig5 = fig_event_study(events, aligned)
        if fig5:
            st.plotly_chart(fig5, use_container_width=True)

        st.caption(events.get("interpretation", ""))

        # Individual events table
        with st.expander(f"Individual shock events ({len(shocks)})"):
            st.dataframe(shocks, use_container_width=True, hide_index=True)
            st.markdown(
                "**Next step for publication:** Manually classify each shock as "
                "**clean** (no major scheduled event within ±72 hours) or **confounded**. "
                "Report Granger and event study results separately for each group."
            )
    else:
        st.warning(
            "No shock events detected. Try lowering the shock threshold in the sidebar "
            "or extending the analysis window."
        )

    st.divider()

    # ── Downloads ──
    st.subheader("Download")
    dl1, dl2, dl3 = st.columns(3)

    csv_aligned = aligned.to_csv(index=False).encode("utf-8")
    dl1.download_button(
        "aligned_frame.csv",
        data=csv_aligned,
        file_name="aligned_frame.csv",
        mime="text/csv",
        use_container_width=True,
        help="Main analytical dataset: probability + coverage + narrative features, daily",
    )

    if features is not None and not features.empty:
        csv_feats = features.to_csv(index=False).encode("utf-8")
        dl2.download_button(
            "narrative_features_daily.csv",
            data=csv_feats,
            file_name="narrative_features_daily.csv",
            mime="text/csv",
            use_container_width=True,
            help="Daily ERS, PCF, NCS, PII aggregates",
        )

    from narrative_tracker.gdelt import build_prediction_market_query
    gdelt_query = build_prediction_market_query(inputs["topic_terms"][:4])
    html_report = build_html_report(result, gdelt_query, inputs["start"], inputs["end"])
    dl3.download_button(
        "narrative_impact_report.html",
        data=html_report,
        file_name="narrative_impact_report.html",
        mime="text/html",
        use_container_width=True,
        help="Self-contained HTML report (same as CLI output) for sharing/archiving",
    )


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    inputs = render_sidebar()

    # Check if inputs changed since last run — if so, clear stale results
    last = st.session_state.get("last_inputs", {})
    inputs_changed = (
        inputs["market_query"]   != last.get("market_query") or
        inputs["topic_terms"]    != last.get("topic_terms") or
        inputs["start"]          != last.get("start") or
        inputs["end"]            != last.get("end")
    )
    if inputs_changed:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_inputs", None)

    # Show cached result if available and user hasn't clicked Run
    if "last_result" in st.session_state and not inputs["run"]:
        render_results(st.session_state["last_result"], st.session_state["last_inputs"])
        return

    if not inputs["run"]:
        # Landing / About page
        st.title("Narrative Impact Tracker")
        st.caption("Empirically measure the influence of prediction market probability movements on media narratives.")

        tab1, tab2 = st.tabs(["How to use", "About & methodology"])

        with tab1:
            st.markdown("""
### Quick start

1. **Search** for a topic in the sidebar — e.g. `Iran attack`, `Fed rate cut`, `Ukraine ceasefire`
2. **Select** the matching market from the dropdown (shows real market names + live volume)
3. **Set topic terms** for the media coverage search — 2–4 keywords work best
4. **Set a date range** — use the period when the market was actively trading
5. Click **Run analysis**

### What you get

| Output | What it shows |
|--------|--------------|
| Probability & coverage timeline | Polymarket WIN probability vs. GDELT media intensity, with shock events marked |
| Narrative feature time series | ERS, PCF, NCS extracted from article headlines over time |
| Cross-correlation | At which lag does probability *lead* coverage? |
| Granger causality tests | Does probability movement predict narrative shifts? (5-lag, ADF-corrected, bidirectional) |
| Event study | Pre/post narrative change around sharp probability movements |
| Downloads | aligned_frame.csv · narrative_features_daily.csv · self-contained HTML report |

### Best markets to analyse

| Search query | Notes |
|---|---|
| `presidential election winner 2024` | Richest historical dataset — July–November 2024 |
| `Iran attack military` | Active right now (April 2026) |
| `Ukraine ceasefire peace` | Long-running, high media coverage |
| `Fed rate cut 2025` | Strong macroeconomic coverage |

### Tips
- **Uncheck "Fetch article tone data"** for a fast first pass — cuts run time from ~3 min to ~10 sec
- If you get a GDELT rate limit error, wait 60 seconds and try again, or shorten the date range
- The date range should cover the period when the market was *actively trading*, not after it resolved
""")

        with tab2:
            st.markdown("""
### What this tool measures

Prediction markets like Polymarket don't just aggregate information — they construct and circulate stories about possible futures. This tool asks: **do probability movements cause measurable changes in how media frames a story?**

It implements a three-mode enrollment framework connecting financial, discursive, and epistemic dimensions of prediction market influence on narrative.

### Narrative variables

| Variable | Full name | What it measures |
|----------|-----------|-----------------|
| **ERS** | Epistemic Register Score | Ratio of high-certainty language ("will win", "inevitable") to hedging language ("might", "could"). Positive = certain framing; negative = hedged. |
| **PCF** | Probability Citation Frequency | Density of explicit probability language and prediction market citations per 1,000 words. Direct measure of market-to-media uptake. |
| **NCS** | Narrative Closure Score | Density of possibility-foreclosing language ("no path to victory", "effectively over", "mathematically eliminated"). Measures prediction markets as story-ending machines. |
| **PII** | Personalisation Intensity (proxy) | Fraction of sentences containing named persons. Captures secondary narrativisation — when structural probability becomes character-driven story. |
| **EAI** | Epistemic Authority Index | Composite index (0–1) combining PCF, ERS, and NCS into a single measure of how strongly coverage positions market probability as narrative authority. EAI = 0.40×PCF + 0.35×ERS + 0.25×NCS. |

*ERS, PCF, NCS, and EAI are extracted from article headlines when using GDELT (free, no key required). Provide a [MediaCloud API key](https://search.mediacloud.org/register) for full-text extraction — this substantially improves metric sensitivity. Validate against a human-coded sample before using as primary outcomes in publication.*

### Statistical methods

**Granger causality** — tests whether lagged probability movements predict narrative variable shifts above and beyond the series' own lags. Applied to first-differenced series after ADF stationarity testing. Both directions tested (probability → narrative and narrative → probability).

**Event study** — identifies sharp probability movements (≥8pp over 2 days by default) and measures pre/post changes in narrative variables within a ±5-day window.

**Cross-correlation** — peak lag at which probability-coverage co-movement is strongest.

> ⚠️ **Granger causality ≠ structural causation.** Both series may respond to a common event with different lag structures. For publication, manually classify each shock event as "clean" (no major scheduled event within ±72 hours) or "confounded" and report results separately.

### Data sources

| Source | What it provides | Cost |
|--------|-----------------|------|
| [Polymarket CLOB API](https://clob.polymarket.com) | Daily YES/NO probabilities | Free, public |
| [GDELT v2 Doc API](https://api.gdeltproject.org) | Coverage volume, article tone, headline text | Free, public |
| [MediaCloud](https://search.mediacloud.org) | Full article text across thousands of outlets | Free (academic key) |

### Known limitations

- GDELT skews toward English-language wire services; under-represents Substack, financial media, non-English press
- Polymarket market search covers ~400 most-active markets; niche topics may not appear
- Polymarket coverage is strongest for US politics, geopolitics, macro, and crypto
- Granger tests require ≥20 observations after differencing; short windows may produce no results

### Citation

If you use this tool in published research:

```
Funk, S. H. (2026). Narrative Impact Tracker [Software].
https://github.com/sayfun/narrative-impact-tracker
```

### Source code

[github.com/sayfun/narrative-impact-tracker](https://github.com/sayfun/narrative-impact-tracker) — MIT licence
""")
        return

    # Run pipeline
    with st.status("Running analysis…", expanded=True) as status:
        st.write("Searching Polymarket…")
        try:
            result = run_pipeline_cached(
                market_query           = inputs["market_query"],
                topic_terms_tuple      = tuple(inputs["topic_terms"]),
                start                  = inputs["start"],
                end                    = inputs["end"],
                shock_threshold        = inputs["shock_threshold"],
                fetch_articles         = inputs["fetch_articles"],
                market_index           = inputs["market_index"],
                mc_api_key             = inputs.get("mc_api_key", ""),
                manual_token_id        = inputs.get("manual_token_id", ""),
                manual_market_question = inputs.get("market_question", ""),
            )
        except RuntimeError as e:
            status.update(label="Failed", state="error")
            st.error(f"**Pipeline error:** {e}")
            st.info(
                "Try `narrative-tracker markets --query '…'` to find valid market queries.",
                icon="💡",
            )
            return
        except Exception as e:
            status.update(label="Failed", state="error")
            msg = str(e)
            if "429" in msg or "Too Many Requests" in msg:
                st.warning(
                    "**GDELT rate limit hit (429).** GDELT's free API allows ~1 request/second. "
                    "The tool now retries automatically with backoff — but very long date ranges "
                    "or many topic terms can still exceed it. Try:\n"
                    "- Shortening the date range\n"
                    "- Unticking 'Fetch article tone data' (much fewer requests)\n"
                    "- Waiting 60 seconds and running again"
                )
            else:
                st.error(f"**Unexpected error:** {e}")
            return

        st.write(f"✓ Matched market: **{result['market_question']}**")
        st.write("Building aligned dataset…")
        st.write("Running statistical analysis…")
        st.write("Rendering charts…")
        status.update(label="Analysis complete", state="complete")

    # Persist to session state so back-navigation doesn't re-run
    st.session_state["last_result"] = result
    st.session_state["last_inputs"] = inputs

    render_results(result, inputs)


if __name__ == "__main__":
    main()

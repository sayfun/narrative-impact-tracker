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
        aggregate_daily_features,
    )

    pipe = NarrativePipeline(
        market_query    = market_query,
        topic_terms     = list(topic_terms_tuple),
        start           = start,
        end             = end,
        shock_threshold = shock_threshold,
        fetch_articles  = fetch_articles,
        market_index    = market_index,
    )
    pipe.collect(verbose=False)

    features_daily = None
    aligned_for_analysis = pipe.aligned.copy()

    if pipe.articles_df is not None and not pipe.articles_df.empty:
        enriched = extract_headline_features_df(pipe.articles_df)
        features_daily = aggregate_daily_features(enriched)

        if features_daily is not None and not features_daily.empty:
            feat_cols = [c for c in features_daily.columns if c != "date"]
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
        "features_daily":   features_daily,
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

@st.cache_data(show_spinner=False, ttl=300)
def search_markets_cached(query: str):
    from narrative_tracker.polymarket import search_markets
    df = search_markets(query, limit=8)
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

    # ── Step 1: search query ──
    st.sidebar.subheader("1 — Find a market")
    search_query = st.sidebar.text_input(
        "Search Polymarket",
        value="presidential election winner 2024",
        help="Type any topic — e.g. 'US attack Iran', 'UK election', 'Fed rate cut'",
    )

    # ── Step 2: pick from real market names ──
    market_index = 0
    market_question = ""
    token_ids = []

    if search_query:
        with st.sidebar:
            with st.spinner("Searching markets…"):
                try:
                    markets_df = search_markets_cached(search_query)
                except Exception:
                    markets_df = None

        if markets_df is not None and not markets_df.empty:
            options = [
                f"{row['question'][:62]}…  [{'ACTIVE' if row['active'] else 'closed'}, ${row['volume_24hr']:,.0f}/24h]"
                if len(row['question']) > 62
                else f"{row['question']}  [{'ACTIVE' if row['active'] else 'closed'}, ${row['volume_24hr']:,.0f}/24h]"
                for _, row in markets_df.iterrows()
            ]
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
            st.sidebar.error(
                f"No Polymarket markets found matching **\"{search_query}\"**.\n\n"
                "Try different keywords — e.g. `Iran attack`, `Fed rate cut`, "
                "`Ukraine ceasefire`. Browse all active markets at "
                "[polymarket.com](https://polymarket.com)."
            )

    # ── Step 3: topic terms for GDELT ──
    st.sidebar.divider()
    st.sidebar.subheader("2 — Coverage query")
    topics_raw = st.sidebar.text_input(
        "Topic terms (comma-separated)",
        value="Trump, Harris, election, president",
        help="Keywords for the GDELT media coverage search. 2–4 terms work best.",
    )
    topic_terms = [t.strip() for t in topics_raw.split(",") if t.strip()]

    # ── Step 4: date range ──
    st.sidebar.divider()
    st.sidebar.subheader("3 — Date range")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=date(2024, 7, 15))
    end_date   = col2.date_input("End",   value=date(2024, 11, 7))

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

    # ── Section 2: Narrative features ──
    if features is not None and not features.empty:
        st.subheader("2 — Narrative Feature Time Series")
        st.caption(
            "ERS, PCF, NCS extracted at headline level. Absolute values are suppressed "
            "relative to full-text analysis; time-series shape is preserved."
        )
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
        # Landing state
        st.title("Narrative Impact Tracker")
        st.markdown("""
**Empirically measure the influence of prediction market probability movements on media narratives.**

This tool implements the FRAME framework (ERS, PCF, NCS metrics) and tests whether
Polymarket probability shifts Granger-cause changes in media coverage framing.

---

### How to use

1. **Set a market query** — e.g. `"presidential election winner 2024"` or `"UK general election"`
2. **Add topic terms** — comma-separated keywords for the GDELT coverage query
3. **Set a date range** — the analysis window
4. Click **Run analysis**

Results include:
- Interactive probability + coverage timeline
- ERS / PCF / NCS narrative feature time series
- Cross-correlation analysis (does probability *lead* coverage?)
- Granger causality tests (5-lag, ADF-corrected, bidirectional)
- Event study around sharp probability movements

---

### Data sources

| Source | What it provides | Cost |
|--------|-----------------|------|
| Polymarket CLOB API | Daily YES/NO probabilities | Free |
| GDELT v2 Doc API | Coverage volume + article tone | Free |

""")
        st.info(
            "**Tip:** Run `narrative-tracker markets --query 'your topic'` in terminal "
            "to browse available Polymarket markets before searching here.",
            icon="💡",
        )
        return

    # Run pipeline
    with st.status("Running analysis…", expanded=True) as status:
        st.write("Searching Polymarket…")
        try:
            result = run_pipeline_cached(
                market_query        = inputs["market_query"],
                topic_terms_tuple   = tuple(inputs["topic_terms"]),
                start               = inputs["start"],
                end                 = inputs["end"],
                shock_threshold     = inputs["shock_threshold"],
                fetch_articles      = inputs["fetch_articles"],
                market_index        = inputs["market_index"],
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

"""
report.py
---------
Self-contained HTML report generator for the Narrative Impact Tracker.

Produces a single .html file with:
  - Inline CSS (no external dependencies)
  - Base64-embedded charts (matplotlib → PNG → base64)
  - Summary statistics
  - Probability & coverage timeline
  - Narrative feature charts (ERS, PCF, NCS)
  - Granger causality results table
  - Event study table
  - Cross-correlation chart
  - Methodology notes (important for peer review)

The report is designed to accompany a methods paper — it prioritises
clarity and interpretive honesty over visual polish.
"""

import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── chart helpers ─────────────────────────────────────────────────────────────

COLOURS = {
    "prob":     "#2166ac",
    "prob_fill":"#d0e4f3",
    "coverage": "#d6604d",
    "ers":      "#4dac26",
    "pcf":      "#b8860b",
    "ncs":      "#8b008b",
    "shock_up": "#1a9641",
    "shock_dn": "#d73027",
    "neutral":  "#aaaaaa",
    "grid":     "#eeeeee",
}

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_style(ax):
    ax.set_facecolor("white")
    ax.grid(color=COLOURS["grid"], linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _format_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


# ── individual charts ─────────────────────────────────────────────────────────

def chart_probability_and_coverage(aligned_df: pd.DataFrame,
                                   shocks_df:  pd.DataFrame,
                                   question:   str) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(question[:90], fontsize=11, fontweight="bold", y=0.98)

    dates = pd.to_datetime(aligned_df["date"]).dt.to_pydatetime()
    prob  = aligned_df["probability"].values

    # ── Panel 1: Probability ──
    ax1.plot(dates, prob, color=COLOURS["prob"], linewidth=2, zorder=3, label="WIN probability")
    ax1.fill_between(dates, 0.5, prob,
                     where=(prob >= 0.5), interpolate=True,
                     alpha=0.18, color=COLOURS["prob"])
    ax1.fill_between(dates, prob, 0.5,
                     where=(prob < 0.5), interpolate=True,
                     alpha=0.12, color=COLOURS["coverage"])
    ax1.axhline(0.5, color=COLOURS["neutral"], linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.set_ylabel("Probability (YES)", fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Mark shocks
    if not shocks_df.empty:
        for _, s in shocks_df.iterrows():
            colour = COLOURS["shock_up"] if s["direction"] == "UP" else COLOURS["shock_dn"]
            ax1.axvline(pd.Timestamp(s["date"]).to_pydatetime(),
                        color=colour, alpha=0.5, linewidth=1.5, linestyle=":")
        ax1.plot([], [], color=COLOURS["shock_up"], linestyle=":", linewidth=1.5, label="↑ Sharp movement")
        ax1.plot([], [], color=COLOURS["shock_dn"], linestyle=":", linewidth=1.5, label="↓ Sharp movement")

    ax1.legend(fontsize=8, loc="upper left")
    _apply_style(ax1)

    # ── Panel 2: Coverage volume ──
    if "volume_intensity" in aligned_df.columns:
        vol = aligned_df["volume_intensity"].fillna(0).values
    else:
        vol = aligned_df["volume_norm"].fillna(0).values

    ax2.fill_between(dates, 0, vol, alpha=0.6, color=COLOURS["coverage"], zorder=2)
    ax2.plot(dates, vol, color=COLOURS["coverage"], linewidth=1.2, zorder=3)
    ax2.set_ylabel("Coverage intensity\n(GDELT relative volume)", fontsize=8)
    ax2.set_xlabel("")
    _apply_style(ax2)
    _format_xaxis(ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _fig_to_b64(fig)


def chart_narrative_features(features_daily: pd.DataFrame) -> str:
    cols_present = [c for c in ["mean_ers", "pcf_adoption_rate", "mean_ncs", "mean_pii_proxy"]
                    if c in features_daily.columns and not features_daily[c].isna().all()]
    if not cols_present:
        return ""

    fig, axes = plt.subplots(len(cols_present), 1, figsize=(13, 3 * len(cols_present)),
                              sharex=True)
    if len(cols_present) == 1:
        axes = [axes]

    labels = {
        "mean_ers":          ("Epistemic Register Score (ERS)", COLOURS["ers"],
                              "Positive = high certainty framing; negative = hedged"),
        "pcf_adoption_rate": ("Probability Citation Frequency — adoption rate", COLOURS["pcf"],
                              "% of articles citing explicit probability language"),
        "mean_ncs":          ("Narrative Closure Score (NCS)", COLOURS["ncs"],
                              "Density of possibility-foreclosure language"),
        "mean_pii_proxy":    ("Personalisation Intensity (PII proxy)", COLOURS["prob"],
                              "% of sentences containing named persons"),
    }

    dates = pd.to_datetime(features_daily["date"]).dt.to_pydatetime()

    for ax, col in zip(axes, cols_present):
        label, colour, subtitle = labels.get(col, (col, "#555", ""))
        vals = features_daily[col].ffill().values  # pandas >= 2.2 compatible
        ax.plot(dates, vals, color=colour, linewidth=1.8)
        ax.fill_between(dates, 0, vals, alpha=0.15, color=colour)
        if col == "mean_ers":
            ax.axhline(0, color=COLOURS["neutral"], linewidth=0.8, linestyle="--")
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(subtitle, fontsize=7.5, color="#666", pad=2)
        _apply_style(ax)

    _format_xaxis(axes[-1])
    plt.suptitle("Narrative Feature Time Series (headline-level)", fontsize=10,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_cross_correlation(xcorr: dict) -> str:
    lags  = xcorr["lags"]
    corrs = xcorr["correlations"]
    peak  = xcorr["peak_lag"]

    fig, ax = plt.subplots(figsize=(9, 4))
    colours = [COLOURS["shock_dn"] if l < 0 else
               COLOURS["shock_up"] if l > 0 else
               COLOURS["neutral"] for l in lags]
    ax.bar(lags, corrs, color=colours, alpha=0.75, width=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Lag (days)  ←narrative leads | prob leads→", fontsize=9)
    ax.set_ylabel("Cross-correlation (r)", fontsize=9)
    ax.set_title("Cross-Correlation: Probability vs. Coverage Volume\n"
                 f"Peak at lag={peak} days (r={xcorr['peak_corr']:.3f})", fontsize=10)
    ax.plot([], [], color=COLOURS["shock_dn"], label="Coverage leads probability")
    ax.plot([], [], color=COLOURS["shock_up"], label="Probability leads coverage")
    ax.legend(fontsize=8)
    _apply_style(ax)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── HTML assembly ─────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: Georgia, 'Times New Roman', serif; color: #222;
       background: #fafafa; padding: 0 0 60px 0; line-height: 1.6; }
.page { max-width: 1060px; margin: 0 auto; padding: 0 24px; }
h1 { font-size: 1.5rem; font-weight: bold; margin: 32px 0 4px 0; color: #1a1a1a; }
h2 { font-size: 1.1rem; font-weight: bold; margin: 36px 0 8px 0;
     color: #2166ac; border-bottom: 2px solid #dde; padding-bottom: 4px; }
h3 { font-size: 0.95rem; font-weight: bold; margin: 20px 0 6px 0; color: #444; }
p, li { font-size: 0.9rem; margin-bottom: 8px; }
.meta { font-size: 0.82rem; color: #666; margin-bottom: 24px; }
.badge { display: inline-block; background: #e8f0fb; color: #2166ac;
         border-radius: 4px; padding: 2px 8px; font-size: 0.78rem;
         font-family: monospace; margin: 2px; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin: 16px 0; }
.stat { background: white; border: 1px solid #dde; border-radius: 6px;
        padding: 14px 16px; }
.stat .val { font-size: 1.5rem; font-weight: bold; color: #2166ac; }
.stat .lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }
img.chart { width: 100%; border: 1px solid #dde; border-radius: 6px;
            margin: 12px 0; background: white; }
table { width: 100%; border-collapse: collapse; font-size: 0.83rem;
        margin: 12px 0; background: white; }
thead { background: #2166ac; color: white; }
th { padding: 8px 10px; text-align: left; font-weight: normal; }
td { padding: 7px 10px; border-bottom: 1px solid #eee; }
tr:hover td { background: #f5f8ff; }
.sig-yes { color: #1a9641; font-weight: bold; }
.sig-no  { color: #aaa; }
.note { background: #fffbea; border-left: 4px solid #e6b800;
        padding: 10px 14px; font-size: 0.82rem; margin: 14px 0;
        border-radius: 0 4px 4px 0; }
.section { margin-top: 40px; }
footer { margin-top: 60px; font-size: 0.75rem; color: #999;
         border-top: 1px solid #ddd; padding-top: 14px; }
"""

def _stat_box(value, label):
    return f'<div class="stat"><div class="val">{value}</div><div class="lbl">{label}</div></div>'


def _granger_table(granger: dict) -> str:
    df = granger.get("summary_table", pd.DataFrame())
    if df.empty:
        return "<p>No Granger results available (insufficient data).</p>"

    rows_html = ""
    for _, r in df.iterrows():
        sig_cls = "sig-yes" if r.get("forward_sig") == "Yes" else "sig-no"
        rows_html += (
            f"<tr><td>{r['target']}</td>"
            f"<td>{r['best_lag_days']}</td>"
            f"<td class='{sig_cls}'>{r['forward_p']:.4f}</td>"
            f"<td>{r['reverse_p']:.4f}</td>"
            f"<td>{r['direction']}</td></tr>"
        )
    return f"""
    <table>
      <thead><tr>
        <th>Target variable</th><th>Best lag (days)</th>
        <th>p (prob→var)</th><th>p (var→prob)</th><th>Interpretation</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _event_table(events_result: dict) -> str:
    agg = events_result.get("aggregate_table", pd.DataFrame())
    if agg.empty:
        return "<p>No event study results (no sharp movements detected).</p>"

    rows_html = ""
    for _, r in agg.iterrows():
        chg = r.get("mean_change", 0) or 0
        chg_cls = "sig-yes" if chg > 0 else ("sig-no" if chg == 0 else "")
        rows_html += (
            f"<tr><td>{r['variable']}</td>"
            f"<td>{r['n_events']}</td>"
            f"<td>{r['mean_pre']:.4f}</td>"
            f"<td>{r['mean_post']:.4f}</td>"
            f"<td class='{chg_cls}'>{chg:+.4f}</td>"
            f"<td>{r['pct_positive']:.0f}%</td></tr>"
        )

    events = events_result.get("events", [])
    event_rows = ""
    for e in events:
        event_rows += (
            f"<tr><td>{e['date']}</td>"
            f"<td>{e['direction']}</td>"
            f"<td>{e['delta']:+.3f}</td>"
            f"<td>{e['probability']:.3f}</td></tr>"
        )

    return f"""
    <h3>Aggregate: mean pre/post change across all shock events</h3>
    <table>
      <thead><tr>
        <th>Variable</th><th>N events</th><th>Mean pre</th>
        <th>Mean post</th><th>Mean change</th><th>% positive</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    <h3>Individual shock events</h3>
    <table>
      <thead><tr>
        <th>Date</th><th>Direction</th><th>Δ probability</th><th>Probability at shock</th>
      </tr></thead>
      <tbody>{event_rows}</tbody>
    </table>"""


def generate_report(
    aligned_df:       pd.DataFrame,
    shocks_df:        pd.DataFrame,
    analysis_results: dict,
    market_question:  str,
    gdelt_query:      str,
    start:            str,
    end:              str,
    features_daily:   "pd.DataFrame | None" = None,
) -> str:
    """
    Generate a self-contained HTML report string.

    Parameters
    ----------
    aligned_df        : Merged Polymarket + GDELT time series.
    shocks_df         : Sharp movement events.
    analysis_results  : Output of analysis.run_full_analysis().
    market_question   : The Polymarket market question string.
    gdelt_query       : The GDELT query used.
    start / end       : Analysis window date strings.
    features_daily    : Daily narrative features (optional).
    """
    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # ── Summary stats ──
    n_days     = len(aligned_df)
    prob_min   = aligned_df["probability"].min()
    prob_max   = aligned_df["probability"].max()
    prob_mean  = aligned_df["probability"].mean()
    n_shocks   = len(shocks_df)
    coverage_col = "volume_intensity" if "volume_intensity" in aligned_df.columns else "volume_norm"
    peak_cov   = aligned_df[coverage_col].max() if coverage_col in aligned_df.columns else 0

    granger     = analysis_results.get("granger", {})
    xcorr       = analysis_results.get("xcorr",   {})
    events_res  = analysis_results.get("events",  {})

    xcorr_peak_lag  = xcorr.get("peak_lag",  "–")
    xcorr_peak_corr = xcorr.get("peak_corr", "–")

    # ── Charts ──
    chart_main   = chart_probability_and_coverage(aligned_df, shocks_df, market_question)
    chart_xcorr  = chart_cross_correlation(xcorr) if xcorr.get("lags") else ""
    chart_feat   = chart_narrative_features(features_daily) if features_daily is not None and not features_daily.empty else ""

    # ── ADF results ──
    adf_rows = ""
    for k, v in granger.get("stationarity", {}).items():
        if isinstance(v, dict) and "p_value" in v:
            sig = "✓" if v["is_stationary"] else "✗"
            adf_rows += (f"<tr><td>{v['name']}</td>"
                         f"<td>{v['adf_stat']:.3f}</td>"
                         f"<td>{v['p_value']:.4f}</td>"
                         f"<td>{sig} {'Stationary' if v['is_stationary'] else 'Non-stationary'}</td></tr>")
    adf_table = f"""
    <table>
      <thead><tr><th>Series</th><th>ADF statistic</th><th>p-value</th><th>Result</th></tr></thead>
      <tbody>{adf_rows}</tbody>
    </table>""" if adf_rows else "<p>ADF results unavailable.</p>"

    granger_interp = granger.get("interpretation", "")
    xcorr_interp   = xcorr.get("interpretation", "")
    events_interp  = events_res.get("interpretation", "")

    sec_offset = 3 if chart_feat else 2

    xcorr_lag_display = (
        f"{xcorr_peak_lag}d / r={xcorr_peak_corr:.3f}"
        if isinstance(xcorr_peak_lag, int) else "–"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Narrative Impact Tracker — Report</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page">

  <h1>Narrative Impact Tracker</h1>
  <div class="meta">
    Generated {generated_at} &nbsp;|&nbsp;
    <span class="badge">narrative-impact-tracker v0.1.0</span>
    <span class="badge">Polymarket CLOB API</span>
    <span class="badge">GDELT v2 Doc API</span>
  </div>

  <div class="section">
    <h2>Market</h2>
    <p><strong>{market_question}</strong></p>
    <p>Analysis window: <strong>{start}</strong> → <strong>{end}</strong></p>
    <p>GDELT query: <code>{gdelt_query}</code></p>
  </div>

  <div class="section">
    <h2>Summary Statistics</h2>
    <div class="stat-grid">
      {_stat_box(f"{n_days}", "Days in window")}
      {_stat_box(f"{prob_min:.1%} – {prob_max:.1%}", "Probability range")}
      {_stat_box(f"{prob_mean:.1%}", "Mean probability")}
      {_stat_box(f"{n_shocks}", "Sharp movements (≥8pp/2d)")}
      {_stat_box(f"{peak_cov:.4f}", "Peak GDELT coverage intensity")}
      {_stat_box(xcorr_lag_display, "Peak cross-correlation lag")}
    </div>
  </div>

  <div class="section">
    <h2>1 — Probability &amp; Coverage Timeline</h2>
    <p>Daily YES-outcome probability from Polymarket CLOB API (blue) and GDELT prediction-market
    coverage intensity (red). Vertical dotted lines mark sharp probability movements
    (≥8 percentage points over 2 days).</p>
    <img class="chart" src="data:image/png;base64,{chart_main}" alt="Probability and coverage chart">
    <div class="note">
      <strong>Methodological note:</strong> GDELT's <em>TimelineVol</em> endpoint returns a
      relative volume intensity metric (share of global GDELT coverage), not raw article counts.
      Values are suitable for detecting relative shifts over time but should not be interpreted
      as absolute coverage totals.
    </div>
  </div>

  {"" if not chart_feat else f'''
  <div class="section">
    <h2>2 — Narrative Feature Time Series</h2>
    <p>Headline-level extraction of three core narrative variables. Note: headline-only
    analysis underestimates absolute feature values relative to full-text analysis.
    Time series shape is preserved; treat as relative indicator.</p>
    <img class="chart" src="data:image/png;base64,{chart_feat}" alt="Narrative features chart">
    <div class="note">
      <strong>ERS</strong> (Epistemic Register Score): positive values indicate high-certainty
      framing; negative values indicate hedged language. <strong>PCF adoption rate</strong>:
      proportion of articles containing explicit probability citations.
      <strong>NCS</strong>: density of narrative-closure language.
    </div>
  </div>
  '''}

  <div class="section">
    <h2>{sec_offset} — Cross-Correlation Analysis</h2>
    <p>Cross-correlation between Polymarket probability and GDELT coverage intensity at lags
    −10 to +10 days.
    Positive lags indicate probability leads coverage; negative lags indicate coverage leads
    probability.</p>
    {"" if not chart_xcorr else f'<img class="chart" src="data:image/png;base64,{chart_xcorr}" alt="Cross-correlation chart">'}
    <p><em>{xcorr_interp}</em></p>
    <div class="note">
      <strong>Interpretation caution:</strong> Cross-correlation detects co-movement, not
      causality. Both series may be responding to the same underlying events with different
      lag structures. See event study below for a stronger identification strategy.
    </div>
  </div>

  <div class="section">
    <h2>{sec_offset + 1} — Granger Causality Tests</h2>
    <p>Tests whether lagged probability movements have predictive power over narrative
    variables above and beyond the narrative series' own lags. Granger causality establishes
    <em>temporal precedence</em> — a necessary but not sufficient condition for causal
    influence. All non-stationary series were first-differenced before testing.</p>

    <h3>Stationarity (Augmented Dickey-Fuller)</h3>
    {adf_table}

    <h3>Granger test results</h3>
    {_granger_table(granger)}
    <p><em>{granger_interp}</em></p>

    <div class="note">
      <strong>Granger causality ≠ structural causality.</strong> A significant result means
      lagged probability movements improve prediction of the narrative variable — but both
      series may be jointly driven by a common third variable (e.g., a debate, major news
      event) with different response lag structures. Complement with the event study below and
      qualitative case analysis of specific narrative episodes.
    </div>
  </div>

  <div class="section">
    <h2>{sec_offset + 2} — Event Study</h2>
    <p>Pre/post analysis of narrative variables around sharp probability movements
    (≥8 percentage points over 2 days). Events preceded by identifiable scheduled
    events should be excluded in published analysis to improve identification.</p>
    {_event_table(events_res)}
    <p><em>{events_interp}</em></p>
    <div class="note">
      <strong>Next step for publication:</strong> Manually code each shock event as
      "clean" (no major scheduled event within ±72 hours) or "confounded". Report
      results separately for clean vs. all events. This is the key robustness check
      reviewers will request.
    </div>
  </div>

  <div class="section">
    <h2>Methodology Notes</h2>
    <ul>
      <li><strong>Data sources:</strong> Polymarket CLOB API (probability history, public);
          GDELT v2 Doc API TimelineVol and ArtList (coverage data, public).</li>
      <li><strong>Narrative variables:</strong> Extracted at headline level only (GDELT does
          not provide full article text via the free API). Absolute values are suppressed;
          time-series patterns are preserved. Validate against human-coded sample before
          using ERS/PCF/NCS as primary outcomes.</li>
      <li><strong>GDELT coverage bias:</strong> Skews toward English-language wire services
          and major outlets. Under-represents Substack, financial media, and non-English
          coverage where Polymarket influence may be concentrated.</li>
      <li><strong>CLOB API limitation:</strong> The prices-history endpoint does not accept
          arbitrary date ranges; this tool uses <code>interval=all</code> with client-side
          filtering. Market-level granularity is limited to active CLOB markets.</li>
      <li><strong>Granger tests:</strong> Conducted on first-differenced series after ADF
          stationarity testing. Maximum lag = 5 days. F-test p-values reported.</li>
      <li><strong>Event study:</strong> ±5-day pre/post window. No correction for
          confounding events applied in this automated output — manual event coding required
          for publication-quality results.</li>
    </ul>
  </div>

  <footer>
    Generated by <strong>narrative-impact-tracker v0.1.0</strong>
    &nbsp;|&nbsp; {generated_at}
    &nbsp;|&nbsp; Designed for use as a methods paper supplement.
    Cite as: [Author] (2025). Narrative Impact Tracker [Software].
  </footer>

</div>
</body>
</html>"""

    return html


def save_report(html: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report saved: {path}")

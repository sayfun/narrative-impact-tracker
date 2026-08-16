"""
analysis.py
-----------
Statistical analysis layer for the Narrative Impact Tracker.

Implements two complementary identification strategies:

  1. Granger Causality Tests
     Tests whether lagged probability movements have predictive power
     over narrative variable time series, above and beyond the narrative
     series' own lags.  This establishes temporal precedence — a necessary
     but not sufficient condition for causal influence.

     Methodological honesty: Granger causality is NOT causal in the
     structural sense.  Both series may be responding to a common
     third variable (e.g., a debate) with different lag structures.
     Always report this limitation explicitly.

  2. Event Study Analysis
     Identifies 'clean shocks' — sharp probability movements not
     preceded by a major scheduled event — and measures pre/post
     changes in narrative variables around these shocks.
     This gets closer to causal identification.

  3. Cross-Correlation Analysis
     Computes the correlation between probability series and narrative
     series at multiple lags, identifying the lag at which correlation
     peaks.  Useful for characterising the narrativisation delay.

All functions return structured dicts that the report generator can
render directly.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, ccf
from statsmodels.tsa.api import VAR
from typing import Optional


# ── stationarity ──────────────────────────────────────────────────────────────

# Narrative variables in reporting priority order. Article-level indices come
# first so that headline results describe them rather than the coarser
# coverage-volume proxies, which are often zero for niche queries.
NARRATIVE_VARS = [
    "mean_ers",
    "mean_pcf",
    "mean_ncs",
    "mean_pii_proxy",
    "eai",
    "mean_tone",
    "volume_norm",         # market-mentioning GDELT stream
    "volume_norm_topic",   # topic-only GDELT stream (no market clause)
    "rolling_3d_vol",
    "rolling_3d_vol_topic",
]

# Human-readable labels for Granger table display
NARRATIVE_VAR_LABELS = {
    "volume_norm":          "Coverage: market-mentioning",
    "volume_norm_topic":    "Coverage: topic-only",
    "rolling_3d_vol":       "Coverage 3d MA: market-mentioning",
    "rolling_3d_vol_topic": "Coverage 3d MA: topic-only",
    "mean_tone":            "Tone (GDELT)",
    "mean_ers":             "Epistemic Register Score",
    "mean_pcf":             "Probability Citation Freq.",
    "mean_ncs":             "Narrative Closure Score",
    "mean_pii_proxy":       "PII proxy",
    "eai":                  "Epistemic Authority Index",
}


def _degenerate_xcorr(max_lag: int, reason: str) -> dict:
    """
    Uniform empty return for cross-correlation on unusable input.

    `lags` and `correlations` are deliberately empty rather than zero-filled.
    Both app.py and report.py gate chart rendering on `if xcorr.get("lags")`,
    then format `peak_corr` with `:.3f`; returning populated lags alongside a
    null peak would pass that guard and raise on the format. Empty lists make
    the existing guards do the right thing.
    """
    return {
        "lags":         [],
        "correlations": [],
        "peak_lag":     None,
        "peak_corr":    None,
        "target":       None,
        "n":            0,
        "interpretation": f"Cross-correlation not computed: {reason}.",
    }


def adf_test(series: pd.Series, name: str = "series") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns dict with: name, adf_stat, p_value, is_stationary, lags_used,
    interpretation.

    Series must be differenced if non-stationary before Granger tests.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return {"name": name, "error": "insufficient observations", "is_stationary": None}

    # Constant series (zero variance) → ADF test is undefined; treat as non-stationary
    if clean.std() == 0:
        return {
            "name": name, "error": "constant series (zero variance)",
            "is_stationary": None,
            "interpretation": f"{name}: constant series — ADF test not applicable",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = adfuller(clean, autolag="AIC")
    except Exception as e:
        return {
            "name": name, "error": str(e),
            "is_stationary": None,
            "interpretation": f"{name}: ADF test failed ({e})",
        }

    is_stationary = result[1] < 0.05
    return {
        "name":           name,
        "adf_stat":       round(float(result[0]), 4),
        "p_value":        round(float(result[1]), 4),
        "lags_used":      result[2],
        "n_obs":          result[3],
        "is_stationary":  is_stationary,
        "interpretation": (
            f"Stationary (p={result[1]:.3f}) — suitable for Granger tests"
            if is_stationary
            else f"Non-stationary (p={result[1]:.3f}) — difference before Granger tests"
        ),
    }


def prepare_for_granger(
    aligned_df: pd.DataFrame,
    prob_col:   str = "probability",
    cov_col:    str = "volume_norm",
    keep_cols:  Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Prepare the aligned DataFrame for Granger testing.

    Applies first differencing and returns the prepared DataFrame plus a
    dict of ADF test results.

    Parameters
    ----------
    keep_cols : Narrative variable columns that must survive into the
                prepared frame. Callers that intend to test article-level
                indices (mean_ers, mean_pcf, ...) MUST pass them here.

    Returns (prepared_df, stationarity_report)

    Note
    ----
    Rows are dropped only where the probability series is missing. Narrative
    columns are retained with their NaNs intact so that a variable which is
    sparse on some days does not shrink the usable sample for every other
    variable. Per-variable dropna happens at test time.
    """
    targets = list(keep_cols) if keep_cols else []
    if cov_col and cov_col not in targets:
        targets.append(cov_col)

    cols = ["date", prob_col] + [c for c in targets if c in aligned_df.columns]
    # de-duplicate while preserving order
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    df = aligned_df[ordered].copy()
    df = df.dropna(subset=[prob_col]).sort_values("date").reset_index(drop=True)

    stationarity = {}
    stationarity[f"{prob_col}_levels"] = adf_test(df[prob_col], "probability (levels)")
    df[f"d_{prob_col}"] = df[prob_col].diff()
    stationarity[f"{prob_col}_diff"] = adf_test(
        df[f"d_{prob_col}"].dropna(), "Δprobability (1st diff)")

    for col in ordered:
        if col in ("date", prob_col):
            continue
        stationarity[f"{col}_levels"] = adf_test(df[col].dropna(), f"{col} (levels)")
        df[f"d_{col}"] = df[col].diff()
        stationarity[f"{col}_diff"] = adf_test(df[f"d_{col}"].dropna(), f"Δ{col}")

    return df.reset_index(drop=True), stationarity


# ── granger causality ─────────────────────────────────────────────────────────

def run_granger_tests(
    aligned_df:   pd.DataFrame,
    prob_col:     str = "probability",
    target_cols:  Optional[list[str]] = None,
    max_lag:      int = 5,
    use_diff:     bool = True,
) -> dict:
    """
    Run Granger causality tests: does probability predict narrative variables?

    Tests both directions:
      H1: probability movements → narrative variable shifts (main hypothesis)
      H2: narrative variable shifts → probability movements (reverse causality check)

    Parameters
    ----------
    aligned_df   : The aligned DataFrame from NarrativePipeline.
    prob_col     : Column name for the probability series.
    target_cols  : Narrative variable columns to test. Defaults to
                   volume_norm, mean_tone, mean_ers, mean_pcf, mean_ncs
                   (whichever are present).
    max_lag      : Maximum lag order to test (days).
    use_diff     : If True, first-difference non-stationary series.

    Returns
    -------
    dict with keys:
        stationarity       : ADF test results for all series
        results            : dict of {target_col: {lag: {forward, reverse}}}
        summary_table      : pd.DataFrame of best-lag results
        interpretation     : human-readable summary string
    """
    if target_cols is None:
        target_cols = [c for c in NARRATIVE_VARS if c in aligned_df.columns]

    # Retain every requested target through preparation. Passing target_cols
    # here is essential: without it the frame is subset to the coverage column
    # alone and all article-level indices are silently dropped before testing.
    df, stationarity = prepare_for_granger(
        aligned_df, prob_col, keep_cols=target_cols)

    # ADF-conditioned differencing: only difference a series if ADF finds it
    # non-stationary (or inconclusive). Differencing a stationary series
    # unnecessarily introduces moving-average structure and reduces power.
    def _needs_diff(col_name: str) -> bool:
        result = stationarity.get(f"{col_name}_levels", {})
        is_stat = result.get("is_stationary")
        return is_stat is not True  # non-stationary or unknown → difference

    prob_series = f"d_{prob_col}" if (use_diff and _needs_diff(prob_col)) else prob_col

    results = {}
    summary_rows = []

    for target in target_cols:
        if target not in df.columns:
            continue

        d_target = f"d_{target}"
        if d_target not in df.columns:
            df[d_target] = df[target].diff()

        target_series = d_target if (use_diff and _needs_diff(target)) else target

        clean = df[[prob_series, target_series]].dropna()
        if len(clean) < max_lag * 3 + 5:
            results[target] = {
                "error": f"insufficient observations after differencing "
                         f"(n={len(clean)}, need >= {max_lag * 3 + 5})"
            }
            continue
        if clean[target_series].nunique() <= 1 or clean[prob_series].nunique() <= 1:
            results[target] = {
                "error": "constant series after differencing; Granger test undefined"
            }
            continue

        lag_results = {}

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Direction: prob → target (main hypothesis)
                fwd_data = clean[[target_series, prob_series]].values
                fwd = grangercausalitytests(fwd_data, maxlag=max_lag, verbose=False)

                # Direction: target → prob (reverse causality)
                rev_data = clean[[prob_series, target_series]].values
                rev = grangercausalitytests(rev_data, maxlag=max_lag, verbose=False)

            for lag in range(1, max_lag + 1):
                fwd_p = fwd[lag][0]["ssr_ftest"][1]
                rev_p = rev[lag][0]["ssr_ftest"][1]
                lag_results[lag] = {
                    "forward_p":  round(float(fwd_p), 4),  # prob → target
                    "reverse_p":  round(float(rev_p), 4),  # target → prob
                    "forward_sig": fwd_p < 0.05,
                    "reverse_sig": rev_p < 0.05,
                }

            # Best lag = smallest forward p-value across lag_results.
            # Apply per-variable Bonferroni correction (multiply by number of lags
            # tested) to account for multiple comparisons across lag orders.
            best_lag        = min(lag_results, key=lambda k: lag_results[k]["forward_p"])
            best_fwd_p_raw  = lag_results[best_lag]["forward_p"]
            best_fwd_p      = min(best_fwd_p_raw * max_lag, 1.0)   # Bonferroni corrected
            best_rev_p      = lag_results[best_lag]["reverse_p"]

            results[target] = {"lags": lag_results, "best_lag": best_lag}

            # Interpretation flag (uses Bonferroni-corrected p for forward direction)
            if best_fwd_p < 0.05 and best_rev_p >= 0.05:
                flag = "✓ Unidirectional: prob→narrative"
            elif best_fwd_p < 0.05 and best_rev_p < 0.05:
                flag = "⚠ Bidirectional (common driver likely)"
            elif best_fwd_p >= 0.05 and best_rev_p < 0.05:
                flag = "↩ Reverse only: narrative→prob"
            else:
                flag = "– No significant Granger relationship"

            summary_rows.append({
                "target":          target,
                "best_lag_days":   best_lag,
                "forward_p_raw":   best_fwd_p_raw,
                "forward_p":       best_fwd_p,        # Bonferroni corrected
                "reverse_p":       best_rev_p,
                "forward_sig":     "Yes" if best_fwd_p < 0.05 else "No",
                "direction":       flag,
            })

        except Exception as e:
            results[target] = {"error": str(e)}

    summary_table = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    sig_forward = [r for r in summary_rows if r.get("forward_sig") == "Yes"]
    interpretation = (
        f"Granger tests ({max_lag}-lag, {'differenced' if use_diff else 'levels'}):\n"
        f"  {len(sig_forward)}/{len(summary_rows)} narrative variables show significant "
        f"forward Granger causality from probability movements.\n"
        + (
            "  Strongest: " + ", ".join(
                f"{r['target']} (lag {r['best_lag_days']}d, p={r['forward_p']:.3f})"
                for r in sorted(sig_forward, key=lambda x: x["forward_p"])[:3]
            )
            if sig_forward else "  No significant forward relationships detected."
        )
    )

    return {
        "stationarity":   stationarity,
        "results":        results,
        "summary_table":  summary_table,
        "interpretation": interpretation,
    }


# ── cross-correlation ─────────────────────────────────────────────────────────

def cross_correlation_analysis(
    aligned_df: pd.DataFrame,
    prob_col:   str = "probability",
    target_col: str = "volume_norm",
    max_lag:    int = 10,
) -> dict:
    """
    Compute cross-correlations between probability and a narrative variable
    at multiple lags.

    Positive lag = probability leads narrative (prediction market → media)
    Negative lag = narrative leads probability (media → prediction market)

    Returns dict with lags, correlations, peak_lag, peak_correlation.

    Note
    ----
    `target_col` defaults to the coverage-volume column for backward
    compatibility. To correlate probability against an article-level index,
    pass it explicitly, or use `cross_correlation_all()` to sweep every
    available narrative variable.
    """
    if target_col not in aligned_df.columns:
        return _degenerate_xcorr(max_lag, f"'{target_col}' not present in frame")

    df = aligned_df[["date", prob_col, target_col]].dropna().sort_values("date")

    if len(df) < 3:
        return _degenerate_xcorr(max_lag, f"insufficient observations (n={len(df)})")
    if df[target_col].nunique() <= 1:
        return _degenerate_xcorr(
            max_lag, f"'{target_col}' is constant; correlation undefined")
    if df[prob_col].nunique() <= 1:
        return _degenerate_xcorr(
            max_lag, f"'{prob_col}' is constant; correlation undefined")

    p = df[prob_col].values
    t = df[target_col].values

    # Normalise both series
    p = (p - p.mean()) / (p.std() + 1e-10)
    t = (t - t.mean()) / (t.std() + 1e-10)

    lags   = list(range(-max_lag, max_lag + 1))
    corrs  = []
    for lag in lags:
        if lag >= 0:
            if len(p) > lag:
                c = np.corrcoef(p[:len(p)-lag], t[lag:])[0, 1] if lag > 0 else np.corrcoef(p, t)[0, 1]
            else:
                c = np.nan
        else:
            abs_lag = abs(lag)
            c = np.corrcoef(p[abs_lag:], t[:len(t)-abs_lag])[0, 1] if abs_lag < len(p) else np.nan
        corrs.append(round(float(c) if not np.isnan(c) else 0.0, 4))

    peak_idx  = int(np.argmax(np.abs(corrs)))
    peak_lag  = lags[peak_idx]
    peak_corr = corrs[peak_idx]

    return {
        "lags":         lags,
        "correlations": corrs,
        "peak_lag":     peak_lag,
        "peak_corr":    peak_corr,
        "target":       target_col,
        "n":            len(df),
        "interpretation": (
            f"Peak cross-correlation r={peak_corr:.3f} at lag={peak_lag} days "
            f"({target_col}, n={len(df)}). "
            + (
                f"Probability leads {target_col} by {peak_lag} days "
                "(consistent with market-to-narrative influence)."
                if peak_lag > 0
                else f"{target_col} leads probability by {abs(peak_lag)} days "
                "(consistent with narrative-to-market influence)."
                if peak_lag < 0
                else "Peak correlation at lag 0 (contemporaneous)."
            )
        ),
    }


def cross_correlation_all(
    aligned_df: pd.DataFrame,
    prob_col:   str = "probability",
    target_cols: Optional[list[str]] = None,
    max_lag:    int = 10,
) -> dict:
    """
    Run cross_correlation_analysis() for every available narrative variable.

    Returns {target_col: xcorr_dict}. Degenerate variables (absent, constant,
    or too short) are included with a null peak and an explanatory message,
    so callers can distinguish "not computed" from "computed, no signal".
    """
    if target_cols is None:
        target_cols = [c for c in NARRATIVE_VARS if c in aligned_df.columns]
    return {
        col: cross_correlation_analysis(aligned_df, prob_col, col, max_lag)
        for col in target_cols
    }


def select_primary_target(aligned_df: pd.DataFrame) -> Optional[str]:
    """
    Choose the narrative variable that headline results should report on.

    Prefers article-level indices over coverage-volume proxies, and skips any
    column that is absent, all-NaN or constant. Returns None when the frame
    carries no usable narrative variable at all.
    """
    for col in NARRATIVE_VARS:
        if col not in aligned_df.columns:
            continue
        s = aligned_df[col].dropna()
        if len(s) >= 3 and s.nunique() > 1:
            return col
    return None


# ── event study ───────────────────────────────────────────────────────────────

def event_study(
    aligned_df:  pd.DataFrame,
    shocks_df:   pd.DataFrame,
    target_cols: Optional[list[str]] = None,
    pre_window:  int = 5,
    post_window: int = 5,
) -> dict:
    """
    Event study analysis around sharp probability movements (shocks).

    For each shock day, computes the mean value of each narrative variable
    in the pre-window and post-window, then calculates the change.

    Parameters
    ----------
    aligned_df  : Full aligned DataFrame.
    shocks_df   : Output of detect_sharp_movements().
    target_cols : Narrative variable columns to analyse.
    pre_window  : Days before shock to average.
    post_window : Days after shock to average.

    Returns
    -------
    dict with:
        events          : list of per-event results
        aggregate_table : DataFrame of mean pre/post/change per variable
        interpretation  : human-readable summary
    """
    if target_cols is None:
        candidates = ["volume_norm", "mean_tone", "rolling_3d_vol",
                      "mean_ers", "mean_pcf", "mean_ncs"]
        target_cols = [c for c in candidates if c in aligned_df.columns]

    if shocks_df.empty or not target_cols:
        return {"events": [], "aggregate_table": pd.DataFrame(),
                "interpretation": "No shock events or no narrative variables available."}

    df = aligned_df.copy().sort_values("date").reset_index(drop=True)
    df["date_norm"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.normalize()

    events = []

    for _, shock in shocks_df.iterrows():
        shock_date = pd.Timestamp(shock["date"]).normalize()
        shock_idx  = df[df["date_norm"] == shock_date].index

        if shock_idx.empty:
            continue
        i = shock_idx[0]

        pre_slice  = df.iloc[max(0, i - pre_window):i]
        post_slice = df.iloc[i + 1: min(len(df), i + post_window + 1)]

        event = {
            "date":        shock_date.date().isoformat(),
            "probability": round(float(shock["probability"]), 4),
            "delta":       round(float(shock["delta"]), 4),
            "direction":   shock["direction"],
            "variables":   {},
        }

        for col in target_cols:
            pre_mean  = float(pre_slice[col].mean())  if not pre_slice[col].isna().all()  else np.nan
            post_mean = float(post_slice[col].mean()) if not post_slice[col].isna().all() else np.nan
            change    = post_mean - pre_mean if not (np.isnan(pre_mean) or np.isnan(post_mean)) else np.nan

            event["variables"][col] = {
                "pre_mean":  round(pre_mean, 4)  if not np.isnan(pre_mean)  else None,
                "post_mean": round(post_mean, 4) if not np.isnan(post_mean) else None,
                "change":    round(change, 4)    if not np.isnan(change)    else None,
            }

        events.append(event)

    # Aggregate across all events
    agg_rows = []
    for col in target_cols:
        changes   = [e["variables"][col]["change"]   for e in events
                     if col in e["variables"] and e["variables"][col]["change"] is not None]
        pre_means = [e["variables"][col]["pre_mean"]  for e in events
                     if col in e["variables"] and e["variables"][col]["pre_mean"] is not None]
        post_means= [e["variables"][col]["post_mean"] for e in events
                     if col in e["variables"] and e["variables"][col]["post_mean"] is not None]

        if not changes:
            continue

        agg_rows.append({
            "variable":        col,
            "n_events":        len(changes),
            "mean_pre":        round(np.mean(pre_means), 4)  if pre_means  else None,
            "mean_post":       round(np.mean(post_means), 4) if post_means else None,
            "mean_change":     round(np.mean(changes), 4),
            "pct_positive":    round(sum(c > 0 for c in changes) / len(changes) * 100, 1),
        })

    agg_table = pd.DataFrame(agg_rows)

    n_events = len(events)
    pos_vol  = sum(1 for e in events
                   if "volume_norm" in e["variables"]
                   and (e["variables"]["volume_norm"]["change"] or 0) > 0)

    interpretation = (
        f"Event study: {n_events} shock events (≥{shocks_df['delta'].abs().min():.0%} "
        f"movement, ±{pre_window}/{post_window}-day window).\n"
        + (f"  Coverage volume increased post-shock in {pos_vol}/{n_events} events."
           if n_events > 0 else "")
    )

    return {
        "events":          events,
        "aggregate_table": agg_table,
        "interpretation":  interpretation,
    }


# ── convenience wrapper ────────────────────────────────────────────────────────

def run_full_analysis(
    aligned_df: pd.DataFrame,
    shocks_df:  pd.DataFrame,
    verbose:    bool = True,
) -> dict:
    """
    Run all analyses on an aligned DataFrame and return a results bundle
    ready for the report generator.

    Each sub-analysis is wrapped in a try/except so that degenerate data
    (e.g. constant series, too few observations) in one test never prevents
    the others from running.
    """
    _empty_granger = {
        "stationarity": {}, "results": {}, "summary_table": pd.DataFrame(),
        "interpretation": "Granger tests skipped (data insufficient or constant series).",
    }
    _empty_xcorr = {
        "lags": [], "correlations": [], "peak_lag": None, "peak_corr": None,
        "interpretation": "Cross-correlation skipped (insufficient data).",
    }
    _empty_events = {
        "events": [], "aggregate_table": pd.DataFrame(),
        "interpretation": "Event study skipped (no shock events or insufficient data).",
    }

    if verbose:
        print("Running statistical analysis …")
        print("  → Granger causality tests")

    try:
        granger = run_granger_tests(aligned_df, max_lag=5)
    except Exception as e:
        granger = {**_empty_granger,
                   "interpretation": f"Granger tests failed: {e}"}

    if verbose:
        print("  → Cross-correlation analysis")
    primary = None
    try:
        # Sweep every narrative variable, then surface the highest-priority
        # usable one as the flat `xcorr` result for backward compatibility.
        xcorr_all = cross_correlation_all(aligned_df)
        primary = select_primary_target(aligned_df)
        if primary and primary in xcorr_all:
            xcorr = xcorr_all[primary]
        else:
            xcorr = {**_empty_xcorr,
                     "interpretation": "Cross-correlation skipped "
                                       "(no usable narrative variable in frame)."}
    except Exception as e:
        xcorr_all = {}
        xcorr = {**_empty_xcorr,
                 "interpretation": f"Cross-correlation failed: {e}"}

    if verbose:
        print("  → Event study")
    try:
        events = event_study(aligned_df, shocks_df)
    except Exception as e:
        events = {**_empty_events,
                  "interpretation": f"Event study failed: {e}"}

    if verbose:
        print(f"\n{granger['interpretation']}")
        print(f"\n{xcorr['interpretation']}")
        print(f"\n{events['interpretation']}")

    return {
        "granger":        granger,
        "xcorr":          xcorr,       # primary variable, flat shape (legacy)
        "xcorr_all":      xcorr_all,   # {variable: xcorr_dict}
        "primary_target": primary,     # which variable `xcorr` describes
        "events":         events,
    }

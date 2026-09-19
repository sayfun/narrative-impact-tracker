"""
Regression tests for the analysis-layer wiring defect.

The defect: prepare_for_granger() subset the aligned frame to
["date", prob_col, cov_col] before run_granger_tests() iterated its target
list, so every article-level index (mean_ers, mean_pcf, mean_ncs, ...) was
dropped before it could be tested. cross_correlation_analysis() had the same
problem via a hard-coded target_col="volume_norm" default.

Symptom: when the coverage-volume column was constant (common for niche
queries, where GDELT returns nothing), Granger reported "0/0 variables
tested" and cross-correlation reported r=0.000, while the article-level
indices sat unused in the frame.

These tests fail against the pre-fix implementation.
"""
import numpy as np
import pandas as pd
import pytest

from narrative_tracker.analysis import (
    NARRATIVE_VARS,
    cross_correlation_all,
    prepare_for_granger,
    run_full_analysis,
    run_granger_tests,
    select_primary_target,
)


@pytest.fixture
def frame():
    """
    60 days of synthetic data reproducing the failure conditions:
      - volume_norm constant at zero (the degenerate coverage column)
      - mean_ers genuinely driven by probability at a 2-day lag
      - mean_pcf noise
      - mean_ncs constant zero (the NCS domain-failure case)
    """
    rng = np.random.default_rng(11)
    n = 60
    dates = pd.date_range("2026-06-01", periods=n, freq="D", tz="UTC")

    prob = np.cumsum(rng.normal(0, 0.04, n)) + 0.5
    prob = np.clip(prob, 0.05, 0.95)

    d_prob = np.diff(prob, prepend=prob[0])
    ers = np.zeros(n)
    for t in range(2, n):
        ers[t] = 0.85 * d_prob[t - 2] + rng.normal(0, 0.02)

    return pd.DataFrame({
        "date": dates,
        "probability": prob,
        "volume_norm": np.zeros(n),          # degenerate on purpose
        "rolling_3d_vol": np.zeros(n),       # degenerate on purpose
        "mean_ers": ers,
        "mean_pcf": rng.normal(0, 0.5, n),
        "mean_ncs": np.zeros(n),             # degenerate on purpose
        "mean_pii_proxy": rng.uniform(0, 1, n),
    })


@pytest.fixture
def shocks(frame):
    d = frame["probability"].diff()
    hits = frame.loc[d.abs() >= 0.05, ["date", "probability"]].copy()
    hits["delta"] = d[d.abs() >= 0.05].values
    hits["direction"] = np.where(hits["delta"] > 0, "UP", "DOWN")
    return hits.reset_index(drop=True)


# ── prepare_for_granger ──────────────────────────────────────────────

def test_prepare_retains_requested_narrative_columns(frame):
    """The core regression: requested targets must survive preparation."""
    targets = ["mean_ers", "mean_pcf", "mean_ncs"]
    prepared, _ = prepare_for_granger(frame, "probability", keep_cols=targets)
    for col in targets:
        assert col in prepared.columns, f"{col} was dropped during preparation"
        assert f"d_{col}" in prepared.columns, f"d_{col} was not differenced"


def test_prepare_does_not_shrink_sample_via_sparse_column(frame):
    """A column that is NaN on some days must not drop those rows for all."""
    sparse = frame.copy()
    sparse.loc[sparse.index[:20], "mean_pcf"] = np.nan
    prepared, _ = prepare_for_granger(
        sparse, "probability", keep_cols=["mean_ers", "mean_pcf"])
    assert prepared["mean_ers"].notna().sum() >= 55


# ── run_granger_tests ────────────────────────────────────────────────

def test_granger_reaches_article_level_indices(frame):
    """Pre-fix this returned results for volume_norm only."""
    out = run_granger_tests(frame, max_lag=5)
    assert "mean_ers" in out["results"], "mean_ers never reached the test loop"
    assert "mean_pcf" in out["results"], "mean_pcf never reached the test loop"


def test_granger_recovers_known_lagged_relationship(frame):
    """ERS is constructed to follow probability at lag 2; it should show up."""
    out = run_granger_tests(frame, max_lag=5)
    res = out["results"]["mean_ers"]
    assert "error" not in res, res.get("error")
    best_p = res["lags"][res["best_lag"]]["forward_p"]
    assert best_p < 0.05, f"expected a significant forward result, got p={best_p}"


def test_granger_labels_constant_series_rather_than_failing_silently(frame):
    """Degenerate columns should report a reason, not vanish."""
    out = run_granger_tests(frame, max_lag=5)
    assert "error" in out["results"]["volume_norm"]
    assert "constant" in out["results"]["volume_norm"]["error"].lower()


# ── cross-correlation ────────────────────────────────────────────────

def test_cross_correlation_sweeps_all_variables(frame):
    out = cross_correlation_all(frame)
    for col in ("mean_ers", "mean_pcf", "mean_pii_proxy"):
        assert col in out
        assert out[col]["peak_lag"] is not None, f"{col} produced no peak"


def test_cross_correlation_marks_constant_target_as_uncomputed(frame):
    """Constant input must yield peak_corr=None, not a misleading r=0.000."""
    out = cross_correlation_all(frame)
    assert out["volume_norm"]["peak_corr"] is None
    assert "not computed" in out["volume_norm"]["interpretation"].lower()


def test_degenerate_xcorr_does_not_trip_downstream_render_guards(frame):
    """
    app.py and report.py both do `if xcorr.get("lags"): chart(xcorr)` and then
    format peak_corr with :.3f. A degenerate result must therefore present an
    empty `lags` list, or those consumers raise on None.
    """
    out = cross_correlation_all(frame)
    deg = out["volume_norm"]
    assert deg["lags"] == [] and deg["correlations"] == []
    assert not deg["lags"], "empty lags is what makes the downstream guard fire"


# ── primary target selection ─────────────────────────────────────────

def test_primary_target_skips_degenerate_columns(frame):
    assert select_primary_target(frame) == "mean_ers"


def test_primary_target_falls_back_when_indices_absent(frame):
    reduced = frame[["date", "probability", "volume_norm"]].copy()
    reduced["volume_norm"] = np.linspace(0, 1, len(reduced))
    assert select_primary_target(reduced) == "volume_norm"


def test_primary_target_returns_none_when_nothing_usable(frame):
    reduced = frame[["date", "probability", "volume_norm"]].copy()
    assert select_primary_target(reduced) is None


# ── run_full_analysis contract ───────────────────────────────────────

def test_full_analysis_preserves_legacy_xcorr_shape(frame, shocks):
    """app.py and report.py index these keys directly."""
    out = run_full_analysis(frame, shocks, verbose=False)
    for key in ("lags", "correlations", "peak_lag", "peak_corr", "interpretation"):
        assert key in out["xcorr"], f"legacy consumers require xcorr['{key}']"


def test_full_analysis_reports_on_an_article_level_index(frame, shocks):
    out = run_full_analysis(frame, shocks, verbose=False)
    assert out["primary_target"] == "mean_ers"
    assert out["xcorr"]["peak_corr"] is not None
    assert "xcorr_all" in out and len(out["xcorr_all"]) >= 4


def test_full_analysis_survives_frame_with_no_narrative_columns(shocks):
    bare = pd.DataFrame({
        "date": pd.date_range("2026-06-01", periods=40, freq="D", tz="UTC"),
        "probability": np.linspace(0.2, 0.8, 40),
    })
    out = run_full_analysis(bare, shocks, verbose=False)
    assert out["primary_target"] is None
    assert out["xcorr"]["peak_corr"] is None


def test_narrative_vars_prioritises_indices_over_volume():
    """Ordering is load-bearing for select_primary_target()."""
    assert NARRATIVE_VARS.index("mean_ers") < NARRATIVE_VARS.index("volume_norm")

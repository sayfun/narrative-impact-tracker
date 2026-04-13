"""
Basic smoke tests for the narrative feature extractor.
These run offline (no API calls) to verify the lexicon and scoring logic.
"""

import pytest
from narrative_tracker.features import extract_features


def test_high_certainty_raises_ers():
    text = "The candidate will win. She is the inevitable frontrunner with a commanding lead."
    result = extract_features(text)
    assert result["ers"] > 0, "High-certainty text should produce positive ERS"
    assert result["high_cert_count"] > result["low_cert_count"]


def test_hedged_language_lowers_ers():
    text = "It remains unclear who might win. The race is too close to call and could go either way."
    result = extract_features(text)
    assert result["ers"] < 0, "Hedged text should produce negative ERS"


def test_pcf_detects_polymarket():
    text = "According to Polymarket, there is a 73% chance the incumbent wins re-election."
    result = extract_features(text)
    assert result["pcf"] > 0, "Polymarket citation should raise PCF"
    assert result["pcf_binary"] == 1


def test_ncs_detects_closure():
    text = "The race is effectively over. She has no viable path to victory."
    result = extract_features(text)
    assert result["ncs"] > 0, "Closure language should raise NCS"
    assert result["ncs_binary"] == 1


def test_empty_text_returns_zeros():
    result = extract_features("")
    assert all(v == 0 for v in result.values()), "Empty text should return all zeros"


def test_neutral_text_has_mid_ers():
    text = "The candidate spoke at a rally today in Ohio."
    result = extract_features(text)
    # Neutral text: ERS should be 0 (no cert markers) or slightly negative (modal verbs)
    assert -1.0 <= result["ers"] <= 1.0

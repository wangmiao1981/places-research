"""Tests for SentimentAnalyzer."""

import json
import os
import tempfile
from unittest.mock import MagicMock, call

import pytest

from sentiment_analyzer import SentimentAnalyzer


VALID_LLM_RESPONSE = json.dumps({
    "positive_themes": [
        {"theme": "Friendly staff", "frequency": 5, "examples": ["Great people"]}
    ],
    "negative_themes": [
        {"theme": "Long wait", "frequency": 2, "examples": ["Waited 30 min"]}
    ],
    "service_quality_patterns": ["Consistent quality"],
    "unmet_needs": ["Online booking"],
    "overall_sentiment": "positive",
})


def make_analyzer(llm_response=VALID_LLM_RESPONSE):
    llm = MagicMock()
    llm.call.return_value = llm_response
    prompt = MagicMock()
    prompt.render.return_value = "rendered prompt"
    return SentimentAnalyzer(llm, prompt), llm, prompt


def make_business(name, reviews):
    return {
        "place_id": f"id_{name}",
        "name": name,
        "rating": 4.0,
        "total_ratings": len(reviews),
        "reviews": reviews,
        "address": "123 Main St",
    }


# ---------------------------------------------------------------------------
# Basic analysis
# ---------------------------------------------------------------------------

def test_businesses_with_reviews_analyzed():
    analyzer, llm, prompt = make_analyzer()
    businesses = [
        make_business("Place A", ["Great!", "Love it"]),
        make_business("Place B", ["Okay", "Fine"]),
    ]
    report = analyzer.analyze(businesses, "cafe", "Seattle")

    assert llm.call.called
    assert report["overall_sentiment"] == "positive"
    assert report["businesses_analyzed"] == 2
    assert report["businesses_without_reviews"] == 0
    assert len(report["positive_themes"]) == 1
    assert report["positive_themes"][0]["theme"] == "Friendly staff"


def test_businesses_without_reviews_skipped_and_counted():
    analyzer, llm, prompt = make_analyzer()
    businesses = [
        make_business("Has Reviews", ["Great!"]),
        make_business("No Reviews", []),
    ]
    report = analyzer.analyze(businesses, "spa", "Portland")

    assert report["businesses_analyzed"] == 1
    assert report["businesses_without_reviews"] == 1


def test_all_businesses_no_reviews_returns_empty_report():
    analyzer, llm, prompt = make_analyzer()
    businesses = [
        make_business("Empty A", []),
        make_business("Empty B", []),
    ]
    report = analyzer.analyze(businesses, "gym", "Denver")

    assert report["businesses_analyzed"] == 0
    assert report["businesses_without_reviews"] == 2
    assert report["positive_themes"] == []
    assert report["negative_themes"] == []
    assert report["service_quality_patterns"] == []
    assert report["unmet_needs"] == []
    assert report["overall_sentiment"] == "unknown"
    assert llm.call.call_count == 0


def test_empty_business_list():
    analyzer, llm, prompt = make_analyzer()
    report = analyzer.analyze([], "retail", "Austin")

    assert report["businesses_analyzed"] == 0
    assert report["businesses_without_reviews"] == 0
    assert report["positive_themes"] == []
    assert report["overall_sentiment"] == "unknown"
    assert llm.call.call_count == 0


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_batching_over_20_reviews():
    """With >20 reviews, LLM should be called multiple times (one per batch)."""
    analyzer, llm, prompt = make_analyzer()
    # 25 reviews across two businesses
    reviews_a = [f"Review {i}" for i in range(15)]
    reviews_b = [f"Review {i}" for i in range(10)]
    businesses = [
        make_business("Big Place", reviews_a),
        make_business("Small Place", reviews_b),
    ]
    report = analyzer.analyze(businesses, "restaurant", "NYC")

    # 25 reviews / 20 per batch = 2 batches → 2 LLM calls
    assert llm.call.call_count == 2
    assert report["businesses_analyzed"] == 2


def test_single_batch_result_returned_directly():
    """With <=20 reviews, exactly one LLM call is made."""
    analyzer, llm, prompt = make_analyzer()
    businesses = [make_business("Place", ["Good", "Nice"])]
    report = analyzer.analyze(businesses, "cafe", "LA")

    assert llm.call.call_count == 1
    assert report["overall_sentiment"] == "positive"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_malformed_json_handled_gracefully():
    analyzer, llm, prompt = make_analyzer(llm_response="this is not json {{{")
    businesses = [make_business("Place", ["Great!"])]

    # Should not raise; returns empty/default structure
    report = analyzer.analyze(businesses, "spa", "Miami")

    assert "positive_themes" in report
    assert "negative_themes" in report
    assert "overall_sentiment" in report


def test_llm_returns_json_with_extra_text():
    """LLM sometimes wraps JSON in markdown code fences."""
    wrapped = f"```json\n{VALID_LLM_RESPONSE}\n```"
    analyzer, llm, prompt = make_analyzer(llm_response=wrapped)
    businesses = [make_business("Place", ["Nice"])]
    report = analyzer.analyze(businesses, "cafe", "Boston")

    assert report["overall_sentiment"] == "positive"


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def test_save_report_creates_valid_json():
    analyzer, _, _ = make_analyzer()
    report = {
        "positive_themes": [],
        "negative_themes": [],
        "service_quality_patterns": [],
        "unmet_needs": [],
        "overall_sentiment": "positive",
        "businesses_analyzed": 1,
        "businesses_without_reviews": 0,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        analyzer.save_report(report, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == report
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def test_output_schema_has_all_required_keys():
    analyzer, _, _ = make_analyzer()
    businesses = [make_business("Place", ["Good"])]
    report = analyzer.analyze(businesses, "cafe", "Chicago")

    required_keys = {
        "positive_themes",
        "negative_themes",
        "service_quality_patterns",
        "unmet_needs",
        "overall_sentiment",
        "businesses_analyzed",
        "businesses_without_reviews",
    }
    assert required_keys.issubset(set(report.keys()))

"""Tests for SentimentAnalyzer."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock


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
    from sentiment_analyzer import SentimentAnalyzer
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

class TestBasicAnalysis(unittest.TestCase):
    def test_businesses_with_reviews_analyzed(self):
        analyzer, llm, prompt = make_analyzer()
        businesses = [
            make_business("Place A", ["Great!", "Love it"]),
            make_business("Place B", ["Okay", "Fine"]),
        ]
        report = analyzer.analyze(businesses, "cafe", "Seattle")

        self.assertTrue(llm.call.called)
        self.assertEqual(report["overall_sentiment"], "positive")
        self.assertEqual(report["businesses_analyzed"], 2)
        self.assertEqual(report["businesses_without_reviews"], 0)
        self.assertEqual(len(report["positive_themes"]), 1)
        self.assertEqual(report["positive_themes"][0]["theme"], "Friendly staff")

    def test_businesses_without_reviews_skipped_and_counted(self):
        analyzer, llm, prompt = make_analyzer()
        businesses = [
            make_business("Has Reviews", ["Great!"]),
            make_business("No Reviews", []),
        ]
        report = analyzer.analyze(businesses, "spa", "Portland")

        self.assertEqual(report["businesses_analyzed"], 1)
        self.assertEqual(report["businesses_without_reviews"], 1)

    def test_all_businesses_no_reviews_returns_empty_report(self):
        analyzer, llm, prompt = make_analyzer()
        businesses = [
            make_business("Empty A", []),
            make_business("Empty B", []),
        ]
        report = analyzer.analyze(businesses, "gym", "Denver")

        self.assertEqual(report["businesses_analyzed"], 0)
        self.assertEqual(report["businesses_without_reviews"], 2)
        self.assertEqual(report["positive_themes"], [])
        self.assertEqual(report["negative_themes"], [])
        self.assertEqual(report["service_quality_patterns"], [])
        self.assertEqual(report["unmet_needs"], [])
        self.assertEqual(report["overall_sentiment"], "unknown")
        self.assertEqual(llm.call.call_count, 0)

    def test_empty_business_list(self):
        analyzer, llm, prompt = make_analyzer()
        report = analyzer.analyze([], "retail", "Austin")

        self.assertEqual(report["businesses_analyzed"], 0)
        self.assertEqual(report["businesses_without_reviews"], 0)
        self.assertEqual(report["positive_themes"], [])
        self.assertEqual(report["overall_sentiment"], "unknown")
        self.assertEqual(llm.call.call_count, 0)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching(unittest.TestCase):
    def test_batching_over_20_reviews(self):
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

        # 25 reviews / 20 per batch = 2 batches -> 2 LLM calls
        self.assertEqual(llm.call.call_count, 2)
        self.assertEqual(report["businesses_analyzed"], 2)

    def test_single_batch_result_returned_directly(self):
        """With <=20 reviews, exactly one LLM call is made."""
        analyzer, llm, prompt = make_analyzer()
        businesses = [make_business("Place", ["Good", "Nice"])]
        report = analyzer.analyze(businesses, "cafe", "LA")

        self.assertEqual(llm.call.call_count, 1)
        self.assertEqual(report["overall_sentiment"], "positive")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling(unittest.TestCase):
    def test_malformed_json_handled_gracefully(self):
        analyzer, llm, prompt = make_analyzer(llm_response="this is not json {{{")
        businesses = [make_business("Place", ["Great!"])]

        # Should not raise; returns empty/default structure
        report = analyzer.analyze(businesses, "spa", "Miami")

        self.assertIn("positive_themes", report)
        self.assertIn("negative_themes", report)
        self.assertIn("overall_sentiment", report)

    def test_llm_returns_json_with_extra_text(self):
        """LLM sometimes wraps JSON in markdown code fences."""
        wrapped = "```json\n{}\n```".format(VALID_LLM_RESPONSE)
        analyzer, llm, prompt = make_analyzer(llm_response=wrapped)
        businesses = [make_business("Place", ["Nice"])]
        report = analyzer.analyze(businesses, "cafe", "Boston")

        self.assertEqual(report["overall_sentiment"], "positive")


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

class TestSaveReport(unittest.TestCase):
    def test_save_report_creates_valid_json(self):
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
            self.assertEqual(loaded, report)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema(unittest.TestCase):
    def test_output_schema_has_all_required_keys(self):
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
        self.assertTrue(required_keys.issubset(set(report.keys())))


if __name__ == "__main__":
    unittest.main()

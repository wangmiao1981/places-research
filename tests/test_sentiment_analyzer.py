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


# ---------------------------------------------------------------------------
# Merge robustness — null fields, extra keys, single vs multi batch
# ---------------------------------------------------------------------------

class TestMergeNullFields(unittest.TestCase):
    """Verify _merge_results handles null/None fields from LLM JSON."""

    def test_single_batch_null_list_fields_become_empty_lists(self):
        """Single-batch: LLM returns null for list fields (e.g. "positive_themes": null)."""
        response = json.dumps({
            "positive_themes": None,
            "negative_themes": None,
            "service_quality_patterns": None,
            "unmet_needs": None,
            "overall_sentiment": "positive",
        })
        analyzer, llm, prompt = make_analyzer(llm_response=response)
        businesses = [make_business("Place", ["Great!"])]
        report = analyzer.analyze(businesses, "spa", "LA")

        # All list fields must be lists, not None
        self.assertIsInstance(report["positive_themes"], list)
        self.assertIsInstance(report["negative_themes"], list)
        self.assertIsInstance(report["service_quality_patterns"], list)
        self.assertIsInstance(report["unmet_needs"], list)
        self.assertEqual(report["positive_themes"], [])

    def test_multi_batch_null_list_fields_become_empty_lists(self):
        """Multi-batch: LLM returns null for list fields across batches."""
        response = json.dumps({
            "positive_themes": None,
            "negative_themes": ["bad service"],
            "service_quality_patterns": None,
            "unmet_needs": None,
            "overall_sentiment": "negative",
        })
        analyzer, llm, prompt = make_analyzer(llm_response=response)
        reviews = [f"Review {i}" for i in range(25)]  # Forces 2 batches
        businesses = [make_business("Place", reviews)]
        report = analyzer.analyze(businesses, "spa", "LA")

        self.assertIsInstance(report["positive_themes"], list)
        self.assertIsInstance(report["service_quality_patterns"], list)
        self.assertIsInstance(report["unmet_needs"], list)
        # negative_themes should have items from both batches
        self.assertEqual(report["negative_themes"], ["bad service", "bad service"])

    def test_single_batch_extra_keys_not_leaked(self):
        """Single-batch: extra keys from LLM should not appear in output."""
        response = json.dumps({
            "positive_themes": ["good"],
            "negative_themes": [],
            "service_quality_patterns": [],
            "unmet_needs": [],
            "overall_sentiment": "positive",
            "secret_internal_data": "should not leak",
            "debug_info": {"model": "test"},
        })
        analyzer, llm, prompt = make_analyzer(llm_response=response)
        businesses = [make_business("Place", ["Nice"])]
        report = analyzer.analyze(businesses, "cafe", "NYC")

        # The known schema keys should be present
        self.assertIn("positive_themes", report)
        # Extra LLM keys should NOT leak through
        self.assertNotIn("secret_internal_data", report)
        self.assertNotIn("debug_info", report)

    def test_single_batch_missing_keys_get_defaults(self):
        """Single-batch: LLM omits some expected keys entirely."""
        response = json.dumps({
            "overall_sentiment": "positive",
            # Missing: positive_themes, negative_themes, etc.
        })
        analyzer, llm, prompt = make_analyzer(llm_response=response)
        businesses = [make_business("Place", ["Good"])]
        report = analyzer.analyze(businesses, "cafe", "SF")

        # All expected list keys should still be present as empty lists
        self.assertEqual(report["positive_themes"], [])
        self.assertEqual(report["negative_themes"], [])
        self.assertEqual(report["service_quality_patterns"], [])
        self.assertEqual(report["unmet_needs"], [])
        self.assertEqual(report["overall_sentiment"], "positive")

    def test_single_and_multi_batch_produce_same_schema(self):
        """Single-batch and multi-batch paths must produce identical key sets."""
        from sentiment_analyzer import _merge_results

        single_result = _merge_results([{
            "positive_themes": ["a"],
            "overall_sentiment": "positive",
        }])

        multi_result = _merge_results([
            {"positive_themes": ["a"], "overall_sentiment": "positive"},
            {"positive_themes": ["b"], "overall_sentiment": "positive"},
        ])

        self.assertEqual(set(single_result.keys()), set(multi_result.keys()),
                        "Single and multi batch paths must have same keys")


class TestMergeResultsDirectly(unittest.TestCase):
    """Unit tests for _merge_results function directly."""

    def test_empty_results(self):
        from sentiment_analyzer import _merge_results
        result = _merge_results([])
        self.assertEqual(result["positive_themes"], [])
        self.assertEqual(result["overall_sentiment"], "unknown")

    def test_single_result_with_all_nulls(self):
        from sentiment_analyzer import _merge_results
        result = _merge_results([{
            "positive_themes": None,
            "negative_themes": None,
            "service_quality_patterns": None,
            "unmet_needs": None,
            "overall_sentiment": None,
        }])
        self.assertIsInstance(result["positive_themes"], list)
        self.assertIsInstance(result["negative_themes"], list)
        self.assertEqual(result["positive_themes"], [])

    def test_mixed_sentiment_across_batches(self):
        from sentiment_analyzer import _merge_results
        result = _merge_results([
            {"positive_themes": ["a"], "overall_sentiment": "positive"},
            {"negative_themes": ["b"], "overall_sentiment": "negative"},
        ])
        self.assertEqual(result["overall_sentiment"], "mixed")
        self.assertEqual(result["positive_themes"], ["a"])
        self.assertEqual(result["negative_themes"], ["b"])

    def test_results_with_no_overall_sentiment(self):
        from sentiment_analyzer import _merge_results
        result = _merge_results([
            {"positive_themes": ["a"]},
            {"positive_themes": ["b"]},
        ])
        self.assertEqual(result["overall_sentiment"], "unknown")
        self.assertEqual(result["positive_themes"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

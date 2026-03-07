"""Integration tests for LLM-powered stages with real API calls.

These tests verify that the sentiment analyzer, web researcher, strategy
analyzer, and report generator produce valid output when connected to real
LLM and search APIs.  They are skipped automatically when API keys are not
available.

Run manually with:
    PYTHONPATH=src python3 -m pytest tests/test_integration_llm_stages.py -v -s
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
_HAS_BEARER = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
_HAS_LLM = _HAS_ANTHROPIC_KEY or _HAS_BEARER
_HAS_TAVILY = bool(os.environ.get("TAVILY_API_KEY"))

SAMPLE_BUSINESSES = [
    {
        "place_id": "p1",
        "name": "Relax Spa & Massage",
        "rating": 4.8,
        "total_ratings": 200,
        "reviews": [
            "Amazing massage! The staff was super friendly and the hot stone treatment was divine.",
            "Very clean facility, great ambiance. A bit pricey but worth it.",
            "I had a deep tissue massage and felt so much better afterwards. Highly recommend!",
            "The therapist was skilled but the waiting area could use some updates.",
            "Best massage I've ever had in San Jose. Will definitely come back.",
        ],
        "address": "100 Main St, San Jose, CA 95125",
        "website": "https://relaxspa.com",
    },
    {
        "place_id": "p2",
        "name": "Budget Bodywork",
        "rating": 3.2,
        "total_ratings": 45,
        "reviews": [
            "Cheap but you get what you pay for. The massage was okay but not great.",
            "Rude receptionist. Won't be back.",
            "It's affordable but the facility needs a deep clean.",
        ],
        "address": "200 Oak Ave, San Jose, CA 95125",
    },
    {
        "place_id": "p3",
        "name": "Healing Hands Therapy",
        "rating": 4.5,
        "total_ratings": 90,
        "reviews": [
            "Excellent sports massage. My back pain was gone after one session.",
            "Love the aromatherapy options. Very relaxing experience.",
        ],
        "address": "300 Elm Blvd, San Jose, CA 95125",
        "website": "https://healinghands.com",
    },
]


def _make_llm_client():
    """Create a real LLMClient instance."""
    from llm_client import LLMClient
    return LLMClient()


def _make_prompt_engine():
    """Create a real PromptEngine instance."""
    from prompt_engine import PromptEngine
    return PromptEngine()


# ---------------------------------------------------------------------------
# Sentiment Analyzer Integration
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LLM, "Requires ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK")
class TestSentimentAnalyzerIntegration(unittest.TestCase):
    """Integration tests for SentimentAnalyzer with real LLM."""

    def setUp(self):
        self.llm = _make_llm_client()
        self.prompts = _make_prompt_engine()

    def test_analyze_real_reviews(self):
        """Verify the analyzer produces structured output from real reviews."""
        from sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer(self.llm, self.prompts)
        result = analyzer.analyze(SAMPLE_BUSINESSES, "massage therapy", "San Jose 95125")

        # Must return a dict with required keys
        self.assertIsInstance(result, dict)
        self.assertIn("positive_themes", result)
        self.assertIn("negative_themes", result)
        self.assertIn("overall_sentiment", result)
        self.assertIn("businesses_analyzed", result)

        # Should have found some themes from the reviews
        self.assertGreater(len(result["positive_themes"]) + len(result["negative_themes"]), 0,
                          "Expected at least one theme from the reviews")

        # Businesses with reviews count
        self.assertEqual(result["businesses_analyzed"], 3)

        # overall_sentiment should be a known value
        self.assertIn(result["overall_sentiment"],
                      ["positive", "negative", "mixed", "neutral", "unknown"])

        print(f"\nSentiment result: {json.dumps(result, indent=2)}")

    def test_analyze_empty_reviews(self):
        """Verify graceful handling with no reviews."""
        from sentiment_analyzer import SentimentAnalyzer
        analyzer = SentimentAnalyzer(self.llm, self.prompts)
        businesses = [{"name": "Empty Spa", "reviews": []}]
        result = analyzer.analyze(businesses, "massage", "San Jose")
        self.assertEqual(result["businesses_analyzed"], 0)
        self.assertEqual(result["overall_sentiment"], "unknown")


# ---------------------------------------------------------------------------
# Strategy Analyzer Integration
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LLM, "Requires ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK")
class TestStrategyAnalyzerIntegration(unittest.TestCase):
    """Integration tests for StrategyAnalyzer with real LLM."""

    def setUp(self):
        self.llm = _make_llm_client()
        self.prompts = _make_prompt_engine()

    def test_analyze_produces_structured_strategy(self):
        """Verify the strategy analyzer returns structured JSON from real LLM."""
        from strategy_analyzer import StrategyAnalyzer
        analyzer = StrategyAnalyzer(self.llm, self.prompts)

        user_profile = {
            "business_type": "massage therapy",
            "planned_services": ["deep tissue", "sports massage", "aromatherapy"],
            "target_clientele": "Athletes/fitness enthusiasts",
            "budget_range": {"min": 50000, "max": 100000},
        }
        stats_report = {
            "total_businesses": 3,
            "rating_stats": {"mean": 4.17, "median": 4.5},
        }
        sentiment_report = {
            "positive_themes": ["friendly staff", "clean facilities"],
            "negative_themes": ["pricey", "rude staff"],
            "overall_sentiment": "positive",
        }
        web_research = {
            "rent_ranges": {"average": "$2500/mo"},
            "industry_trends": [{"trend": "Growing wellness market"}],
        }

        result = analyzer.analyze(
            business_type="massage therapy",
            location="San Jose 95125",
            user_profile=user_profile,
            stats_report=stats_report,
            sentiment_report=sentiment_report,
            web_research=web_research,
        )

        # Should return a non-empty dict (not {} from parse failure)
        self.assertIsInstance(result, dict)
        # If the LLM is working correctly, there should be meaningful keys
        # But we can't guarantee specific keys since it depends on the prompt
        print(f"\nStrategy result keys: {list(result.keys())}")
        print(f"Strategy result: {json.dumps(result, indent=2)[:500]}")

        # At minimum, it should not be empty (that would indicate JSON parse failure)
        if result == {}:
            self.fail("Strategy analyzer returned empty dict — LLM likely returned "
                     "markdown-fenced JSON that wasn't stripped")


# ---------------------------------------------------------------------------
# Report Generator Integration
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LLM, "Requires ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK")
class TestReportGeneratorIntegration(unittest.TestCase):
    """Integration tests for ReportGenerator with real LLM."""

    def setUp(self):
        self.llm = _make_llm_client()
        self.prompts = _make_prompt_engine()

    def test_generate_produces_markdown_report(self):
        """Verify the report generator produces a Markdown report from real LLM."""
        from report_generator import ReportGenerator
        gen = ReportGenerator(self.llm, self.prompts)

        result = gen.generate(
            business_type="massage therapy",
            location="San Jose 95125",
            user_profile={"business_type": "massage", "budget_range": {"min": 50000, "max": 100000}},
            stats_report={"total_businesses": 3, "rating_stats": {"mean": 4.2}},
            sentiment_report={"overall_sentiment": "positive", "positive_themes": ["friendly"]},
            web_research={"rent_ranges": {"average": "$2500/mo"}},
            strategy={"market_saturation": {"level": "moderate"}},
        )

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100, "Report should be substantial")
        # Should contain markdown headers
        self.assertIn("#", result, "Report should contain markdown headers")

        print(f"\nReport length: {len(result)} chars")
        print(f"Report preview:\n{result[:300]}...")

    def test_generate_with_missing_data(self):
        """Verify generate_with_missing_data works with partial inputs."""
        from report_generator import ReportGenerator
        gen = ReportGenerator(self.llm, self.prompts)
        result = gen.generate_with_missing_data(
            "massage therapy", "San Jose 95125",
            stats_report={"total_businesses": 3},
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)

    def test_save_and_load_report(self):
        """Verify save_report writes readable Markdown."""
        from report_generator import ReportGenerator
        gen = ReportGenerator(self.llm, self.prompts)
        report = gen.generate_with_missing_data("massage", "San Jose")

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            gen.save_report(report, path)
            loaded = open(path, encoding="utf-8").read()
            self.assertEqual(loaded, report)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Web Researcher Integration (requires both LLM and Tavily)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LLM and _HAS_TAVILY,
                     "Requires both LLM key and TAVILY_API_KEY")
class TestWebResearcherIntegration(unittest.TestCase):
    """Integration tests for WebResearcher with real LLM and search API."""

    def setUp(self):
        self.llm = _make_llm_client()
        self.prompts = _make_prompt_engine()
        from web_searcher import WebSearcher
        self.searcher = WebSearcher()

    def test_research_produces_structured_output(self):
        """Verify web researcher returns structured market intelligence."""
        from web_researcher import WebResearcher
        wr = WebResearcher(self.llm, self.searcher, self.prompts)
        result = wr.research("massage therapy", "San Jose CA")

        self.assertIsInstance(result, dict)
        self.assertIn("rent_ranges", result)
        self.assertIn("industry_trends", result)
        self.assertIn("queries_executed", result)
        self.assertEqual(result["queries_executed"], 5)
        self.assertGreater(result["raw_result_count"], 0,
                          "Expected at least some search results")

        print(f"\nWeb research: {json.dumps(result, indent=2)[:500]}")


# ---------------------------------------------------------------------------
# Full Pipeline Integration
# ---------------------------------------------------------------------------

@unittest.skipUnless(_HAS_LLM, "Requires ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK")
class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end integration test of the analysis pipeline (skips web stage)."""

    def test_pipeline_stats_through_report(self):
        """Run stats → sentiment → strategy → report with real LLM.

        Skips interview (needs stdin) and web research (needs Tavily).
        """
        from stats_analyzer import StatsAnalyzer
        from sentiment_analyzer import SentimentAnalyzer
        from strategy_analyzer import StrategyAnalyzer
        from report_generator import ReportGenerator

        llm = _make_llm_client()
        prompts = _make_prompt_engine()

        # Stage 1: Stats (no LLM needed)
        stats = StatsAnalyzer(SAMPLE_BUSINESSES)
        stats_report = stats.analyze()
        self.assertGreater(stats_report["total_businesses"], 0)

        # Stage 2: Sentiment (LLM)
        sentiment = SentimentAnalyzer(llm, prompts)
        sentiment_report = sentiment.analyze(SAMPLE_BUSINESSES, "massage", "San Jose")
        self.assertIn("overall_sentiment", sentiment_report)

        # Mock user profile and web research for remaining stages
        user_profile = {
            "business_type": "massage",
            "target_clientele": "General public",
            "budget_range": {"min": 50000, "max": 100000},
        }
        web_research = {"rent_ranges": {"average": "$2500/mo"}}

        # Stage 3: Strategy (LLM)
        strategy = StrategyAnalyzer(llm, prompts)
        strategy_report = strategy.analyze(
            "massage", "San Jose", user_profile,
            stats_report, sentiment_report, web_research,
        )
        self.assertIsInstance(strategy_report, dict)

        # Stage 4: Report (LLM)
        report_gen = ReportGenerator(llm, prompts)
        final_report = report_gen.generate(
            "massage", "San Jose",
            user_profile, stats_report, sentiment_report,
            web_research, strategy_report,
        )
        self.assertIsInstance(final_report, str)
        self.assertIn("#", final_report)

        # Token usage should reflect all LLM calls
        usage = llm.get_usage_summary()
        self.assertGreater(usage["total_calls"], 0)
        self.assertGreater(usage["total_input_tokens"], 0)

        print(f"\nPipeline usage: {json.dumps(usage, indent=2)}")
        print(f"Final report length: {len(final_report)} chars")

    def test_pipeline_with_save_artifacts(self):
        """Run partial pipeline and save all artifacts to disk."""
        from stats_analyzer import StatsAnalyzer
        from sentiment_analyzer import SentimentAnalyzer

        llm = _make_llm_client()
        prompts = _make_prompt_engine()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Stats
            stats = StatsAnalyzer(SAMPLE_BUSINESSES)
            stats_report = stats.analyze()
            stats.save_report(stats_report, os.path.join(tmpdir, "stats.json"))

            # Sentiment
            sentiment = SentimentAnalyzer(llm, prompts)
            sentiment_report = sentiment.analyze(SAMPLE_BUSINESSES, "massage", "SJ")
            sentiment.save_report(sentiment_report, os.path.join(tmpdir, "sentiment.json"))

            # Verify saved files are valid JSON
            for fname in ("stats.json", "sentiment.json"):
                path = os.path.join(tmpdir, fname)
                self.assertTrue(os.path.exists(path))
                with open(path) as f:
                    data = json.load(f)
                self.assertIsInstance(data, dict)


if __name__ == "__main__":
    unittest.main()

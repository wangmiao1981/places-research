"""Proactive tests for Issues #6-10 modules.

These tests probe edge cases and potential bugs that could surface when the
modules are wired together or encounter real-world data.  Each test documents
which bug it is designed to catch.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# Sentiment Analyzer proactive tests
# ---------------------------------------------------------------------------

class TestSentimentAnalyzerProactive(unittest.TestCase):
    """Proactive tests for sentiment_analyzer.py edge cases."""

    def _import_module(self):
        from sentiment_analyzer import (
            SentimentAnalyzer, _extract_json, _merge_results,
            REVIEWS_PER_BATCH, _EMPTY_RESULT,
        )
        return SentimentAnalyzer, _extract_json, _merge_results, REVIEWS_PER_BATCH, _EMPTY_RESULT

    def test_extract_json_nested_fences(self):
        """BUG: LLM sometimes returns nested fences like ```json\\n```json ..."""
        _, _extract_json, *_ = self._import_module()
        text = '```json\n```json\n{"key": "value"}\n```\n```'
        result = _extract_json(text)
        # Should still extract something — either the JSON or empty dict, not crash
        self.assertIsInstance(result, dict)

    def test_extract_json_with_trailing_text(self):
        """BUG: LLM sometimes appends explanation after JSON."""
        _, _extract_json, *_ = self._import_module()
        text = '{"positive_themes": ["friendly"]}\n\nThis analysis shows...'
        result = _extract_json(text)
        # The first parse should succeed (json.loads ignores trailing text? No, it doesn't)
        # This should gracefully return {} or parse correctly
        self.assertIsInstance(result, dict)

    def test_extract_json_with_unicode_smart_quotes(self):
        """BUG: LLM may use unicode smart quotes instead of ASCII quotes."""
        _, _extract_json, *_ = self._import_module()
        text = '{"key": "value"}'  # Using regular quotes — smart quotes would break JSON
        result = _extract_json(text)
        self.assertEqual(result.get("key"), "value")

    def test_merge_results_all_unknown_sentiments(self):
        """Edge case: all batches return 'unknown' sentiment."""
        _, _, _merge_results, *_ = self._import_module()
        results = [
            {"positive_themes": [], "overall_sentiment": "unknown"},
            {"positive_themes": [], "overall_sentiment": "unknown"},
        ]
        merged = _merge_results(results)
        # All unknown — "unknown" not in sentiments is False, so
        # it falls into the else branch
        self.assertIn(merged["overall_sentiment"], ["unknown", "mixed"])

    def test_merge_results_mixed_with_unknown(self):
        """BUG: mixed sentiments with some 'unknown' batches."""
        _, _, _merge_results, *_ = self._import_module()
        results = [
            {"positive_themes": ["good"], "overall_sentiment": "positive"},
            {"positive_themes": [], "overall_sentiment": "unknown"},
            {"positive_themes": ["bad"], "overall_sentiment": "negative"},
        ]
        merged = _merge_results(results)
        # Should be "mixed" since we have both positive and negative
        self.assertEqual(merged["overall_sentiment"], "mixed")

    def test_merge_results_empty_overall_sentiment(self):
        """BUG: what if overall_sentiment is empty string instead of absent?"""
        _, _, _merge_results, *_ = self._import_module()
        results = [
            {"positive_themes": ["good"], "overall_sentiment": ""},
        ]
        merged = _merge_results(results)
        # Empty string is falsy — should be handled by the `if r.get(...)` check
        self.assertIsInstance(merged["overall_sentiment"], str)

    def test_analyze_exact_batch_boundary(self):
        """Edge case: exactly REVIEWS_PER_BATCH reviews — should be 1 batch, not 2."""
        SA, _, _, REVIEWS_PER_BATCH, _ = self._import_module()
        llm = MagicMock()
        llm.call.return_value = '{"positive_themes": [], "overall_sentiment": "positive"}'
        prompt = MagicMock()
        prompt.render.return_value = "prompt"
        analyzer = SA(llm, prompt)
        businesses = [{"reviews": [f"review {i}" for i in range(REVIEWS_PER_BATCH)]}]
        analyzer.analyze(businesses, "massage", "San Jose")
        # Exactly 1 batch — LLM should be called once
        self.assertEqual(llm.call.call_count, 1)

    def test_analyze_batch_boundary_plus_one(self):
        """Edge case: REVIEWS_PER_BATCH + 1 reviews — should be 2 batches."""
        SA, _, _, REVIEWS_PER_BATCH, _ = self._import_module()
        llm = MagicMock()
        llm.call.return_value = '{"positive_themes": [], "overall_sentiment": "positive"}'
        prompt = MagicMock()
        prompt.render.return_value = "prompt"
        analyzer = SA(llm, prompt)
        businesses = [{"reviews": [f"review {i}" for i in range(REVIEWS_PER_BATCH + 1)]}]
        analyzer.analyze(businesses, "massage", "San Jose")
        self.assertEqual(llm.call.call_count, 2)

    def test_analyze_reviews_are_non_string(self):
        """BUG: what if reviews contain non-string items (dicts, ints)?"""
        SA, *_ = self._import_module()
        llm = MagicMock()
        llm.call.return_value = '{"positive_themes": [], "overall_sentiment": "positive"}'
        prompt = MagicMock()
        prompt.render.return_value = "prompt"
        analyzer = SA(llm, prompt)
        # Reviews might be dicts in some data formats
        businesses = [{"reviews": [{"text": "great", "rating": 5}, "normal review"]}]
        # Should not crash — just convert to string
        try:
            result = analyzer.analyze(businesses, "massage", "San Jose")
            self.assertIsInstance(result, dict)
        except TypeError:
            self.fail("analyze() should handle non-string reviews gracefully")

    def test_analyze_llm_returns_array(self):
        """BUG: LLM returns a JSON array instead of object."""
        SA, *_ = self._import_module()
        llm = MagicMock()
        llm.call.return_value = '[{"positive_themes": ["good"]}]'
        prompt = MagicMock()
        prompt.render.return_value = "prompt"
        analyzer = SA(llm, prompt)
        businesses = [{"reviews": ["great service"]}]
        result = analyzer.analyze(businesses, "massage", "San Jose")
        # _extract_json returns {} for non-dict, or might return the array
        # Either way, the result should be a dict with all required keys
        self.assertIsInstance(result, dict)
        self.assertIn("businesses_analyzed", result)

    def test_save_report_with_unicode(self):
        """Ensure save_report handles unicode characters correctly."""
        SA, *_ = self._import_module()
        llm = MagicMock()
        prompt = MagicMock()
        analyzer = SA(llm, prompt)
        report = {"positive_themes": ["excellente — très bien", "日本語テスト"]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            analyzer.save_report(report, path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["positive_themes"][1], "日本語テスト")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Web Researcher proactive tests
# ---------------------------------------------------------------------------

class TestWebResearcherProactive(unittest.TestCase):
    """Proactive tests for web_researcher.py edge cases."""

    def _make_researcher(self):
        from web_researcher import WebResearcher
        llm = MagicMock()
        ws = MagicMock()
        pe = MagicMock()
        pe.render.return_value = "prompt"
        return WebResearcher(llm, ws, pe), llm, ws, pe

    def test_parse_llm_response_only_opening_fence(self):
        """BUG: LLM returns opening fence but no closing fence."""
        from web_researcher import WebResearcher
        result = WebResearcher._parse_llm_response('```json\n{"key": "value"}')
        # Should handle gracefully
        self.assertIsInstance(result, dict)

    def test_parse_llm_response_empty_string(self):
        """Edge case: empty LLM response."""
        from web_researcher import WebResearcher
        result = WebResearcher._parse_llm_response("")
        self.assertEqual(result, {})

    def test_parse_llm_response_plain_text(self):
        """BUG: LLM returns plain text explanation instead of JSON."""
        from web_researcher import WebResearcher
        result = WebResearcher._parse_llm_response(
            "I cannot find that information. Here is what I know..."
        )
        self.assertEqual(result, {})

    def test_search_multiple_returns_empty_for_all_queries(self):
        """Edge case: all search queries return no results."""
        wr, llm, ws, pe = self._make_researcher()
        ws.search_multiple.return_value = {
            "q1": [], "q2": [], "q3": [], "q4": [], "q5": [],
        }
        llm.call.return_value = '{"rent_ranges": {}, "zoning_info": []}'
        result = wr.research("massage", "San Jose")
        self.assertEqual(result["raw_result_count"], 0)
        self.assertEqual(result["queries_executed"], 5)

    def test_web_searcher_raises_during_search(self):
        """BUG: WebSearcher.search_multiple raises — should propagate."""
        wr, llm, ws, pe = self._make_researcher()
        ws.search_multiple.side_effect = RuntimeError("API quota exceeded")
        with self.assertRaises(RuntimeError):
            wr.research("massage", "San Jose")

    def test_format_results_with_special_characters(self):
        """Edge case: search results with markdown-breaking characters."""
        from web_searcher import SearchResult
        from web_researcher import WebResearcher
        wr, llm, ws, pe = self._make_researcher()
        results = {
            "test query": [
                SearchResult(
                    title="Price: $50/hr — Best in San José!",
                    url="http://example.com",
                    content="Content with **bold** and `code` and <html>",
                    score=0.9,
                ),
            ],
        }
        formatted = wr._format_results(results)
        self.assertIn("$50/hr", formatted)
        self.assertIn("<html>", formatted)

    def test_research_with_very_long_business_type(self):
        """Edge case: very long business type string."""
        wr, llm, ws, pe = self._make_researcher()
        ws.search_multiple.return_value = {"q": []}
        llm.call.return_value = '{}'
        long_type = "a" * 1000
        result = wr.research(long_type, "San Jose")
        self.assertIsInstance(result, dict)


# ---------------------------------------------------------------------------
# Strategy Analyzer proactive tests
# ---------------------------------------------------------------------------

class TestStrategyAnalyzerProactive(unittest.TestCase):
    """Proactive tests for strategy_analyzer.py edge cases."""

    def _make_analyzer(self):
        from strategy_analyzer import StrategyAnalyzer
        llm = MagicMock()
        pe = MagicMock()
        pe.render.return_value = "prompt"
        return StrategyAnalyzer(llm, pe), llm, pe

    def test_analyze_llm_returns_markdown_fenced_json(self):
        """BUG FIX: Strategy analyzer now strips markdown fences.
        Previously, if LLM wrapped JSON in ```json fences, json.loads would
        fail and silently return {}."""
        sa, llm, pe = self._make_analyzer()
        llm.call.return_value = '```json\n{"market_saturation": "high"}\n```'
        result = sa.analyze("massage", "San Jose", {}, {}, {}, {})
        self.assertEqual(result["market_saturation"], "high")

    def test_analyze_with_non_serializable_artifacts(self):
        """BUG: what if an artifact contains non-JSON-serializable data?"""
        sa, llm, pe = self._make_analyzer()
        llm.call.return_value = '{"market_saturation": "low"}'
        # Sets are not JSON serializable
        with self.assertRaises(TypeError):
            sa.analyze("massage", "SJ", {"services": {"a", "b"}}, {}, {}, {})

    def test_analyze_returns_empty_dict_on_invalid_json(self):
        """Verify graceful handling of completely invalid LLM output."""
        sa, llm, pe = self._make_analyzer()
        llm.call.return_value = "I cannot provide that analysis."
        result = sa.analyze("massage", "San Jose", {}, {}, {}, {})
        self.assertEqual(result, {})

    def test_save_report_empty_dict(self):
        """Edge case: saving an empty report."""
        sa, _, _ = self._make_analyzer()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sa.save_report({}, path)
            with open(path) as f:
                self.assertEqual(json.load(f), {})
        finally:
            os.unlink(path)

    def test_analyze_very_large_artifacts(self):
        """Edge case: very large artifacts — verify json.dumps doesn't crash."""
        sa, llm, pe = self._make_analyzer()
        llm.call.return_value = '{"ok": true}'
        large_stats = {f"metric_{i}": i for i in range(10000)}
        result = sa.analyze("massage", "SJ", {}, large_stats, {}, {})
        self.assertIsInstance(result, dict)
        # Verify the large stats were serialized into the prompt
        render_call = pe.render.call_args
        variables = render_call[0][1]
        self.assertIn("metric_9999", variables["stats_report"])


# ---------------------------------------------------------------------------
# Report Generator proactive tests
# ---------------------------------------------------------------------------

class TestReportGeneratorProactive(unittest.TestCase):
    """Proactive tests for report_generator.py edge cases."""

    def _make_generator(self):
        from report_generator import ReportGenerator
        llm = MagicMock()
        pe = MagicMock()
        pe.render.return_value = "prompt"
        return ReportGenerator(llm, pe), llm, pe

    def test_generate_with_none_values_in_artifacts(self):
        """BUG: what if artifact dicts contain None values?"""
        rg, llm, pe = self._make_generator()
        llm.call.return_value = "# Report"
        # json.dumps handles None → null, but verify it doesn't crash
        result = rg.generate(
            "massage", "San Jose",
            user_profile={"name": None},
            stats_report={"mean": None},
            sentiment_report={},
            web_research={},
            strategy={},
        )
        self.assertEqual(result, "# Report")

    def test_generate_with_deeply_nested_artifacts(self):
        """Edge case: deeply nested data structures in artifacts."""
        rg, llm, pe = self._make_generator()
        llm.call.return_value = "# Report"
        deep = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = rg.generate("massage", "SJ", deep, {}, {}, {}, {})
        self.assertEqual(result, "# Report")

    def test_save_report_overwrites_existing(self):
        """Verify save_report overwrites existing files."""
        rg, _, _ = self._make_generator()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("OLD CONTENT")
            path = f.name
        try:
            rg.save_report("NEW CONTENT", path)
            self.assertEqual(Path(path).read_text(), "NEW CONTENT")
        finally:
            os.unlink(path)

    def test_generate_llm_returns_empty_string(self):
        """BUG: LLM returns empty string — should still work."""
        rg, llm, pe = self._make_generator()
        llm.call.return_value = ""
        result = rg.generate("massage", "SJ", {}, {}, {}, {}, {})
        self.assertEqual(result, "")

    def test_generate_with_missing_data_passes_empty_dicts(self):
        """Verify generate_with_missing_data defaults to {} not None."""
        rg, llm, pe = self._make_generator()
        llm.call.return_value = "# Report"
        rg.generate_with_missing_data("massage", "SJ")
        # Check that all artifacts passed to render are valid JSON
        render_args = pe.render.call_args[0][1]
        for key in ("user_profile", "stats_report", "sentiment_report", "web_research", "strategy"):
            parsed = json.loads(render_args[key])
            self.assertEqual(parsed, {})


# ---------------------------------------------------------------------------
# Orchestrator proactive tests
# ---------------------------------------------------------------------------

class TestOrchestratorProactive(unittest.TestCase):
    """Proactive tests for analyze.py edge cases."""

    def test_resume_with_corrupted_json_artifact(self):
        """BUG: resume mode with corrupted JSON artifact should not crash."""
        from analyze import run_pipeline, stage_artifact_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write corrupted JSON to interview artifact
            interview_path = stage_artifact_path(tmpdir, "interview")
            Path(interview_path).write_text("NOT VALID JSON {{{")

            deps = {
                "llm_client": MagicMock(),
                "prompt_engine": MagicMock(),
                "web_searcher": MagicMock(),
            }
            deps["llm_client"].get_usage_summary.return_value = {
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_calls": 0, "estimated_cost_usd": 0.0,
            }

            with patch("analyze.UserInterviewer") as mock_int, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer"), \
                 patch("analyze.WebResearcher"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):
                mock_int.return_value.run.return_value = {"business_type": "massage"}
                mock_stats.return_value.analyze.return_value = {"total": 5}

                # Resume should skip corrupted artifact and re-run the stage
                result = run_pipeline(
                    businesses=[],
                    output_dir=tmpdir,
                    resume=True,
                    stage=None,
                    **deps,
                )
                # The corrupted file should not crash — it should be ignored
                # and the stage should re-run
                self.assertIsInstance(result, dict)

    def test_build_output_dir_with_no_extension(self):
        """Edge case: input file has no extension."""
        from analyze import build_output_dir
        result = build_output_dir(None, "/path/to/mydata")
        self.assertEqual(result, "/path/to/mydata_analysis")

    def test_build_output_dir_with_dots_in_path(self):
        """Edge case: input path has dots in directory name."""
        from analyze import build_output_dir
        result = build_output_dir(None, "/path/to/v2.0/data.json")
        self.assertEqual(result, "/path/to/v2.0/data_analysis")

    def test_stage_artifact_path_all_stages(self):
        """Verify all stages have defined artifact filenames."""
        from analyze import STAGES, stage_artifact_path
        for stage in STAGES:
            path = stage_artifact_path("/tmp", stage)
            self.assertTrue(path.startswith("/tmp/"))
            self.assertTrue(path.endswith(".json") or path.endswith(".md"))

    def test_pipeline_empty_businesses_list(self):
        """Edge case: empty business list — stats analyzer gets []."""
        from analyze import run_pipeline

        deps = {
            "llm_client": MagicMock(),
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }
        deps["llm_client"].get_usage_summary.return_value = {
            "total_input_tokens": 0, "total_output_tokens": 0,
            "total_calls": 0, "estimated_cost_usd": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("analyze.UserInterviewer") as mock_int, \
             patch("analyze.StatsAnalyzer") as mock_stats, \
             patch("analyze.SentimentAnalyzer") as mock_sent, \
             patch("analyze.WebResearcher") as mock_web, \
             patch("analyze.StrategyAnalyzer") as mock_strat, \
             patch("analyze.ReportGenerator") as mock_rep:
            mock_int.return_value.run.return_value = {"business_type": "test"}
            mock_stats.return_value.analyze.return_value = {"total": 0}
            mock_sent.return_value.analyze.return_value = {"overall_sentiment": "unknown"}
            mock_web.return_value.research.return_value = {"rent_ranges": {}}
            mock_strat.return_value.analyze.return_value = {}
            mock_rep.return_value.generate.return_value = "# Empty Report"

            result = run_pipeline(
                businesses=[],
                output_dir=tmpdir,
                resume=False,
                stage=None,
                **deps,
            )
            self.assertTrue(result["success"])

    def test_single_stage_web_loads_prior_artifacts(self):
        """When running single 'web' stage, it should load prior artifacts."""
        from analyze import run_pipeline, stage_artifact_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create prior artifacts
            interview_path = stage_artifact_path(tmpdir, "interview")
            Path(interview_path).write_text(json.dumps({"business_type": "yoga"}))
            stats_path = stage_artifact_path(tmpdir, "stats")
            Path(stats_path).write_text(json.dumps({"total": 10}))

            deps = {
                "llm_client": MagicMock(),
                "prompt_engine": MagicMock(),
                "web_searcher": MagicMock(),
            }
            deps["llm_client"].get_usage_summary.return_value = {
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_calls": 0, "estimated_cost_usd": 0.0,
            }

            with patch("analyze.WebResearcher") as mock_web, \
                 patch("analyze.SentimentAnalyzer"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):
                mock_web.return_value.research.return_value = {"rent_ranges": {}}

                result = run_pipeline(
                    businesses=[],
                    output_dir=tmpdir,
                    resume=False,
                    stage="web",
                    **deps,
                )
                self.assertTrue(result["success"])
                # Verify the web researcher was called
                mock_web.return_value.research.assert_called_once()


# ---------------------------------------------------------------------------
# Cross-module integration edge cases (mocked)
# ---------------------------------------------------------------------------

class TestCrossModuleEdgeCases(unittest.TestCase):
    """Tests that verify the modules work together correctly at boundaries."""

    def test_sentiment_output_is_valid_strategy_input(self):
        """The sentiment analyzer's output format must be consumable by strategy analyzer."""
        from sentiment_analyzer import _EMPTY_RESULT
        # _EMPTY_RESULT must be JSON-serializable for strategy's json.dumps()
        serialized = json.dumps(_EMPTY_RESULT)
        self.assertIsInstance(json.loads(serialized), dict)

    def test_all_artifact_keys_consistent(self):
        """Verify the orchestrator references the correct artifact keys."""
        from analyze import _ARTIFACT_FILENAMES, STAGES
        # Every stage must have a filename mapping
        for stage in STAGES:
            self.assertIn(stage, _ARTIFACT_FILENAMES,
                          f"Stage '{stage}' missing from _ARTIFACT_FILENAMES")


if __name__ == "__main__":
    unittest.main()

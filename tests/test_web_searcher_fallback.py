# Copyright 2025 Miao Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for issue #24: WebSearcher API key error message and pipeline fallback.

Covers:
  - Clear, user-friendly EnvironmentError when TAVILY_API_KEY is missing.
  - analyze.run_pipeline continues (success=True) when the web stage fails.
  - warnings field is populated when web stage is skipped.
  - Subsequent stages (strategy, report) still execute after a web failure.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyze import STAGES, run_pipeline, stage_artifact_path


# ---------------------------------------------------------------------------
# Shared test businesses fixture
# ---------------------------------------------------------------------------

SAMPLE_BUSINESSES = [
    {"place_id": "p1", "name": "Relax Spa", "rating": 4.5, "total_ratings": 100, "reviews": []},
]


# ---------------------------------------------------------------------------
# WebSearcher: missing API key error message
# ---------------------------------------------------------------------------

class TestWebSearcherMissingApiKey(unittest.TestCase):
    """Verify that missing TAVILY_API_KEY raises a clear EnvironmentError."""

    def _raise_without_key(self):
        """Call WebSearcher() with no key in environment or argument."""
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from web_searcher import WebSearcher
            WebSearcher(cache_dir="/tmp/test_cache_fallback")

    def test_raises_environment_error(self):
        with self.assertRaises(EnvironmentError):
            self._raise_without_key()

    def test_error_message_mentions_tavily_api_key(self):
        """The error message must mention TAVILY_API_KEY so users know what to set."""
        with self.assertRaises(EnvironmentError) as ctx:
            self._raise_without_key()
        self.assertIn("TAVILY_API_KEY", str(ctx.exception))

    def test_error_message_mentions_export(self):
        """The error message should show how to export the variable."""
        with self.assertRaises(EnvironmentError) as ctx:
            self._raise_without_key()
        self.assertIn("export", str(ctx.exception))

    def test_error_message_mentions_tavily_url(self):
        """The error message should point the user to where they get a key."""
        with self.assertRaises(EnvironmentError) as ctx:
            self._raise_without_key()
        self.assertIn("tavily.com", str(ctx.exception))

    def test_error_is_not_cryptic(self):
        """The error message must be longer than a bare 'key not found' stub."""
        with self.assertRaises(EnvironmentError) as ctx:
            self._raise_without_key()
        self.assertGreater(len(str(ctx.exception)), 80)


# ---------------------------------------------------------------------------
# run_pipeline: web stage failure is non-fatal
# ---------------------------------------------------------------------------

class TestPipelineWebStageFallback(unittest.TestCase):
    """Verify that a web stage error does not abort the full pipeline."""

    def _make_deps(self):
        deps = {
            "llm_client": MagicMock(),
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }
        deps["llm_client"].get_usage_summary.return_value = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "estimated_cost_usd": 0.0,
        }
        return deps

    def _run_pipeline_with_web_failure(self, tmpdir, web_exc=None):
        """Run the full pipeline with the web stage raising web_exc."""
        if web_exc is None:
            web_exc = EnvironmentError(
                "TAVILY_API_KEY is not set.\n\nexport TAVILY_API_KEY='tvly-...'\n"
                "Sign up at https://tavily.com"
            )
        deps = self._make_deps()

        with patch("analyze.UserInterviewer") as mock_interview, \
             patch("analyze.StatsAnalyzer") as mock_stats, \
             patch("analyze.SentimentAnalyzer") as mock_sentiment, \
             patch("analyze.WebResearcher") as mock_web, \
             patch("analyze.StrategyAnalyzer") as mock_strategy, \
             patch("analyze.ReportGenerator") as mock_report:

            mock_interview.return_value.run.return_value = {"business_type": "massage"}
            mock_stats.return_value.analyze.return_value = {"total_businesses": 1}
            mock_sentiment.return_value.analyze.return_value = {"overall_sentiment": "positive"}
            mock_web.return_value.research.side_effect = web_exc
            mock_strategy.return_value.analyze.return_value = {"market_saturation": {}}
            mock_report.return_value.generate.return_value = "# Report\nContent"

            result = run_pipeline(
                businesses=list(SAMPLE_BUSINESSES),
                output_dir=tmpdir,
                resume=False,
                stage=None,
                **deps,
            )

        return result, {
            "interview": mock_interview,
            "stats": mock_stats,
            "sentiment": mock_sentiment,
            "web": mock_web,
            "strategy": mock_strategy,
            "report": mock_report,
        }

    # ------------------------------------------------------------------
    # success / failure flags
    # ------------------------------------------------------------------

    def test_pipeline_succeeds_when_web_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertTrue(result["success"])

    def test_failed_stage_is_none_when_web_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertIsNone(result["failed_stage"])

    def test_error_is_none_when_web_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertIsNone(result["error"])

    # ------------------------------------------------------------------
    # warnings field
    # ------------------------------------------------------------------

    def test_warnings_field_present_in_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertIn("warnings", result)

    def test_warnings_contains_web_skip_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertTrue(len(result["warnings"]) > 0)
            self.assertIn("web", result["warnings"][0].lower())

    def test_warnings_empty_on_clean_run(self):
        """No warnings should appear when everything succeeds."""
        deps = self._make_deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher") as mock_web, \
                 patch("analyze.StrategyAnalyzer") as mock_strategy, \
                 patch("analyze.ReportGenerator") as mock_report:

                mock_interview.return_value.run.return_value = {"business_type": "massage"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 1}
                mock_sentiment.return_value.analyze.return_value = {}
                mock_web.return_value.research.return_value = {"rent_ranges": {}}
                mock_strategy.return_value.analyze.return_value = {}
                mock_report.return_value.generate.return_value = "# Report"

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    **deps,
                )

            self.assertEqual(result["warnings"], [])

    # ------------------------------------------------------------------
    # completed_stages still includes 'web'
    # ------------------------------------------------------------------

    def test_web_stage_listed_in_completed_when_it_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertIn("web", result["completed_stages"])

    def test_all_stages_listed_in_completed_when_web_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(tmpdir)
            self.assertEqual(result["completed_stages"], STAGES)

    # ------------------------------------------------------------------
    # downstream stages still execute
    # ------------------------------------------------------------------

    def test_strategy_stage_still_runs_after_web_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, mocks = self._run_pipeline_with_web_failure(tmpdir)
            mocks["strategy"].return_value.analyze.assert_called_once()

    def test_report_stage_still_runs_after_web_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, mocks = self._run_pipeline_with_web_failure(tmpdir)
            mocks["report"].return_value.generate.assert_called_once()

    def test_strategy_receives_empty_web_research_on_web_failure(self):
        """Strategy stage should receive an empty dict for web_research."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, mocks = self._run_pipeline_with_web_failure(tmpdir)
            call_kwargs = mocks["strategy"].return_value.analyze.call_args[1]
            self.assertEqual(call_kwargs["web_research"], {})

    def test_report_receives_empty_web_research_on_web_failure(self):
        """Report stage should receive an empty dict for web_research."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, mocks = self._run_pipeline_with_web_failure(tmpdir)
            call_kwargs = mocks["report"].return_value.generate.call_args[1]
            self.assertEqual(call_kwargs["web_research"], {})

    # ------------------------------------------------------------------
    # non-web failures are still fatal
    # ------------------------------------------------------------------

    def test_non_web_failure_is_still_fatal(self):
        """Failures in stages other than web must still abort the pipeline."""
        deps = self._make_deps()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):

                mock_interview.return_value.run.return_value = {"business_type": "massage"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 1}
                mock_sentiment.return_value.analyze.side_effect = RuntimeError("LLM exploded")

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    **deps,
                )

            self.assertFalse(result["success"])
            self.assertEqual(result["failed_stage"], "sentiment")

    # ------------------------------------------------------------------
    # various web failure types
    # ------------------------------------------------------------------

    def test_runtime_error_in_web_stage_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(
                tmpdir, web_exc=RuntimeError("Tavily API 500")
            )
            self.assertTrue(result["success"])

    def test_connection_error_in_web_stage_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(
                tmpdir, web_exc=OSError("Connection refused")
            )
            self.assertTrue(result["success"])

    def test_warning_message_contains_original_exception_text(self):
        """The warning recorded in the result should include the original error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_pipeline_with_web_failure(
                tmpdir,
                web_exc=EnvironmentError("TAVILY_API_KEY is not set."),
            )
            self.assertTrue(
                any("TAVILY_API_KEY" in w for w in result["warnings"]),
                f"Expected TAVILY_API_KEY in warnings, got: {result['warnings']}",
            )


if __name__ == "__main__":
    unittest.main()

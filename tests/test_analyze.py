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

"""Tests for analyze.py — CLI orchestrator that wires all analysis stages."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analyze import (
    STAGES,
    build_output_dir,
    parse_args,
    run_pipeline,
    stage_artifact_path,
)


SAMPLE_BUSINESSES = [
    {"place_id": "p1", "name": "Relax Spa", "rating": 4.5, "total_ratings": 100, "reviews": []},
    {"place_id": "p2", "name": "Deep Tissue", "rating": 4.0, "total_ratings": 50, "reviews": []},
]


class TestParseArgs(unittest.TestCase):

    def test_input_required(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_input_only(self):
        args = parse_args(["--input", "data.json"])
        self.assertEqual(args.input, "data.json")
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.resume)
        self.assertIsNone(args.stage)

    def test_all_flags(self):
        args = parse_args([
            "--input", "data.json",
            "--output-dir", "/tmp/out",
            "--resume",
            "--stage", "sentiment",
        ])
        self.assertEqual(args.input, "data.json")
        self.assertEqual(args.output_dir, "/tmp/out")
        self.assertTrue(args.resume)
        self.assertEqual(args.stage, "sentiment")

    def test_stage_choices(self):
        for stage in ("interview", "stats", "sentiment", "web", "strategy", "report"):
            args = parse_args(["--input", "d.json", "--stage", stage])
            self.assertEqual(args.stage, stage)

    def test_invalid_stage(self):
        with self.assertRaises(SystemExit):
            parse_args(["--input", "d.json", "--stage", "invalid"])


class TestBuildOutputDir(unittest.TestCase):

    def test_explicit_output_dir(self):
        self.assertEqual(build_output_dir("/tmp/out", "data.json"), "/tmp/out")

    def test_auto_from_input(self):
        result = build_output_dir(None, "/path/to/grid_search_results.json")
        self.assertEqual(result, "/path/to/grid_search_results_analysis")

    def test_auto_strips_json_extension(self):
        result = build_output_dir(None, "businesses.json")
        self.assertEqual(result, "businesses_analysis")


class TestStageArtifactPath(unittest.TestCase):

    def test_paths(self):
        self.assertEqual(
            stage_artifact_path("/out", "interview"),
            "/out/user_profile.json",
        )
        self.assertEqual(
            stage_artifact_path("/out", "stats"),
            "/out/stats_report.json",
        )
        self.assertEqual(
            stage_artifact_path("/out", "sentiment"),
            "/out/sentiment_report.json",
        )
        self.assertEqual(
            stage_artifact_path("/out", "web"),
            "/out/web_research.json",
        )
        self.assertEqual(
            stage_artifact_path("/out", "strategy"),
            "/out/strategy_report.json",
        )
        self.assertEqual(
            stage_artifact_path("/out", "report"),
            "/out/final_report.md",
        )


class TestStageOrder(unittest.TestCase):

    def test_stage_order(self):
        self.assertEqual(
            STAGES,
            ["interview", "stats", "sentiment", "web", "strategy", "report"],
        )


class TestRunPipeline(unittest.TestCase):

    def _make_deps(self):
        """Return a dict of mocked dependencies for run_pipeline."""
        return {
            "llm_client": MagicMock(),
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

    def _run_with_mocks(self, output_dir, businesses=None, resume=False,
                        stage=None, deps=None):
        if businesses is None:
            businesses = list(SAMPLE_BUSINESSES)
        if deps is None:
            deps = self._make_deps()

        # Set up LLM client usage summary
        deps["llm_client"].get_usage_summary.return_value = {
            "total_input_tokens": 1000,
            "total_output_tokens": 500,
            "total_calls": 5,
            "estimated_cost_usd": 0.05,
        }

        with patch("analyze.UserInterviewer") as mock_interview, \
             patch("analyze.StatsAnalyzer") as mock_stats, \
             patch("analyze.SentimentAnalyzer") as mock_sentiment, \
             patch("analyze.WebResearcher") as mock_web, \
             patch("analyze.StrategyAnalyzer") as mock_strategy, \
             patch("analyze.ReportGenerator") as mock_report:

            # Configure mock return values
            mock_interview.return_value.run.return_value = {"business_type": "massage"}
            mock_stats.return_value.analyze.return_value = {"total_businesses": 2}
            mock_sentiment.return_value.analyze.return_value = {"overall_sentiment": "positive"}
            mock_web.return_value.research.return_value = {"rent_ranges": {}}
            mock_strategy.return_value.analyze.return_value = {"market_saturation": {}}
            mock_report.return_value.generate.return_value = "# Report\nContent"

            result = run_pipeline(
                businesses=businesses,
                output_dir=output_dir,
                resume=resume,
                stage=stage,
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

    def test_full_pipeline_creates_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mocks = self._run_with_mocks(tmpdir)
            self.assertTrue(result["success"])
            # All 6 artifacts should exist
            for stage in STAGES:
                path = stage_artifact_path(tmpdir, stage)
                self.assertTrue(os.path.exists(path), f"Missing artifact: {path}")

    def test_full_pipeline_calls_stages_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mocks = self._run_with_mocks(tmpdir)
            # Each stage constructor or method should have been called
            mocks["interview"].return_value.run.assert_called_once()
            mocks["stats"].return_value.analyze.assert_called_once()
            mocks["sentiment"].return_value.analyze.assert_called_once()
            mocks["web"].return_value.research.assert_called_once()
            mocks["strategy"].return_value.analyze.assert_called_once()
            mocks["report"].return_value.generate.assert_called_once()

    def test_single_stage_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mocks = self._run_with_mocks(tmpdir, stage="stats")
            self.assertTrue(result["success"])
            # Only stats should have been called
            mocks["interview"].return_value.run.assert_not_called()
            mocks["stats"].return_value.analyze.assert_called_once()
            mocks["sentiment"].return_value.analyze.assert_not_called()

    def test_resume_skips_completed_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create interview and stats artifacts
            interview_path = stage_artifact_path(tmpdir, "interview")
            stats_path = stage_artifact_path(tmpdir, "stats")
            Path(interview_path).write_text(json.dumps({"business_type": "massage"}))
            Path(stats_path).write_text(json.dumps({"total_businesses": 2}))

            result, mocks = self._run_with_mocks(tmpdir, resume=True)
            self.assertTrue(result["success"])
            # Interview and stats should be skipped
            mocks["interview"].return_value.run.assert_not_called()
            mocks["stats"].return_value.analyze.assert_not_called()
            # Remaining stages should run
            mocks["sentiment"].return_value.analyze.assert_called_once()

    def test_resume_loads_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create interview artifact with specific data
            interview_path = stage_artifact_path(tmpdir, "interview")
            profile = {"business_type": "yoga", "budget_range": {"min": 10000, "max": 50000}}
            Path(interview_path).write_text(json.dumps(profile))

            result, mocks = self._run_with_mocks(tmpdir, resume=True)
            # Strategy should receive the loaded profile data
            strategy_call = mocks["strategy"].return_value.analyze.call_args
            self.assertEqual(strategy_call[1]["user_profile"]["business_type"], "yoga")

    def test_error_handling_saves_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deps = self._make_deps()
            deps["llm_client"].get_usage_summary.return_value = {
                "total_input_tokens": 0, "total_output_tokens": 0,
                "total_calls": 0, "estimated_cost_usd": 0.0,
            }

            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):

                mock_interview.return_value.run.return_value = {"business_type": "massage"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 2}
                # Sentiment stage raises an error
                mock_sentiment.return_value.analyze.side_effect = RuntimeError("LLM failed")

                result = run_pipeline(
                    businesses=SAMPLE_BUSINESSES,
                    output_dir=tmpdir,
                    resume=False,
                    stage=None,
                    **deps,
                )

            self.assertFalse(result["success"])
            self.assertEqual(result["failed_stage"], "sentiment")
            self.assertIn("LLM failed", result["error"])
            # Interview and stats artifacts should still exist
            self.assertTrue(os.path.exists(stage_artifact_path(tmpdir, "interview")))
            self.assertTrue(os.path.exists(stage_artifact_path(tmpdir, "stats")))

    def test_result_includes_usage_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_with_mocks(tmpdir)
            self.assertIn("usage", result)
            self.assertEqual(result["usage"]["total_calls"], 5)
            self.assertEqual(result["usage"]["estimated_cost_usd"], 0.05)

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "sub", "dir")
            result, _ = self._run_with_mocks(out)
            self.assertTrue(os.path.isdir(out))

    def test_report_stage_writes_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_with_mocks(tmpdir)
            report_path = stage_artifact_path(tmpdir, "report")
            content = Path(report_path).read_text()
            self.assertEqual(content, "# Report\nContent")

    def test_json_artifacts_are_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _ = self._run_with_mocks(tmpdir)
            for stage in ("interview", "stats", "sentiment", "web", "strategy"):
                path = stage_artifact_path(tmpdir, stage)
                with open(path) as f:
                    data = json.load(f)
                self.assertIsInstance(data, dict)

    def test_single_stage_report_requires_prior_artifacts(self):
        """Running only 'report' stage with no prior artifacts should still attempt it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result, mocks = self._run_with_mocks(tmpdir, stage="report")
            self.assertTrue(result["success"])
            mocks["report"].return_value.generate.assert_called_once()


class TestParseArgsNewFlags(unittest.TestCase):
    """Tests for the new --dry-run and --max-tokens flags."""

    def test_dry_run_flag_default_false(self):
        args = parse_args(["--input", "data.json"])
        self.assertFalse(args.dry_run)

    def test_dry_run_flag_set(self):
        args = parse_args(["--input", "data.json", "--dry-run"])
        self.assertTrue(args.dry_run)

    def test_max_tokens_default_none(self):
        args = parse_args(["--input", "data.json"])
        self.assertIsNone(args.max_tokens)

    def test_max_tokens_set(self):
        args = parse_args(["--input", "data.json", "--max-tokens", "50000"])
        self.assertEqual(args.max_tokens, 50000)

    def test_max_tokens_must_be_integer(self):
        with self.assertRaises(SystemExit):
            parse_args(["--input", "data.json", "--max-tokens", "not_an_int"])


class TestDryRun(unittest.TestCase):
    """Tests for --dry-run mode."""

    def _make_llm_client(self):
        mock = MagicMock()
        mock.get_usage_summary.return_value = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "estimated_cost_usd": 0.0,
        }
        return mock

    def test_dry_run_returns_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.SentimentAnalyzer") as mock_sa:
                # REVIEWS_PER_BATCH must be importable from sentiment_analyzer
                mock_sa  # not needed but kept for clarity
                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    dry_run=True,
                    llm_client=self._make_llm_client(),
                )
        self.assertTrue(result["success"])
        self.assertTrue(result.get("dry_run"))

    def test_dry_run_does_not_call_llm(self):
        llm_client = self._make_llm_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_pipeline(
                businesses=list(SAMPLE_BUSINESSES),
                output_dir=tmpdir,
                dry_run=True,
                llm_client=llm_client,
            )
        # The LLM client's call/stream methods must not have been invoked
        llm_client.call.assert_not_called()
        llm_client.stream.assert_not_called()

    def test_dry_run_does_not_invoke_stage_constructors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher") as mock_web, \
                 patch("analyze.StrategyAnalyzer") as mock_strategy, \
                 patch("analyze.ReportGenerator") as mock_report:

                run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    dry_run=True,
                    llm_client=self._make_llm_client(),
                )

                mock_interview.assert_not_called()
                mock_stats.assert_not_called()
                mock_sentiment.assert_not_called()
                mock_web.assert_not_called()
                mock_strategy.assert_not_called()
                mock_report.assert_not_called()

    def test_dry_run_no_artifacts_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_pipeline(
                businesses=list(SAMPLE_BUSINESSES),
                output_dir=tmpdir,
                dry_run=True,
                llm_client=self._make_llm_client(),
            )
            for s in STAGES:
                path = stage_artifact_path(tmpdir, s)
                self.assertFalse(
                    os.path.exists(path),
                    f"Dry-run should not create artifact: {path}",
                )

    def test_dry_run_completed_stages_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                businesses=list(SAMPLE_BUSINESSES),
                output_dir=tmpdir,
                dry_run=True,
                llm_client=self._make_llm_client(),
            )
        self.assertEqual(result["completed_stages"], [])

    def test_dry_run_single_stage(self):
        """dry_run with stage= only shows plan for that stage."""
        llm_client = self._make_llm_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                businesses=list(SAMPLE_BUSINESSES),
                output_dir=tmpdir,
                dry_run=True,
                stage="sentiment",
                llm_client=llm_client,
            )
        self.assertTrue(result["success"])
        llm_client.call.assert_not_called()


class TestMaxTokensBudget(unittest.TestCase):
    """Tests for --max-tokens budget enforcement."""

    def _make_deps_with_usage(self, total_tokens: int):
        """Return deps where llm_client reports total_tokens used."""
        mock = MagicMock()
        mock.get_usage_summary.return_value = {
            "total_input_tokens": total_tokens,
            "total_output_tokens": 0,
            "total_calls": 1,
            "estimated_cost_usd": 0.001,
        }
        return {
            "llm_client": mock,
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

    def test_pipeline_stops_when_budget_exceeded_before_sentiment(self):
        """If token usage already exceeds budget before sentiment, stop."""
        deps = self._make_deps_with_usage(total_tokens=10_000)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):

                mock_interview.return_value.run.return_value = {"business_type": "spa"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 2}

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    max_tokens=5_000,  # budget already exceeded
                    **deps,
                )

        self.assertFalse(result["success"])
        self.assertEqual(result["failed_stage"], "sentiment")
        self.assertIn("Token budget exceeded", result["error"])
        self.assertIn("5000", result["error"])
        # Non-LLM stages (interview, stats) should have completed
        self.assertIn("interview", result["completed_stages"])
        self.assertIn("stats", result["completed_stages"])
        # Sentiment should not have been called
        mock_sentiment.return_value.analyze.assert_not_called()

    def test_pipeline_stops_when_budget_exceeded_before_strategy(self):
        """Budget exceeded after sentiment/web but before strategy."""
        call_count = [0]

        def usage_side_effect():
            # Returns increasing token counts to simulate accumulation.
            # Call order: interview(before,after)=2, stats(before,after)=2,
            # sentiment(budget,before,after)=3, web(budget,before,after)=3,
            # strategy(budget)=1.  Total through web = 10 calls.
            # We want strategy budget check (call 11) to exceed the limit.
            call_count[0] += 1
            tokens = 0 if call_count[0] <= 10 else 20_000
            return {
                "total_input_tokens": tokens,
                "total_output_tokens": 0,
                "total_calls": call_count[0],
                "estimated_cost_usd": 0.0,
            }

        mock_llm = MagicMock()
        mock_llm.get_usage_summary.side_effect = usage_side_effect
        deps = {
            "llm_client": mock_llm,
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher") as mock_web, \
                 patch("analyze.StrategyAnalyzer") as mock_strategy, \
                 patch("analyze.ReportGenerator"):

                mock_interview.return_value.run.return_value = {"business_type": "spa"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 2}
                mock_sentiment.return_value.analyze.return_value = {"overall_sentiment": "positive"}
                mock_web.return_value.research.return_value = {"rent_ranges": {}}

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    max_tokens=15_000,
                    **deps,
                )

        self.assertFalse(result["success"])
        self.assertEqual(result["failed_stage"], "strategy")
        self.assertIn("Token budget exceeded", result["error"])
        mock_strategy.return_value.analyze.assert_not_called()

    def test_no_max_tokens_runs_full_pipeline(self):
        """When max_tokens is None, pipeline runs to completion."""
        mock_llm = MagicMock()
        mock_llm.get_usage_summary.return_value = {
            "total_input_tokens": 999_999,
            "total_output_tokens": 999_999,
            "total_calls": 100,
            "estimated_cost_usd": 99.99,
        }
        deps = {
            "llm_client": mock_llm,
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher") as mock_web, \
                 patch("analyze.StrategyAnalyzer") as mock_strategy, \
                 patch("analyze.ReportGenerator") as mock_report:

                mock_interview.return_value.run.return_value = {"business_type": "spa"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 2}
                mock_sentiment.return_value.analyze.return_value = {"overall_sentiment": "positive"}
                mock_web.return_value.research.return_value = {"rent_ranges": {}}
                mock_strategy.return_value.analyze.return_value = {"market_saturation": {}}
                mock_report.return_value.generate.return_value = "# Report"

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    max_tokens=None,
                    **deps,
                )

        self.assertTrue(result["success"])
        self.assertEqual(set(result["completed_stages"]), set(STAGES))


class TestStageUsageTracking(unittest.TestCase):
    """Tests for per-stage token usage tracking."""

    def _run_with_incrementing_usage(self, output_dir):
        """Run the full pipeline with a mock that increments token counts per stage."""
        token_state = {"input": 0, "output": 0, "calls": 0, "cost": 0.0}

        def usage_side_effect():
            return {
                "total_input_tokens": token_state["input"],
                "total_output_tokens": token_state["output"],
                "total_calls": token_state["calls"],
                "estimated_cost_usd": token_state["cost"],
            }

        mock_llm = MagicMock()
        mock_llm.get_usage_summary.side_effect = usage_side_effect

        deps = {
            "llm_client": mock_llm,
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

        with patch("analyze.UserInterviewer") as mock_interview, \
             patch("analyze.StatsAnalyzer") as mock_stats, \
             patch("analyze.SentimentAnalyzer") as mock_sentiment, \
             patch("analyze.WebResearcher") as mock_web, \
             patch("analyze.StrategyAnalyzer") as mock_strategy, \
             patch("analyze.ReportGenerator") as mock_report:

            mock_interview.return_value.run.return_value = {"business_type": "spa"}
            mock_stats.return_value.analyze.return_value = {"total_businesses": 2}

            # LLM stages bump the counters
            def sentiment_effect(*args, **kwargs):
                token_state["input"] += 100
                token_state["output"] += 50
                token_state["calls"] += 1
                return {"overall_sentiment": "positive"}
            mock_sentiment.return_value.analyze.side_effect = sentiment_effect

            def web_effect(*args, **kwargs):
                token_state["input"] += 200
                token_state["output"] += 80
                token_state["calls"] += 2
                return {"rent_ranges": {}}
            mock_web.return_value.research.side_effect = web_effect

            def strategy_effect(*args, **kwargs):
                token_state["input"] += 300
                token_state["output"] += 120
                token_state["calls"] += 3
                return {"market_saturation": {}}
            mock_strategy.return_value.analyze.side_effect = strategy_effect

            def report_effect(*args, **kwargs):
                token_state["input"] += 400
                token_state["output"] += 160
                token_state["calls"] += 4
                return "# Report"
            mock_report.return_value.generate.side_effect = report_effect

            result = run_pipeline(
                businesses=[
                    {"place_id": "p1", "name": "Spa", "rating": 4.5, "reviews": []},
                ],
                output_dir=output_dir,
                **deps,
            )

        return result

    def test_stage_usage_key_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_with_incrementing_usage(tmpdir)
        self.assertIn("stage_usage", result)

    def test_stage_usage_has_all_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_with_incrementing_usage(tmpdir)
        stage_usage = result["stage_usage"]
        for s in STAGES:
            self.assertIn(s, stage_usage, f"Missing stage_usage for: {s}")

    def test_stage_usage_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_with_incrementing_usage(tmpdir)
        for s, su in result["stage_usage"].items():
            self.assertIn("input_tokens", su, f"Missing input_tokens for stage: {s}")
            self.assertIn("output_tokens", su, f"Missing output_tokens for stage: {s}")
            self.assertIn("cost_usd", su, f"Missing cost_usd for stage: {s}")

    def test_non_llm_stages_have_zero_tokens(self):
        """interview and stats don't use LLM, so their delta should be 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_with_incrementing_usage(tmpdir)
        su = result["stage_usage"]
        self.assertEqual(su["interview"]["input_tokens"], 0)
        self.assertEqual(su["interview"]["output_tokens"], 0)
        self.assertEqual(su["stats"]["input_tokens"], 0)
        self.assertEqual(su["stats"]["output_tokens"], 0)

    def test_llm_stages_have_nonzero_tokens(self):
        """sentiment, web, strategy, report each consume tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_with_incrementing_usage(tmpdir)
        su = result["stage_usage"]
        self.assertEqual(su["sentiment"]["input_tokens"], 100)
        self.assertEqual(su["sentiment"]["output_tokens"], 50)
        self.assertEqual(su["web"]["input_tokens"], 200)
        self.assertEqual(su["web"]["output_tokens"], 80)
        self.assertEqual(su["strategy"]["input_tokens"], 300)
        self.assertEqual(su["strategy"]["output_tokens"], 120)
        self.assertEqual(su["report"]["input_tokens"], 400)
        self.assertEqual(su["report"]["output_tokens"], 160)

    def test_stage_usage_preserved_on_failure(self):
        """stage_usage is included in the result even when pipeline fails."""
        mock_llm = MagicMock()
        mock_llm.get_usage_summary.return_value = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "estimated_cost_usd": 0.0,
        }
        deps = {
            "llm_client": mock_llm,
            "prompt_engine": MagicMock(),
            "web_searcher": MagicMock(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("analyze.UserInterviewer") as mock_interview, \
                 patch("analyze.StatsAnalyzer") as mock_stats, \
                 patch("analyze.SentimentAnalyzer") as mock_sentiment, \
                 patch("analyze.WebResearcher"), \
                 patch("analyze.StrategyAnalyzer"), \
                 patch("analyze.ReportGenerator"):

                mock_interview.return_value.run.return_value = {"business_type": "spa"}
                mock_stats.return_value.analyze.return_value = {"total_businesses": 2}
                mock_sentiment.return_value.analyze.side_effect = RuntimeError("API error")

                result = run_pipeline(
                    businesses=list(SAMPLE_BUSINESSES),
                    output_dir=tmpdir,
                    **deps,
                )

        self.assertFalse(result["success"])
        self.assertIn("stage_usage", result)
        # interview and stats completed before failure
        self.assertIn("interview", result["stage_usage"])
        self.assertIn("stats", result["stage_usage"])


if __name__ == "__main__":
    unittest.main()

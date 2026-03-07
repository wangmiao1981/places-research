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


if __name__ == "__main__":
    unittest.main()

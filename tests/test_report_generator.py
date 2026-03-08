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

"""Tests for report_generator.py — Markdown report generation."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from report_generator import ReportGenerator

SAMPLE_REPORT = """# Market Analysis Report

## Executive Summary
This report analyzes the massage therapy market in San Jose 95125.

## Market Overview
Based on analysis of 50 businesses in the area...

## Competitive Landscape
The market shows moderate saturation...

## Customer Sentiment Insights
Customers value friendly staff and clean facilities...

## Location Analysis & Recommendation
The 95125 zip code offers good foot traffic...

## Strategic Recommendations
1. Focus on sports recovery niche
2. Offer online booking

## Risk Assessment
Key risks include market saturation...

## Appendix
Data collected from Google Places API.
"""


def _make_mocks():
    llm = MagicMock()
    llm.call.return_value = SAMPLE_REPORT
    prompts = MagicMock()
    prompts.render.return_value = "rendered prompt"
    return llm, prompts


def _sample_artifacts():
    return {
        "user_profile": {"business_type": "massage", "budget_range": {"min": 50000, "max": 100000}},
        "stats_report": {"total_businesses": 50, "rating_stats": {"mean": 4.2}},
        "sentiment_report": {"overall_sentiment": "positive", "positive_themes": []},
        "web_research": {"rent_ranges": {"average": "$2500/mo"}},
        "strategy": {"market_saturation": {"level": "moderate"}},
    }


class TestReportGeneratorGenerate(unittest.TestCase):

    def test_generate_returns_string(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        result = gen.generate("massage", "San Jose 95125", **arts)
        self.assertIsInstance(result, str)

    def test_generate_passes_all_artifacts_to_prompt(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        gen.generate("massage", "San Jose 95125", **arts)
        prompts.render.assert_called_once()
        args, kwargs = prompts.render.call_args
        self.assertEqual(args[0], "report_generation")
        variables = args[1]
        self.assertEqual(variables["business_type"], "massage")
        self.assertEqual(variables["location"], "San Jose 95125")
        # All artifact dicts should be JSON-serialized strings
        self.assertIn("total_businesses", variables["stats_report"])
        self.assertIn("overall_sentiment", variables["sentiment_report"])

    def test_generate_uses_max_tokens_8192(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        gen.generate("massage", "San Jose 95125", **arts)
        _, kwargs = llm.call.call_args
        self.assertEqual(kwargs["max_tokens"], 8192)

    def test_generate_returns_llm_output_directly(self):
        llm, prompts = _make_mocks()
        llm.call.return_value = "# My Report\nContent here."
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        result = gen.generate("massage", "San Jose 95125", **arts)
        self.assertEqual(result, "# My Report\nContent here.")

    def test_generate_with_empty_dicts(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        result = gen.generate("massage", "San Jose", {}, {}, {}, {}, {})
        self.assertIsInstance(result, str)
        # Should still call LLM
        llm.call.assert_called_once()

    def test_artifacts_serialized_as_json(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        gen.generate("massage", "San Jose 95125", **arts)
        variables = prompts.render.call_args[0][1]
        # Verify each artifact is valid JSON
        for key in ("user_profile", "stats_report", "sentiment_report", "web_research", "strategy"):
            parsed = json.loads(variables[key])
            self.assertIsInstance(parsed, dict)


class TestReportGeneratorSave(unittest.TestCase):

    def test_save_report_writes_markdown(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            gen.save_report("# Report\nContent", path)
            with open(path) as f:
                content = f.read()
            self.assertEqual(content, "# Report\nContent")
        finally:
            os.unlink(path)

    def test_save_report_not_json(self):
        """save_report writes plain text, not JSON."""
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            gen.save_report("# Report", path)
            with open(path) as f:
                content = f.read()
            # Should NOT be valid JSON
            with self.assertRaises(json.JSONDecodeError):
                json.loads(content)
        finally:
            os.unlink(path)

    def test_save_report_creates_parent_dirs(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "report.md")
            gen.save_report("# Report", path)
            self.assertTrue(os.path.exists(path))


class TestGenerateWithMissingData(unittest.TestCase):

    def test_missing_artifacts_use_empty_dicts(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        result = gen.generate_with_missing_data("massage", "San Jose")
        self.assertIsInstance(result, str)
        llm.call.assert_called_once()

    def test_partial_artifacts(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        result = gen.generate_with_missing_data(
            "massage", "San Jose",
            stats_report={"total": 50},
        )
        self.assertIsInstance(result, str)
        variables = prompts.render.call_args[0][1]
        # stats_report should have real data
        self.assertIn("50", variables["stats_report"])
        # others should be empty dicts
        self.assertEqual(json.loads(variables["sentiment_report"]), {})

    def test_all_artifacts_provided(self):
        llm, prompts = _make_mocks()
        gen = ReportGenerator(llm, prompts)
        arts = _sample_artifacts()
        result = gen.generate_with_missing_data("massage", "San Jose", **arts)
        self.assertIsInstance(result, str)


class TestReportContent(unittest.TestCase):

    def test_sample_report_has_required_sections(self):
        """The mock report should contain all required sections."""
        self.assertIn("Executive Summary", SAMPLE_REPORT)
        self.assertIn("Market Overview", SAMPLE_REPORT)
        self.assertIn("Competitive Landscape", SAMPLE_REPORT)
        self.assertIn("Customer Sentiment", SAMPLE_REPORT)
        self.assertIn("Location Analysis", SAMPLE_REPORT)
        self.assertIn("Strategic Recommendations", SAMPLE_REPORT)
        self.assertIn("Risk Assessment", SAMPLE_REPORT)
        self.assertIn("Appendix", SAMPLE_REPORT)


if __name__ == "__main__":
    unittest.main()

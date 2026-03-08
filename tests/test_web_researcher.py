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

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from web_searcher import SearchResult


VALID_LLM_RESPONSE = json.dumps({
    "rent_ranges": {"low": "$2000/mo", "high": "$5000/mo", "average": "$3500/mo", "source": "web"},
    "zoning_info": ["Commercial C1 zone required", "Health permit needed"],
    "industry_trends": [{"trend": "Growing demand for wellness", "source": "industry report"}],
    "demographics": {"median_age": 35, "foot_traffic": "high"},
    "competitor_online_presence": [
        {"name": "Rival Spa", "platforms": ["Yelp", "Google"], "rating": 4.5}
    ],
})


def _make_mocks():
    llm = MagicMock()
    llm.call.return_value = VALID_LLM_RESPONSE

    searcher = MagicMock()
    searcher.search_multiple.return_value = {
        "massage therapy commercial rent San Jose": [
            SearchResult(title="T", url="http://u.com", content="c", score=0.9)
        ]
    }

    prompt = MagicMock()
    prompt.render.return_value = "rendered prompt"

    return llm, searcher, prompt


class TestGenerateQueries(unittest.TestCase):
    def setUp(self):
        from web_researcher import WebResearcher
        self.llm, self.searcher, self.prompt = _make_mocks()
        self.researcher = WebResearcher(self.llm, self.searcher, self.prompt)

    def test_generates_five_queries(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertEqual(len(queries), 5)

    def test_query_contains_business_type_and_location(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        combined = " ".join(queries)
        self.assertIn("massage therapy", combined)
        self.assertIn("San Jose", combined)

    def test_commercial_rent_query(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertTrue(any("commercial rent" in q for q in queries))

    def test_zoning_query(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertTrue(any("zoning" in q for q in queries))

    def test_industry_trends_query(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertTrue(any("industry trends" in q for q in queries))

    def test_competition_query(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertTrue(any("reviews" in q for q in queries))

    def test_demographics_query(self):
        queries = self.researcher._generate_queries("massage therapy", "San Jose")
        self.assertTrue(any("demographics" in q for q in queries))

    def test_returns_list_of_strings(self):
        queries = self.researcher._generate_queries("coffee shop", "Austin")
        self.assertIsInstance(queries, list)
        for q in queries:
            self.assertIsInstance(q, str)


class TestFormatResults(unittest.TestCase):
    def setUp(self):
        from web_researcher import WebResearcher
        self.llm, self.searcher, self.prompt = _make_mocks()
        self.researcher = WebResearcher(self.llm, self.searcher, self.prompt)

    def test_format_includes_query(self):
        results = {
            "massage therapy rent": [
                SearchResult(title="Title A", url="http://a.com", content="body a", score=0.9)
            ]
        }
        text = self.researcher._format_results(results)
        self.assertIn("massage therapy rent", text)

    def test_format_includes_title_and_content(self):
        results = {
            "query": [
                SearchResult(title="My Title", url="http://x.com", content="My Content", score=0.8)
            ]
        }
        text = self.researcher._format_results(results)
        self.assertIn("My Title", text)
        self.assertIn("My Content", text)

    def test_format_empty_results(self):
        text = self.researcher._format_results({})
        self.assertIsInstance(text, str)

    def test_format_empty_result_list(self):
        text = self.researcher._format_results({"no results query": []})
        self.assertIsInstance(text, str)
        self.assertIn("no results query", text)

    def test_format_multiple_queries(self):
        results = {
            "query one": [SearchResult(title="T1", url="http://a.com", content="c1", score=0.9)],
            "query two": [SearchResult(title="T2", url="http://b.com", content="c2", score=0.7)],
        }
        text = self.researcher._format_results(results)
        self.assertIn("query one", text)
        self.assertIn("query two", text)


class TestResearchFlow(unittest.TestCase):
    def setUp(self):
        from web_researcher import WebResearcher
        self.llm, self.searcher, self.prompt = _make_mocks()
        self.researcher = WebResearcher(self.llm, self.searcher, self.prompt)

    def test_returns_dict(self):
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertIsInstance(result, dict)

    def test_result_has_required_keys(self):
        result = self.researcher.research("massage therapy", "San Jose")
        for key in ("rent_ranges", "zoning_info", "industry_trends", "demographics",
                    "competitor_online_presence", "queries_executed", "raw_result_count"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_calls_search_multiple(self):
        self.researcher.research("massage therapy", "San Jose")
        self.searcher.search_multiple.assert_called_once()

    def test_calls_prompt_render(self):
        self.researcher.research("massage therapy", "San Jose")
        self.prompt.render.assert_called_once()

    def test_calls_llm(self):
        self.researcher.research("massage therapy", "San Jose")
        self.llm.call.assert_called_once()

    def test_queries_executed_count(self):
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertEqual(result["queries_executed"], 5)

    def test_raw_result_count_tracked(self):
        self.searcher.search_multiple.return_value = {
            "q1": [SearchResult(title="T", url="http://u.com", content="c", score=0.9)],
            "q2": [
                SearchResult(title="T2", url="http://u2.com", content="c2", score=0.8),
                SearchResult(title="T3", url="http://u3.com", content="c3", score=0.7),
            ],
        }
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertEqual(result["raw_result_count"], 3)

    def test_prompt_render_receives_correct_variables(self):
        self.researcher.research("massage therapy", "San Jose")
        call_kwargs = self.prompt.render.call_args
        # first positional arg is template name
        self.assertEqual(call_kwargs[0][0], "web_research_synthesis")
        variables = call_kwargs[0][1]
        self.assertIn("business_type", variables)
        self.assertIn("location", variables)
        self.assertIn("search_results", variables)
        self.assertIn("research_focus", variables)

    def test_rent_ranges_from_llm(self):
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertEqual(result["rent_ranges"]["average"], "$3500/mo")

    def test_competitor_online_presence_from_llm(self):
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertEqual(result["competitor_online_presence"][0]["name"], "Rival Spa")


class TestEmptySearchResults(unittest.TestCase):
    def setUp(self):
        from web_researcher import WebResearcher
        self.llm, self.searcher, self.prompt = _make_mocks()
        self.researcher = WebResearcher(self.llm, self.searcher, self.prompt)

    def test_handles_empty_search_results(self):
        self.searcher.search_multiple.return_value = {}
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["raw_result_count"], 0)

    def test_handles_all_empty_result_lists(self):
        self.searcher.search_multiple.return_value = {
            "q1": [], "q2": [], "q3": [], "q4": [], "q5": []
        }
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertEqual(result["raw_result_count"], 0)


class TestMalformedLLMResponse(unittest.TestCase):
    def setUp(self):
        from web_researcher import WebResearcher
        self.llm, self.searcher, self.prompt = _make_mocks()
        self.researcher = WebResearcher(self.llm, self.searcher, self.prompt)

    def test_handles_malformed_json(self):
        self.llm.call.return_value = "this is not json {"
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertIsInstance(result, dict)
        # Should still have the tracking keys
        self.assertIn("queries_executed", result)
        self.assertIn("raw_result_count", result)

    def test_malformed_json_has_empty_fields(self):
        self.llm.call.return_value = "not json"
        result = self.researcher.research("massage therapy", "San Jose")
        # LLM parsed fields should be empty/default when JSON fails
        self.assertIn("rent_ranges", result)
        self.assertIn("zoning_info", result)

    def test_handles_empty_llm_response(self):
        self.llm.call.return_value = ""
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertIsInstance(result, dict)

    def test_handles_partial_json_object(self):
        self.llm.call.return_value = '{"rent_ranges": {"low": "$1000"}}'
        result = self.researcher.research("massage therapy", "San Jose")
        self.assertIsInstance(result, dict)
        self.assertIn("rent_ranges", result)


class TestSaveReport(unittest.TestCase):
    def test_save_report_creates_file(self):
        from web_researcher import WebResearcher
        llm, searcher, prompt = _make_mocks()
        researcher = WebResearcher(llm, searcher, prompt)
        report = {"rent_ranges": {}, "zoning_info": [], "queries_executed": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            researcher.save_report(report, path)
            self.assertTrue(os.path.exists(path))

    def test_save_report_valid_json(self):
        from web_researcher import WebResearcher
        llm, searcher, prompt = _make_mocks()
        researcher = WebResearcher(llm, searcher, prompt)
        report = {"rent_ranges": {"low": "$1000"}, "queries_executed": 5}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            researcher.save_report(report, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["queries_executed"], 5)

    def test_save_report_roundtrip(self):
        from web_researcher import WebResearcher
        llm, searcher, prompt = _make_mocks()
        researcher = WebResearcher(llm, searcher, prompt)
        report = {
            "rent_ranges": {"low": "$2000", "high": "$5000", "average": "$3500", "source": "web"},
            "zoning_info": ["C1 zone"],
            "industry_trends": [{"trend": "wellness boom", "source": "report"}],
            "demographics": {"median_age": 35},
            "competitor_online_presence": [],
            "queries_executed": 5,
            "raw_result_count": 10,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "full_report.json")
            researcher.save_report(report, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, report)


if __name__ == "__main__":
    unittest.main()

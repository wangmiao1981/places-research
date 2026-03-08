import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

VALID_LLM_RESPONSE = json.dumps({
    "market_saturation": {"level": "moderate", "reasoning": "Several competitors exist"},
    "competitor_tiers": {"leaders": ["Alpha Spa"], "mid_tier": ["Beta Massage"], "vulnerable": ["Gamma Touch"]},
    "service_gaps": [{"gap": "Late night service", "opportunity_size": "large", "evidence": "No competitors open after 10pm"}],
    "differentiation_recommendations": [{"recommendation": "Offer mobile service", "rationale": "Untapped market", "difficulty": "medium"}],
    "location_recommendations": [{"area": "Downtown", "reasoning": "High foot traffic", "risk_level": "low"}],
    "risk_factors": [{"risk": "High competition", "mitigation": "Differentiate on quality", "severity": "medium"}],
})


def _make_mocks(llm_response=VALID_LLM_RESPONSE):
    llm = MagicMock()
    llm.call.return_value = llm_response
    prompt = MagicMock()
    prompt.render.return_value = "rendered prompt text"
    return llm, prompt


class TestStrategyAnalyzerInit(unittest.TestCase):
    def test_init_stores_clients(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        self.assertIs(analyzer.llm_client, llm)
        self.assertIs(analyzer.prompt_engine, prompt)


class TestAnalyzePassesArtifactsToPrompt(unittest.TestCase):
    def setUp(self):
        from strategy_analyzer import StrategyAnalyzer
        self.llm, self.prompt = _make_mocks()
        self.analyzer = StrategyAnalyzer(self.llm, self.prompt)
        self.user_profile = {"name": "Jane", "budget": 50000}
        self.stats_report = {"total_businesses": 10}
        self.sentiment_report = {"average_sentiment": "positive"}
        self.web_research = {"top_result": "massage trends"}

    def test_render_called_with_correct_template(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        call_args = self.prompt.render.call_args
        self.assertEqual(call_args[0][0], "competitive_strategy")

    def test_render_called_with_business_type(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        call_kwargs = self.prompt.render.call_args[1]
        self.assertEqual(call_kwargs.get("business_type"), "massage")

    def test_render_variables_contain_all_fields(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        variables = self.prompt.render.call_args[0][1]
        self.assertIn("business_type", variables)
        self.assertIn("location", variables)
        self.assertIn("user_profile", variables)
        self.assertIn("stats_report", variables)
        self.assertIn("sentiment_report", variables)
        self.assertIn("web_research", variables)

    def test_render_variables_serialize_dicts_as_json(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        variables = self.prompt.render.call_args[0][1]
        # Should be JSON strings, not raw dicts
        self.assertEqual(variables["user_profile"], json.dumps(self.user_profile))
        self.assertEqual(variables["stats_report"], json.dumps(self.stats_report))
        self.assertEqual(variables["sentiment_report"], json.dumps(self.sentiment_report))
        self.assertEqual(variables["web_research"], json.dumps(self.web_research))

    def test_render_variables_contain_correct_location(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        variables = self.prompt.render.call_args[0][1]
        self.assertEqual(variables["location"], "San Jose, CA")

    def test_llm_call_uses_rendered_prompt(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        call_args = self.llm.call.call_args
        # user may be passed as positional arg[1] or keyword arg "user"
        user_value = call_args[1].get("user") if call_args[1].get("user") else (call_args[0][1] if len(call_args[0]) > 1 else None)
        self.assertEqual(user_value, "rendered prompt text")

    def test_llm_call_max_tokens_is_4096(self):
        self.analyzer.analyze(
            "massage", "San Jose, CA",
            self.user_profile, self.stats_report,
            self.sentiment_report, self.web_research,
        )
        call_kwargs = self.llm.call.call_args[1]
        self.assertEqual(call_kwargs.get("max_tokens"), 4096)


class TestAnalyzeReturnsCorrectSchema(unittest.TestCase):
    def setUp(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        self.analyzer = StrategyAnalyzer(llm, prompt)

    def _run(self):
        return self.analyzer.analyze(
            "massage", "San Jose, CA",
            {}, {}, {}, {},
        )

    def test_returns_dict(self):
        result = self._run()
        self.assertIsInstance(result, dict)

    def test_market_saturation_key(self):
        result = self._run()
        self.assertIn("market_saturation", result)
        self.assertIn("level", result["market_saturation"])
        self.assertIn("reasoning", result["market_saturation"])

    def test_competitor_tiers_key(self):
        result = self._run()
        self.assertIn("competitor_tiers", result)
        self.assertIn("leaders", result["competitor_tiers"])
        self.assertIn("mid_tier", result["competitor_tiers"])
        self.assertIn("vulnerable", result["competitor_tiers"])

    def test_service_gaps_key(self):
        result = self._run()
        self.assertIn("service_gaps", result)
        self.assertIsInstance(result["service_gaps"], list)

    def test_differentiation_recommendations_key(self):
        result = self._run()
        self.assertIn("differentiation_recommendations", result)
        self.assertIsInstance(result["differentiation_recommendations"], list)

    def test_location_recommendations_key(self):
        result = self._run()
        self.assertIn("location_recommendations", result)
        self.assertIsInstance(result["location_recommendations"], list)

    def test_risk_factors_key(self):
        result = self._run()
        self.assertIn("risk_factors", result)
        self.assertIsInstance(result["risk_factors"], list)


class TestAnalyzeEmptyInputDicts(unittest.TestCase):
    def test_empty_dicts_do_not_raise(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        result = analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        self.assertIn("market_saturation", result)

    def test_minimal_dicts_serialized_correctly(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        variables = prompt.render.call_args[0][1]
        self.assertEqual(variables["user_profile"], "{}")
        self.assertEqual(variables["stats_report"], "{}")


class TestAnalyzeMalformedJsonResponse(unittest.TestCase):
    def test_malformed_json_returns_empty_dict(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks(llm_response="this is not json {{{")
        analyzer = StrategyAnalyzer(llm, prompt)
        result = analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})

    def test_empty_string_response_returns_empty_dict(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks(llm_response="")
        analyzer = StrategyAnalyzer(llm, prompt)
        result = analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {})


class TestSaveReport(unittest.TestCase):
    def test_save_creates_valid_json(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        report = {"market_saturation": {"level": "low", "reasoning": "few players"}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            analyzer.save_report(report, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["market_saturation"]["level"], "low")
        finally:
            os.unlink(path)

    def test_save_creates_parent_dirs(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        report = {"risk_factors": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "report.json")
            analyzer.save_report(report, path)
            self.assertTrue(os.path.exists(path))

    def test_save_roundtrip_full_report(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        report = json.loads(VALID_LLM_RESPONSE)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            analyzer.save_report(report, path)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, report)
        finally:
            os.unlink(path)


class TestBusinessTypePassedToRender(unittest.TestCase):
    def test_business_type_kwarg_passed_to_render(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        analyzer.analyze("nail_salon", "Austin, TX", {}, {}, {}, {})
        call_kwargs = prompt.render.call_args[1]
        self.assertEqual(call_kwargs.get("business_type"), "nail_salon")

    def test_different_business_types_passed_through(self):
        from strategy_analyzer import StrategyAnalyzer
        for btype in ["massage", "coffee_shop", "gym"]:
            llm, prompt = _make_mocks()
            analyzer = StrategyAnalyzer(llm, prompt)
            analyzer.analyze(btype, "Seattle, WA", {}, {}, {}, {})
            call_kwargs = prompt.render.call_args[1]
            self.assertEqual(call_kwargs.get("business_type"), btype)


class TestMaxTokensHigherThanDefault(unittest.TestCase):
    def test_max_tokens_is_4096(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        call_kwargs = self.llm.call.call_args[1] if hasattr(self, 'llm') else llm.call.call_args[1]
        self.assertEqual(call_kwargs.get("max_tokens"), 4096)

    def test_max_tokens_greater_than_1024_default(self):
        from strategy_analyzer import StrategyAnalyzer
        llm, prompt = _make_mocks()
        analyzer = StrategyAnalyzer(llm, prompt)
        analyzer.analyze("massage", "San Jose, CA", {}, {}, {}, {})
        call_kwargs = llm.call.call_args[1]
        self.assertGreater(call_kwargs.get("max_tokens"), 1024)


if __name__ == "__main__":
    unittest.main()

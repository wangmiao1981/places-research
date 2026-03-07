"""
Strategy Analyzer — synthesizes all research data into strategic recommendations.

Uses the competitive_strategy prompt template to generate structured JSON output
via an LLM call, combining stats, sentiment, web research, and user profile data.
"""

import json
from pathlib import Path

from llm_client import LLMClient
from prompt_engine import PromptEngine


class StrategyAnalyzer:
    """Synthesize all research artifacts into strategic recommendations."""

    def __init__(self, llm_client: LLMClient, prompt_engine: PromptEngine) -> None:
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine

    def analyze(
        self,
        business_type: str,
        location: str,
        user_profile: dict,
        stats_report: dict,
        sentiment_report: dict,
        web_research: dict,
    ) -> dict:
        """Synthesize all data into strategic recommendations.

        Args:
            business_type: Type of business (e.g. "massage", "coffee_shop").
            location: Target location string (e.g. "San Jose, CA").
            user_profile: Dict describing the user's business profile/goals.
            stats_report: Dict with statistical analysis of local competitors.
            sentiment_report: Dict with sentiment analysis of competitor reviews.
            web_research: Dict with web research findings.

        Returns:
            Structured dict with keys: market_saturation, competitor_tiers,
            service_gaps, differentiation_recommendations,
            location_recommendations, risk_factors.
            Returns empty dict if the LLM returns malformed JSON.
        """
        variables = {
            "business_type": business_type,
            "location": location,
            "user_profile": json.dumps(user_profile),
            "stats_report": json.dumps(stats_report),
            "sentiment_report": json.dumps(sentiment_report),
            "web_research": json.dumps(web_research),
        }

        prompt = self.prompt_engine.render(
            "competitive_strategy",
            variables,
            business_type=business_type,
        )

        response = self.llm_client.call(
            system="You are a strategic business analyst. Respond only with valid JSON.",
            user=prompt,
            max_tokens=4096,
        )

        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return {}

    def save_report(self, report: dict, path: str) -> None:
        """Save the strategy report as a JSON file.

        Args:
            report: The strategy report dict to save.
            path: Absolute path to the output JSON file. Parent directories
                  are created automatically if they do not exist.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))

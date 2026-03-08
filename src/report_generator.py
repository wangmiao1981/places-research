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

"""
report_generator.py — Assemble all analysis artifacts into a polished Markdown report.

Uses Claude to generate a professional market analysis report from stats,
sentiment analysis, web research, and strategy artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ReportGenerator:
    """Generate a Markdown market analysis report from all pipeline artifacts."""

    def __init__(self, llm_client, prompt_engine) -> None:
        self._llm = llm_client
        self._prompts = prompt_engine

    def generate(
        self,
        business_type: str,
        location: str,
        user_profile: dict,
        stats_report: dict,
        sentiment_report: dict,
        web_research: dict,
        strategy: dict,
    ) -> str:
        """Generate a polished Markdown report from all artifacts."""
        prompt = self._prompts.render(
            "report_generation",
            {
                "business_type": business_type,
                "location": location,
                "user_profile": json.dumps(user_profile, indent=2),
                "stats_report": json.dumps(stats_report, indent=2),
                "sentiment_report": json.dumps(sentiment_report, indent=2),
                "web_research": json.dumps(web_research, indent=2),
                "strategy": json.dumps(strategy, indent=2),
            },
        )
        return self._llm.call(
            system="You are a professional business analyst. Write clear, actionable reports.",
            user=prompt,
            max_tokens=8192,
        )

    def generate_with_missing_data(
        self,
        business_type: str,
        location: str,
        user_profile: Optional[dict] = None,
        stats_report: Optional[dict] = None,
        sentiment_report: Optional[dict] = None,
        web_research: Optional[dict] = None,
        strategy: Optional[dict] = None,
    ) -> str:
        """Generate report even when some artifacts are missing."""
        return self.generate(
            business_type=business_type,
            location=location,
            user_profile=user_profile if user_profile is not None else {},
            stats_report=stats_report if stats_report is not None else {},
            sentiment_report=sentiment_report if sentiment_report is not None else {},
            web_research=web_research if web_research is not None else {},
            strategy=strategy if strategy is not None else {},
        )

    def save_report(self, report: str, path: str) -> None:
        """Save Markdown report to file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")

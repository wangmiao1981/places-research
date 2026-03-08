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
web_researcher.py — Web-based market intelligence gatherer and synthesizer.

Uses WebSearcher to fetch search results across multiple market research queries,
then synthesizes them with an LLM via PromptEngine.
"""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

from llm_client import LLMClient
from prompt_engine import PromptEngine
from web_searcher import SearchResult, WebSearcher


class WebResearcher:
    """Gather and synthesize web research for market intelligence."""

    def __init__(
        self,
        llm_client: LLMClient,
        web_searcher: WebSearcher,
        prompt_engine: PromptEngine,
    ) -> None:
        self.llm_client = llm_client
        self.web_searcher = web_searcher
        self.prompt_engine = prompt_engine

    def research(
        self,
        business_type: str,
        location: str,
        user_profile: Optional[dict] = None,
    ) -> dict:
        """Gather and synthesize web research for the market.

        Args:
            business_type: Type of business (e.g. "massage therapy").
            location: Target location (e.g. "San Jose").
            user_profile: Optional user profile dict (reserved for future use).

        Returns:
            Structured dict with: rent_ranges, zoning_info, industry_trends,
            demographics, competitor_online_presence, queries_executed,
            raw_result_count.
        """
        queries = self._generate_queries(business_type, location)
        all_results: Dict[str, List[SearchResult]] = self.web_searcher.search_multiple(queries)

        raw_result_count = sum(len(v) for v in all_results.values())
        formatted = self._format_results(all_results)

        prompt = self.prompt_engine.render(
            "web_research_synthesis",
            {
                "business_type": business_type,
                "location": location,
                "search_results": formatted,
                "research_focus": (
                    f"Provide market intelligence for opening a {business_type} in {location}."
                ),
            },
        )

        raw_text = self.llm_client.call(
            system="You are a market research analyst. Respond only with valid JSON.",
            user=prompt,
            max_tokens=2048,
        )

        # Parse LLM JSON response; fall back to empty structure on malformed JSON.
        parsed = self._parse_llm_response(raw_text)

        return {
            "rent_ranges": parsed.get("rent_ranges", {}),
            "zoning_info": parsed.get("zoning_info", []),
            "industry_trends": parsed.get("industry_trends", []),
            "demographics": parsed.get("demographics", {}),
            "competitor_online_presence": parsed.get("competitor_online_presence", []),
            "queries_executed": len(queries),
            "raw_result_count": raw_result_count,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def save_report(self, report: dict, path: str) -> None:
        """Save report as JSON to the given file path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_queries(self, business_type: str, location: str) -> List[str]:
        """Generate the five standard market research search queries."""
        return [
            f"{business_type} commercial rent {location}",
            f"{business_type} zoning regulations {location}",
            f"{business_type} industry trends {datetime.date.today().year}",
            f"best {business_type} {location} reviews",
            f"{location} demographics foot traffic",
        ]

    def _format_results(self, results: Dict[str, List[SearchResult]]) -> str:
        """Format search results dict into plain text for the LLM."""
        sections: List[str] = []
        for query, result_list in results.items():
            section_lines = [f"### Query: {query}"]
            if not result_list:
                section_lines.append("(no results)")
            else:
                for i, r in enumerate(result_list, start=1):
                    section_lines.append(f"{i}. {r.title}")
                    section_lines.append(f"   URL: {r.url}")
                    section_lines.append(f"   {r.content}")
            sections.append("\n".join(section_lines))
        return "\n\n".join(sections)

    @staticmethod
    def _parse_llm_response(text: str) -> dict:
        """Parse JSON from LLM response; return empty dict on failure."""
        if not text:
            return {}
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Remove first and last fence lines
            inner = lines[1:] if len(lines) > 1 else lines
            if inner and inner[-1].strip().startswith("```"):
                inner = inner[:-1]
            stripped = "\n".join(inner)
        try:
            result = json.loads(stripped)
            if not isinstance(result, dict):
                return {}
            return result
        except (json.JSONDecodeError, ValueError):
            return {}

"""
Sentiment Analyzer — uses Claude to analyze business reviews.
"""

import json
import re
from typing import Any, Dict, List

from llm_client import LLMClient
from prompt_engine import PromptEngine

REVIEWS_PER_BATCH = 20
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048

_EMPTY_RESULT = {
    "positive_themes": [],
    "negative_themes": [],
    "service_quality_patterns": [],
    "unmet_needs": [],
    "overall_sentiment": "unknown",
}


def _extract_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    # Strip ```json ... ``` or ``` ... ``` wrappers
    stripped = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # Try the original text as-is
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}


def _merge_results(results: List[dict]) -> dict:
    """Merge sentiment results from multiple batches."""
    if not results:
        return dict(_EMPTY_RESULT)
    if len(results) == 1:
        merged = dict(_EMPTY_RESULT)
        merged.update(results[0])
        return merged

    merged = {
        "positive_themes": [],
        "negative_themes": [],
        "service_quality_patterns": [],
        "unmet_needs": [],
        "overall_sentiment": "unknown",
    }
    sentiments = []
    for r in results:
        merged["positive_themes"].extend(r.get("positive_themes", []))
        merged["negative_themes"].extend(r.get("negative_themes", []))
        merged["service_quality_patterns"].extend(r.get("service_quality_patterns", []))
        merged["unmet_needs"].extend(r.get("unmet_needs", []))
        if r.get("overall_sentiment"):
            sentiments.append(r["overall_sentiment"])

    # Determine overall sentiment from batch sentiments
    if sentiments:
        if all(s == "positive" for s in sentiments):
            merged["overall_sentiment"] = "positive"
        elif all(s == "negative" for s in sentiments):
            merged["overall_sentiment"] = "negative"
        elif "unknown" not in sentiments:
            merged["overall_sentiment"] = "mixed"
        else:
            non_unknown = [s for s in sentiments if s != "unknown"]
            if non_unknown:
                merged["overall_sentiment"] = non_unknown[0] if len(set(non_unknown)) == 1 else "mixed"

    return merged


class SentimentAnalyzer:
    def __init__(self, llm_client: LLMClient, prompt_engine: PromptEngine):
        self._llm = llm_client
        self._prompt = prompt_engine

    def analyze(self, businesses: list, business_type: str, location: str) -> dict:
        """Analyze reviews from businesses and return a sentiment report."""
        businesses_without_reviews = 0
        all_reviews: List[str] = []
        businesses_with_reviews = 0

        for biz in businesses:
            reviews = biz.get("reviews") or []
            if not reviews:
                businesses_without_reviews += 1
            else:
                businesses_with_reviews += 1
                all_reviews.extend(reviews)

        if not all_reviews:
            result = dict(_EMPTY_RESULT)
            result["businesses_analyzed"] = 0
            result["businesses_without_reviews"] = businesses_without_reviews
            return result

        # Split reviews into batches
        batches = [
            all_reviews[i: i + REVIEWS_PER_BATCH]
            for i in range(0, len(all_reviews), REVIEWS_PER_BATCH)
        ]

        batch_results = []
        for batch in batches:
            reviews_data = "\n".join(f"- {r}" for r in batch)
            prompt = self._prompt.render(
                "sentiment_analysis",
                {
                    "business_type": business_type,
                    "location": location,
                    "reviews_data": reviews_data,
                },
                business_type=business_type,
            )
            raw = self._llm.call(
                system="You are a market research analyst. Respond with valid JSON only.",
                user=prompt,
                model=DEFAULT_MODEL,
                max_tokens=MAX_TOKENS,
            )
            parsed = _extract_json(raw)
            if parsed:
                batch_results.append(parsed)
            else:
                # Malformed response — append empty structure so schema is preserved
                batch_results.append(dict(_EMPTY_RESULT))

        merged = _merge_results(batch_results)
        merged["businesses_analyzed"] = businesses_with_reviews
        merged["businesses_without_reviews"] = businesses_without_reviews
        return merged

    def save_report(self, report: dict, path: str) -> None:
        """Save report as JSON to the given path."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

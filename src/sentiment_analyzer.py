"""
Sentiment Analyzer — uses Claude to analyze business reviews.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from llm_client import LLMClient
from prompt_engine import PromptEngine

REVIEWS_PER_BATCH = 20
MAX_TOKENS = 2048


def _make_empty_result() -> dict:
    """Return a new empty result dict with fresh lists each time."""
    return {
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
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # Try the original text as-is
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
    # LLM may return a JSON array — extract first dict element if so
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _merge_results(results: List[dict]) -> dict:
    """Merge sentiment results from multiple batches."""
    if not results:
        return _make_empty_result()

    merged = _make_empty_result()
    sentiments = []
    for r in results:
        merged["positive_themes"].extend(r.get("positive_themes") or [])
        merged["negative_themes"].extend(r.get("negative_themes") or [])
        merged["service_quality_patterns"].extend(r.get("service_quality_patterns") or [])
        merged["unmet_needs"].extend(r.get("unmet_needs") or [])
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
                for review in reviews:
                    if isinstance(review, dict):
                        all_reviews.append(review.get("text", ""))
                    else:
                        all_reviews.append(review)

        if not all_reviews:
            result = _make_empty_result()
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
                max_tokens=MAX_TOKENS,
            )
            parsed = _extract_json(raw)
            if parsed:
                batch_results.append(parsed)
            else:
                # Malformed response — append empty structure so schema is preserved
                batch_results.append(_make_empty_result())

        merged = _merge_results(batch_results)
        merged["businesses_analyzed"] = businesses_with_reviews
        merged["businesses_without_reviews"] = businesses_without_reviews
        return merged

    def save_report(self, report: dict, path: str) -> None:
        """Save report as JSON to the given path."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

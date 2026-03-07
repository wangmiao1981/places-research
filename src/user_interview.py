"""
user_interview.py — Interactive CLI questionnaire for collecting business context.

Collects user's business plan details to enrich competitive-analysis reports
produced by the grid-search pipeline.
"""

import json
import re
from collections import Counter
from typing import Optional

# Words to ignore when detecting business type from names
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "at", "by", "for",
    "to", "with", "from", "is", "it", "on", "be", "as", "that",
    "this", "was", "are", "were", "been", "have", "has", "had",
    "my", "your", "our", "their", "its",
    "best", "great", "good", "new", "top", "prime", "premier", "pro",
    "center", "place", "shop", "service", "services", "studio",
    "group", "inc", "llc", "co", "corp",
}

# Multiple-choice option maps
_CLIENTELE_OPTIONS = {
    "a": "General public",
    "b": "Professionals/office workers",
    "c": "Athletes/fitness enthusiasts",
    "d": "Seniors",
    "e": "Luxury/high-end clients",
    "f": "Budget-conscious clients",
    "g": "Other",
}

_LOCATION_OPTIONS = {
    "a": "High foot traffic",
    "b": "Low rent",
    "c": "Near complementary businesses",
    "d": "Ample parking",
    "e": "Residential area",
    "f": "Commercial/business district",
}


class UserInterviewer:
    """Interactive CLI that collects business context from the user."""

    def __init__(self, business_data: list) -> None:
        self._business_data = business_data
        self._detected_type: str = self._detect_business_type(business_data)
        self._data_source: Optional[str] = self._extract_data_source(business_data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_business_type(data: list) -> str:
        """Return the most common meaningful word across all business names."""
        if not data:
            return "unknown"

        word_counts: Counter = Counter()
        for entry in data:
            name = entry.get("name", "")
            words = re.findall(r"[a-z]+", name.lower())
            for word in words:
                if word not in _STOPWORDS and len(word) > 2:
                    word_counts[word] += 1

        if not word_counts:
            return "unknown"

        return word_counts.most_common(1)[0][0]

    @staticmethod
    def _extract_data_source(data: list) -> Optional[str]:
        """Pull _source_file from the first entry that has it."""
        for entry in data:
            src = entry.get("_source_file")
            if src:
                return src
        return None

    @staticmethod
    def _parse_budget(raw: str) -> Optional[dict]:
        """Parse 'MIN-MAX' (with optional spaces around dash) → dict or None."""
        raw = raw.strip()
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
        if not match:
            return None
        lo, hi = int(match.group(1)), int(match.group(2))
        if lo > hi:
            return None
        return {"min": lo, "max": hi}

    @staticmethod
    def _ask(prompt: str) -> str:
        """Thin wrapper around input() to ease mocking."""
        return input(prompt)

    # ------------------------------------------------------------------
    # Multiple-choice helpers
    # ------------------------------------------------------------------

    def _ask_multiple_choice(self, prompt: str, options: dict) -> str:
        """Present a multiple-choice question; keep asking until valid answer."""
        option_lines = "\n".join(
            f"  ({k}) {v}" for k, v in options.items()
        )
        full_prompt = f"{prompt}\n{option_lines}\nYour choice: "

        while True:
            answer = self._ask(full_prompt).strip().lower()
            if answer in options:
                if answer == "g" and options.get("g") == "Other":
                    custom = self._ask("Please specify: ").strip()
                    return custom if custom else "Other"
                return options[answer]
            print("  Invalid choice. Please select one of the listed options.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run the interactive interview and return a user-profile dict."""

        # ── Q1: Business type confirmation ──────────────────────────────
        n = len(self._business_data)
        while True:
            confirm = self._ask(
                f"The data shows {n} {self._detected_type} businesses. "
                "Is this the type of business you plan to open? (y/n) "
            ).strip().lower()
            if confirm == "y":
                business_type = self._detected_type
                break
            elif confirm == "n":
                business_type = self._ask(
                    "What type of business do you plan to open? "
                ).strip()
                break
            else:
                print("  Please enter 'y' or 'n'.")

        # ── Q2: Planned services ────────────────────────────────────────
        raw_services = self._ask(
            "What specific services do you plan to offer? (comma-separated) "
        )
        planned_services = [s.strip() for s in raw_services.split(",") if s.strip()]

        # ── Q3: Target clientele ────────────────────────────────────────
        target_clientele = self._ask_multiple_choice(
            "Who is your target clientele?",
            _CLIENTELE_OPTIONS,
        )

        # ── Q4: Budget range ────────────────────────────────────────────
        while True:
            raw_budget = self._ask(
                "What is your estimated startup budget? (e.g., 50000-100000) "
            )
            budget_range = self._parse_budget(raw_budget)
            if budget_range is not None:
                break
            print(
                "  Invalid format. Please enter a numeric range like '50000-100000' "
                "(min must be ≤ max)."
            )

        # ── Q5: Differentiators ─────────────────────────────────────────
        differentiators = self._ask(
            "What will make your business different from competitors? "
            "(describe briefly) "
        )

        # ── Q6: Location preference ─────────────────────────────────────
        location_preference = self._ask_multiple_choice(
            "What matters most for your location?",
            _LOCATION_OPTIONS,
        )

        return {
            "business_type": business_type,
            "planned_services": planned_services,
            "target_clientele": target_clientele,
            "budget_range": budget_range,
            "differentiators": differentiators,
            "location_preference": location_preference,
            "market_size": len(self._business_data),
            "data_source": self._data_source,
        }

    def load_profile(self, path: str) -> dict:
        """Load an existing profile from a JSON file (non-interactive mode)."""
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save_profile(self, profile: dict, path: str) -> None:
        """Save a profile dict to a JSON file."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, ensure_ascii=False)

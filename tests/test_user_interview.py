"""
Tests for user_interview.py — Interactive CLI questionnaire.
Uses mocked stdin for non-interactive testing.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from user_interview import UserInterviewer


# ---------------------------------------------------------------------------
# Sample data shared across tests
# ---------------------------------------------------------------------------

SAMPLE_BUSINESS_DATA = [
    {"place_id": "1", "name": "Happy Massage Spa", "types": ["spa", "health"]},
    {"place_id": "2", "name": "Relaxing Massage Center", "types": ["spa"]},
    {"place_id": "3", "name": "Massage Therapy Studio", "types": ["health"]},
    {"place_id": "4", "name": "Deep Tissue Massage", "types": ["spa", "health"]},
    {"place_id": "5", "name": "Swedish Massage Place", "types": ["spa"]},
]

SAMPLE_BUSINESS_DATA_WITH_SOURCE = [
    {
        "place_id": "1",
        "name": "Happy Massage Spa",
        "types": ["spa"],
        "_source_file": "output/massage_san_jose_95125.json",
    }
]


# ===========================================================================
# Business-type detection
# ===========================================================================

class TestDetectBusinessType(unittest.TestCase):
    def test_detects_most_common_word_in_names(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        # "massage" appears in all 5 names → should be detected
        self.assertEqual(interviewer._detected_type, "massage")

    def test_detects_type_from_single_entry(self):
        data = [{"place_id": "x", "name": "Yoga Studio"}]
        interviewer = UserInterviewer(data)
        # name contains "yoga" and "studio"; either is acceptable
        self.assertIn(interviewer._detected_type, ["yoga", "studio"])

    def test_empty_data_returns_unknown(self):
        interviewer = UserInterviewer([])
        self.assertEqual(interviewer._detected_type, "unknown")

    def test_all_stopwords_returns_unknown(self):
        # All words in names are stopwords or too short — should return "unknown"
        data = [{"place_id": "1", "name": "The Best Co"}, {"place_id": "2", "name": "A Co"}]
        interviewer = UserInterviewer(data)
        self.assertEqual(interviewer._detected_type, "unknown")

    def test_ignores_common_stopwords(self):
        data = [
            {"place_id": "1", "name": "The Best Barbershop"},
            {"place_id": "2", "name": "The Great Barbershop"},
            {"place_id": "3", "name": "The Premium Barbershop"},
        ]
        interviewer = UserInterviewer(data)
        self.assertEqual(interviewer._detected_type, "barbershop")

    def test_data_source_stored_on_instance(self):
        data = SAMPLE_BUSINESS_DATA_WITH_SOURCE
        interviewer = UserInterviewer(data)
        self.assertEqual(
            interviewer._data_source, "output/massage_san_jose_95125.json"
        )

    def test_no_data_source_field_is_none(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        self.assertIsNone(interviewer._data_source)


# ===========================================================================
# Full interview flow (happy path — user confirms detected type)
# ===========================================================================

class TestFullInterviewFlow(unittest.TestCase):
    def _run_with_inputs(self, inputs, data=None):
        if data is None:
            data = SAMPLE_BUSINESS_DATA
        interviewer = UserInterviewer(data)
        # Pin detected type for determinism
        interviewer._detected_type = "massage"
        with patch("builtins.input", side_effect=inputs):
            return interviewer.run()

    def test_full_happy_path_returns_correct_schema(self):
        inputs = [
            "y",                          # business type confirmed
            "Swedish, deep tissue",       # planned services
            "a",                          # target clientele → General public
            "50000-100000",               # budget range
            "Focus on sports recovery",   # differentiators
            "a",                          # location preference → High foot traffic
        ]
        profile = self._run_with_inputs(inputs)

        self.assertEqual(profile["business_type"], "massage")
        self.assertEqual(profile["planned_services"], ["Swedish", "deep tissue"])
        self.assertEqual(profile["target_clientele"], "General public")
        self.assertEqual(profile["budget_range"], {"min": 50000, "max": 100000})
        self.assertEqual(profile["differentiators"], "Focus on sports recovery")
        self.assertEqual(profile["location_preference"], "High foot traffic")
        self.assertEqual(profile["market_size"], len(SAMPLE_BUSINESS_DATA))
        self.assertIn("data_source", profile)

    def test_profile_contains_all_required_keys(self):
        inputs = ["y", "facials", "b", "20000-40000", "unique service", "c"]
        profile = self._run_with_inputs(inputs)
        required_keys = {
            "business_type",
            "planned_services",
            "target_clientele",
            "budget_range",
            "differentiators",
            "location_preference",
            "market_size",
            "data_source",
        }
        self.assertTrue(required_keys.issubset(set(profile.keys())))

    def test_market_size_equals_len_of_business_data(self):
        inputs = ["y", "nails", "c", "10000-30000", "friendly staff", "b"]
        profile = self._run_with_inputs(inputs, data=SAMPLE_BUSINESS_DATA)
        self.assertEqual(profile["market_size"], 5)


# ===========================================================================
# Business-type override (user says "n")
# ===========================================================================

class TestBusinessTypeOverride(unittest.TestCase):
    def _run_override(self, custom_type):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = [
            "n",           # reject detected type
            custom_type,   # provide correct type
            "Swedish",     # services
            "a",           # clientele
            "30000-60000", # budget
            "great location", # differentiators
            "d",           # location preference
        ]
        with patch("builtins.input", side_effect=inputs):
            return interviewer.run()

    def test_custom_type_used_in_profile(self):
        profile = self._run_override("yoga")
        self.assertEqual(profile["business_type"], "yoga")

    def test_override_does_not_affect_market_size(self):
        profile = self._run_override("pilates")
        self.assertEqual(profile["market_size"], len(SAMPLE_BUSINESS_DATA))


# ===========================================================================
# Budget validation
# ===========================================================================

class TestBudgetValidation(unittest.TestCase):
    def _run_budget_test(self, budget_inputs):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        # y=confirm, Swedish=services, a=clientele, <budget_inputs>, great service=differentiators, a=location
        prefix = ["y", "Swedish", "a"]
        suffix = ["great service", "a"]
        inputs = prefix + budget_inputs + suffix
        with patch("builtins.input", side_effect=inputs):
            return interviewer.run()

    def test_valid_budget_range_parsed(self):
        profile = self._run_budget_test(["50000-100000"])
        self.assertEqual(profile["budget_range"], {"min": 50000, "max": 100000})

    def test_invalid_then_valid_budget_retries(self):
        profile = self._run_budget_test(["not_a_range", "badformat", "20000-80000"])
        self.assertEqual(profile["budget_range"], {"min": 20000, "max": 80000})

    def test_budget_single_number_rejected(self):
        profile = self._run_budget_test(["50000", "10000-50000"])
        self.assertEqual(profile["budget_range"], {"min": 10000, "max": 50000})

    def test_budget_words_rejected(self):
        profile = self._run_budget_test(["low budget", "5000-15000"])
        self.assertEqual(profile["budget_range"], {"min": 5000, "max": 15000})

    def test_budget_with_spaces_around_dash_valid(self):
        profile = self._run_budget_test(["30000 - 70000"])
        self.assertEqual(profile["budget_range"], {"min": 30000, "max": 70000})


# ===========================================================================
# Multiple-choice selections
# ===========================================================================

class TestMultipleChoiceSelection(unittest.TestCase):
    CLIENTELE_MAP = {
        "a": "General public",
        "b": "Professionals/office workers",
        "c": "Athletes/fitness enthusiasts",
        "d": "Seniors",
        "e": "Luxury/high-end clients",
        "f": "Budget-conscious clients",
    }

    LOCATION_MAP = {
        "a": "High foot traffic",
        "b": "Low rent",
        "c": "Near complementary businesses",
        "d": "Ample parking",
        "e": "Residential area",
        "f": "Commercial/business district",
    }

    def _run_choices(self, clientele_choice, location_choice):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", clientele_choice, "50000-100000", "good service", location_choice]
        with patch("builtins.input", side_effect=inputs):
            return interviewer.run()

    def test_all_clientele_options(self):
        for letter, label in self.CLIENTELE_MAP.items():
            profile = self._run_choices(letter, "a")
            self.assertEqual(profile["target_clientele"], label, f"Failed for clientele option '{letter}'")

    def test_all_location_options(self):
        for letter, label in self.LOCATION_MAP.items():
            profile = self._run_choices("a", letter)
            self.assertEqual(profile["location_preference"], label, f"Failed for location option '{letter}'")

    def test_clientele_other_option(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "g", "Pet owners", "50000-100000", "good service", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["target_clientele"], "Pet owners")

    def test_invalid_clientele_choice_retries(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "z", "a", "50000-100000", "good service", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["target_clientele"], "General public")

    def test_invalid_location_choice_retries(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "a", "50000-100000", "good service", "x", "b"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["location_preference"], "Low rent")


# ===========================================================================
# Planned services parsing
# ===========================================================================

class TestPlannedServicesParsing(unittest.TestCase):
    def _run_services(self, services_input):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", services_input, "a", "50000-100000", "good service", "a"]
        with patch("builtins.input", side_effect=inputs):
            return interviewer.run()

    def test_comma_separated_services_parsed_to_list(self):
        profile = self._run_services("Swedish, deep tissue, hot stone")
        self.assertEqual(profile["planned_services"], ["Swedish", "deep tissue", "hot stone"])

    def test_single_service(self):
        profile = self._run_services("facial")
        self.assertEqual(profile["planned_services"], ["facial"])

    def test_services_stripped_of_whitespace(self):
        profile = self._run_services("  Swedish ,  deep tissue  ")
        self.assertEqual(profile["planned_services"], ["Swedish", "deep tissue"])

    def test_empty_services_handled(self):
        # Empty input should produce empty list or re-prompt; implementation may vary
        # Here we test that it returns a list (even if empty)
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "", "a", "50000-100000", "good service", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertIsInstance(profile["planned_services"], list)


# ===========================================================================
# Save / load profile
# ===========================================================================

class TestSaveLoadProfile(unittest.TestCase):
    def _sample_profile(self):
        return {
            "business_type": "massage",
            "planned_services": ["Swedish", "deep tissue"],
            "target_clientele": "General public",
            "budget_range": {"min": 50000, "max": 100000},
            "differentiators": "Focus on sports recovery",
            "location_preference": "High foot traffic",
            "market_size": 99,
            "data_source": "output/massage_san_jose_95125.json",
        }

    def test_save_and_load_roundtrip(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        profile = self._sample_profile()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            interviewer.save_profile(profile, tmp_path)
            loaded = interviewer.load_profile(tmp_path)
            self.assertEqual(loaded, profile)
        finally:
            os.unlink(tmp_path)

    def test_save_creates_valid_json(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        profile = self._sample_profile()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            interviewer.save_profile(profile, tmp_path)
            with open(tmp_path) as f:
                data = json.load(f)
            self.assertEqual(data["business_type"], "massage")
            self.assertEqual(data["budget_range"], {"min": 50000, "max": 100000})
        finally:
            os.unlink(tmp_path)

    def test_load_profile_returns_dict(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        profile = self._sample_profile()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(profile, f)
            tmp_path = f.name

        try:
            loaded = interviewer.load_profile(tmp_path)
            self.assertIsInstance(loaded, dict)
        finally:
            os.unlink(tmp_path)

    def test_load_profile_missing_file_raises(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        with self.assertRaises((FileNotFoundError, IOError)):
            interviewer.load_profile("/nonexistent/path/profile.json")


# ===========================================================================
# Output schema validation
# ===========================================================================

class TestOutputSchema(unittest.TestCase):
    def test_budget_range_is_dict_with_min_max(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertIsInstance(profile["budget_range"], dict)
        self.assertIn("min", profile["budget_range"])
        self.assertIn("max", profile["budget_range"])
        self.assertIsInstance(profile["budget_range"]["min"], int)
        self.assertIsInstance(profile["budget_range"]["max"], int)

    def test_planned_services_is_list(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish, deep tissue", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertIsInstance(profile["planned_services"], list)

    def test_market_size_is_int(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertIsInstance(profile["market_size"], int)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    def test_business_type_confirmation_case_insensitive(self):
        """Y and N (uppercase) should be accepted."""
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["Y", "Swedish", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["business_type"], "massage")

    def test_business_type_invalid_then_valid_retries(self):
        """Non-y/n input for confirmation should retry."""
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["maybe", "y", "Swedish", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["business_type"], "massage")

    def test_budget_min_greater_than_max_rejected(self):
        """100000-50000 (min > max) should be rejected and retried."""
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "a", "100000-50000", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["budget_range"], {"min": 50000, "max": 100000})

    def test_data_source_none_when_no_source_field(self):
        interviewer = UserInterviewer(SAMPLE_BUSINESS_DATA)
        inputs = ["y", "Swedish", "a", "50000-100000", "great", "a"]
        interviewer._detected_type = "massage"
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertIsNone(profile["data_source"])

    def test_data_source_from_first_entry_source_file(self):
        data = [{"place_id": "1", "name": "Massage X", "_source_file": "output/massage.json"}]
        interviewer = UserInterviewer(data)
        interviewer._detected_type = "massage"
        inputs = ["y", "Swedish", "a", "50000-100000", "great", "a"]
        with patch("builtins.input", side_effect=inputs):
            profile = interviewer.run()
        self.assertEqual(profile["data_source"], "output/massage.json")


if __name__ == "__main__":
    unittest.main()

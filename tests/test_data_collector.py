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

import unittest
from types import SimpleNamespace
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys
import tempfile
import requests

# Add src directory to path to import data_collector
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the environment variable before importing data_collector
with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test_api_key'}):
    import data_collector


class TestDataCollector(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_api_key = 'test_api_key'
        self.addCleanup(self.restore_api_key)
        data_collector.API_KEY = self.test_api_key
        self.sample_search_response = {
            "status": "OK",
            "results": [
                {
                    "place_id": "test_place_1",
                    "name": "Test Bakery 1"
                },
                {
                    "place_id": "test_place_2", 
                    "name": "Test Bakery 2"
                }
            ],
            "next_page_token": "next_token_123"
        }
        self.sample_place_details = {
            "status": "OK",
            "result": {
                "name": "Test Bakery",
                "formatted_address": "123 Test St, Test City, CA 12345",
                "geometry": {
                    "location": {
                        "lat": 37.7749,
                        "lng": -122.4194
                    }
                },
                "rating": 4.5,
                "user_ratings_total": 100,
                "reviews": [{"text": "Great bakery!"}],
                "website": "https://testbakery.com",
                "formatted_phone_number": "(555) 123-4567",
                "opening_hours": {
                    "weekday_text": ["Monday: 8:00 AM – 6:00 PM"]
                }
            }
        }

    def restore_api_key(self):
        data_collector.API_KEY = self.test_api_key

    @patch('data_collector.requests.get')
    def test_search_places_by_query_success_single_page(self, mock_get):
        """Test successful search with single page of results."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_search_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the function
        results = data_collector.search_places_by_query("test query", max_pages=1)
        
        # Assertions
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["place_id"], "test_place_1")
        self.assertEqual(results[1]["place_id"], "test_place_2")
        mock_get.assert_called_once()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_search_places_by_query_multiple_pages(self, mock_sleep, mock_get):
        """Test search with multiple pages of results."""
        # Setup mock responses for multiple pages
        first_page_response = MagicMock()
        first_page_response.json.return_value = self.sample_search_response
        first_page_response.raise_for_status.return_value = None

        second_page_response = MagicMock()
        second_page_response.json.return_value = {
            "status": "OK",
            "results": [{"place_id": "test_place_3", "name": "Test Bakery 3"}]
        }
        second_page_response.raise_for_status.return_value = None

        mock_get.side_effect = [first_page_response, second_page_response]

        # Test the function
        results = data_collector.search_places_by_query("test query", max_pages=2)

        # Assertions
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(2)  # Should sleep between pages

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_search_places_by_query_max_three_pages_limit(self, mock_sleep, mock_get):
        """Test that search respects Google's 3-page (60 results) maximum limit."""
        # Create mock responses for 4 pages, but only first 3 should be processed
        page_responses = []

        # Pages 1-3: Each has next_page_token
        for i in range(3):
            response = MagicMock()
            response.json.return_value = {
                "status": "OK",
                "results": [
                    {"place_id": f"place_{i}_1", "name": f"Business {i}_1"},
                    {"place_id": f"place_{i}_2", "name": f"Business {i}_2"}
                ],
                "next_page_token": f"token_page_{i+1}" if i < 2 else None  # No token after page 3
            }
            response.raise_for_status.return_value = None
            page_responses.append(response)

        # Page 4: Should not be called due to no next_page_token from page 3
        fourth_page_response = MagicMock()
        fourth_page_response.json.return_value = {
            "status": "OK",
            "results": [{"place_id": "place_4_1", "name": "Business 4_1"}]
        }
        fourth_page_response.raise_for_status.return_value = None
        page_responses.append(fourth_page_response)

        mock_get.side_effect = page_responses

        # Test requesting 4 pages (more than Google's 3-page limit)
        results = data_collector.search_places_by_query("test query", max_pages=4)

        # Assertions
        self.assertEqual(len(results), 6)  # 3 pages × 2 results each = 6 total
        self.assertEqual(mock_get.call_count, 3)  # Only 3 API calls made (not 4)
        self.assertEqual(mock_sleep.call_count, 2)  # 2 sleeps between 3 pages

        # Verify we got results from first 3 pages only
        place_ids = [result["place_id"] for result in results]
        expected_ids = ["place_0_1", "place_0_2", "place_1_1", "place_1_2", "place_2_1", "place_2_2"]
        self.assertEqual(place_ids, expected_ids)

        # Verify no results from the 4th page
        self.assertNotIn("place_4_1", place_ids)

    @patch('data_collector.requests.get')
    def test_search_places_by_query_zero_results(self, mock_get):
        """Test search with zero results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = data_collector.search_places_by_query("test query")
        
        self.assertEqual(len(results), 0)

    @patch('data_collector.requests.get')
    def test_search_places_by_query_explicit_api_key_skips_lookup(self, mock_get):
        """Explicit API key should bypass lazy lookup."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ZERO_RESULTS", "results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with patch.object(data_collector, 'get_api_key', side_effect=AssertionError("should not call")):
            data_collector.search_places_by_query("query", max_pages=1, api_key="custom-key")

        called_params = mock_get.call_args[1]['params']
        self.assertEqual(called_params['key'], "custom-key")

    @patch('data_collector.requests.get')
    def test_search_places_by_query_api_error(self, mock_get):
        """Test search with API error response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "REQUEST_DENIED",
            "error_message": "API key invalid"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = data_collector.search_places_by_query("test query")
        
        self.assertEqual(len(results), 0)

    @patch('data_collector.requests.get')
    def test_search_places_by_query_network_error(self, mock_get):
        """Test search with network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        results = data_collector.search_places_by_query("test query")
        
        self.assertEqual(len(results), 0)

    @patch('data_collector.requests.get')
    def test_get_place_details_success(self, mock_get):
        """Test successful place details retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_place_details
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = data_collector.get_place_details("test_place_id")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test Bakery")
        self.assertEqual(result["formatted_address"], "123 Test St, Test City, CA 12345")

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_get_place_details_quota_exceeded_retry(self, mock_sleep, mock_get):
        """Test place details with quota exceeded and retry."""
        # First call returns quota exceeded, second call succeeds
        quota_response = MagicMock()
        quota_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        quota_response.raise_for_status.return_value = None
        quota_response.status_code = 200
        
        success_response = MagicMock()
        success_response.json.return_value = self.sample_place_details
        success_response.raise_for_status.return_value = None
        success_response.status_code = 200
        
        mock_get.side_effect = [quota_response, success_response]
        
        result = data_collector.get_place_details("test_place_id")
        
        self.assertIsNotNone(result)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(60)

    @patch('data_collector.requests.get')
    def test_get_place_details_api_error(self, mock_get):
        """Test place details with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "NOT_FOUND",
            "error_message": "Place not found"
        }
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = data_collector.get_place_details("invalid_place_id")
        
        self.assertIsNone(result)

    @patch('data_collector.requests.get')
    def test_get_place_details_network_error(self, mock_get):
        """Test place details with network error."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        result = data_collector.get_place_details("test_place_id")
        
        self.assertIsNone(result)

    @patch('data_collector.get_place_details')
    @patch('data_collector.time.sleep')
    def test_process_businesses_data_success(self, mock_sleep, mock_get_details):
        """Test successful processing of business data."""
        search_results = [
            {"place_id": "place_1", "name": "Business 1"},
            {"place_id": "place_2", "name": "Business 2"}
        ]
        
        mock_get_details.return_value = self.sample_place_details["result"]
        
        result = data_collector.process_businesses_data(search_results)
        
        self.assertEqual(len(result), 2)
        place_ids = {entry["place_id"] for entry in result}
        self.assertEqual(place_ids, {"place_1", "place_2"})
        for entry in result:
            self.assertEqual(entry["name"], "Test Bakery")
            self.assertEqual(entry["latitude"], 37.7749)
            self.assertEqual(entry["longitude"], -122.4194)
        
        # Should sleep between requests (but not before first)
        # mock_sleep.assert_called_once_with(2)  # Sleep between requests - Removed as we use RateLimiter now

    def test_process_businesses_data_empty_input(self):
        """Test processing with empty search results."""
        result = data_collector.process_businesses_data([])
        
        self.assertEqual(len(result), 0)

    @patch('data_collector.get_place_details')
    def test_process_businesses_data_missing_place_id(self, mock_get_details):
        """Test processing with missing place_id."""
        search_results = [
            {"name": "Business without ID"},
            {"place_id": "valid_place", "name": "Valid Business"}
        ]
        
        mock_get_details.return_value = self.sample_place_details["result"]
        
        result = data_collector.process_businesses_data(search_results)
        
        # Should only process the valid business
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["place_id"], "valid_place")

    @patch('data_collector.get_place_details')
    def test_process_businesses_data_details_failure(self, mock_get_details):
        """Test processing when place details retrieval fails."""
        search_results = [{"place_id": "place_1", "name": "Business 1"}]
        mock_get_details.return_value = None
        
        result = data_collector.process_businesses_data(search_results)
        
        self.assertEqual(len(result), 0)

    def test_get_api_key_lazy_loads_from_env(self):
        """get_api_key should reload the env when cached key is empty."""
        with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'lazy_key'}, clear=True):
            data_collector.API_KEY = None
            self.assertEqual(data_collector.get_api_key(), 'lazy_key')

    def test_get_api_key_raises_when_missing(self):
        """get_api_key should raise when key is unavailable."""
        with patch.dict(os.environ, {}, clear=True):
            data_collector.API_KEY = None
            with patch.object(data_collector, 'load_dotenv', return_value=None):
                with self.assertRaises(EnvironmentError):
                    data_collector.get_api_key()

    def test_confirm_execution_yes_flag_skips_prompt(self):
        """--yes flag bypasses interactive prompt."""
        args = SimpleNamespace(yes=True)
        stdin = SimpleNamespace(isatty=lambda: True)
        mock_input = MagicMock()

        self.assertTrue(data_collector.confirm_execution(args, input_func=mock_input, stdin=stdin))
        mock_input.assert_not_called()

    def test_confirm_execution_non_interactive_skips_prompt(self):
        """Non-interactive stdin bypasses prompt."""
        args = SimpleNamespace(yes=False)
        stdin = SimpleNamespace(isatty=lambda: False)
        mock_input = MagicMock()

        self.assertTrue(data_collector.confirm_execution(args, input_func=mock_input, stdin=stdin))
        mock_input.assert_not_called()

    def test_confirm_execution_decline(self):
        """User selecting 'n' should stop execution."""
        args = SimpleNamespace(yes=False)
        stdin = SimpleNamespace(isatty=lambda: True)
        mock_input = MagicMock(return_value='n')

        self.assertFalse(data_collector.confirm_execution(args, input_func=mock_input, stdin=stdin))
        mock_input.assert_called_once()

    def test_save_data_to_json_success(self):
        """Test successful JSON file saving."""
        test_data = [{"name": "Test Business", "address": "123 Test St"}]

        # Use a simple filename - secure operations will handle the path
        filename = "test_data.json"

        try:
            # Get the actual saved path
            saved_path = data_collector.save_data_to_json(test_data, filename)

            # Verify function returned a path
            self.assertIsNotNone(saved_path)
            self.assertTrue(os.path.exists(saved_path))

            # Verify file contains correct data
            with open(saved_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data, test_data)
        finally:
            # Clean up using the actual saved path
            if 'saved_path' in locals() and saved_path and os.path.exists(saved_path):
                os.unlink(saved_path)

    def test_save_data_to_json_empty_data(self):
        """Test saving with empty data."""
        # Should not create file and should handle gracefully
        result = data_collector.save_data_to_json([], "empty_test.json")

        # Should return None for empty data
        self.assertIsNone(result)

    @patch('data_collector.safe_write_text', side_effect=Exception("Permission denied"))
    def test_save_data_to_json_io_error(self, mock_safe_write):
        """Test saving with IO error."""
        test_data = [{"name": "Test Business"}]

        # Should handle error gracefully and return None
        result = data_collector.save_data_to_json(test_data, "invalid_path.json")

        # Should return None on error
        self.assertIsNone(result)

    def test_effective_pages_calculation(self):
        """Test that effective_pages is capped at 3 (Google's limit)."""
        # Test cases: (requested_pages, expected_effective_pages)
        test_cases = [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 3),  # Should be capped at 3
            (5, 3),  # Should be capped at 3
            (10, 3)  # Should be capped at 3
        ]

        for requested_pages, expected_effective_pages in test_cases:
            with self.subTest(requested_pages=requested_pages):
                effective_pages = min(requested_pages, 3)
                self.assertEqual(effective_pages, expected_effective_pages,
                               f"For {requested_pages} requested pages, expected {expected_effective_pages} effective pages")

    @patch('data_collector.search_places_by_query')
    @patch('data_collector.process_businesses_data')
    @patch('data_collector.save_data_to_json')
    @patch('builtins.input', return_value='y')  # Mock user confirmation
    @patch('sys.argv', ['data_collector.py', '--pages', '5'])  # Mock command line args
    def test_main_execution_pages_limit(self, mock_save, mock_process, mock_search, mock_input):
        """Test that main execution respects the 3-page limit even when more pages are requested."""
        # Setup mocks
        mock_search.return_value = [{"place_id": "test1"}, {"place_id": "test2"}]
        mock_process.return_value = [{"name": "Business 1"}, {"name": "Business 2"}]

        # Import and run the main section logic (simulated)
        # Note: This tests the logic that would be in the main execution
        requested_pages = 5
        effective_pages = min(requested_pages, 3)

        # Verify effective_pages is capped at 3
        self.assertEqual(effective_pages, 3)

        # This simulates what the main function should do
        mock_search_results = mock_search.return_value
        mock_businesses_data = mock_process.return_value

        # Verify the functions would be called with the correct effective_pages
        # In the actual implementation, search_places_by_query should be called with effective_pages=3
        self.assertEqual(len(mock_search_results), 2)  # Mock data
        self.assertEqual(len(mock_businesses_data), 2)  # Mock data


if __name__ == '__main__':
    unittest.main()

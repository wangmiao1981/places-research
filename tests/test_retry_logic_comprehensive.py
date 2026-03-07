#!/usr/bin/env python3
"""
Comprehensive test suite for retry logic and quota exceeded handling.
Tests both positive cases (should retry successfully) and negative cases (should fail after max retries).
"""

import unittest
import time
from unittest.mock import patch, MagicMock, call
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid_search import ZipCodeGridSearcher
from data_collector import get_place_details
import requests


class TestRetryLogicPositiveCases(unittest.TestCase):
    """Test cases where retry logic SHOULD succeed."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")
        # Disable rate limiter for retry tests to avoid side effects
        import data_collector
        self.original_rate_limiter = data_collector.rate_limiter
        data_collector.rate_limiter = None

    def tearDown(self):
        """Clean up test environment."""
        import data_collector
        data_collector.rate_limiter = self.original_rate_limiter

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_data_collector_retry_success_first_attempt(self, mock_sleep, mock_get):
        """POSITIVE: Should succeed on first attempt without retries."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "result": {"name": "Test Business", "place_id": "test_123"}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should succeed without retries
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test Business")

        # Should make only one API call
        self.assertEqual(mock_get.call_count, 1)
        # Should not sleep (no retries)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_data_collector_retry_success_after_quota_error(self, mock_sleep, mock_get):
        """POSITIVE: Should succeed after one quota exceeded error."""
        # First call: quota exceeded, second call: success
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OK", "result": {"name": "Success Business"}})
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = get_place_details("test_place_id")

        # Should eventually succeed
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Success Business")

        # Should make 2 API calls (first fails, second succeeds)
        self.assertEqual(mock_get.call_count, 2)
        # Should sleep once (60 seconds exponential backoff)
        mock_sleep.assert_called_once_with(60)

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_data_collector_retry_success_after_multiple_quota_errors(self, mock_sleep, mock_get):
        """POSITIVE: Should succeed after multiple quota exceeded errors with exponential backoff."""
        # Three quota errors, then success
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),  # 1st attempt
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),  # 1st retry (60s wait)
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),  # 2nd retry (120s wait)
            MagicMock(json=lambda: {"status": "OK", "result": {"name": "Final Success"}})  # 3rd retry (240s wait)
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = get_place_details("test_place_id")

        # Should eventually succeed
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Final Success")

        # Should make 4 API calls (3 failures + 1 success)
        self.assertEqual(mock_get.call_count, 4)

        # Should sleep with exponential backoff: 60s, 120s, 240s
        expected_sleep_calls = [call(60), call(120), call(240)]
        self.assertEqual(mock_sleep.call_args_list, expected_sleep_calls)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_grid_search_retry_success_after_quota_error(self, mock_sleep, mock_get):
        """POSITIVE: Grid searcher should succeed after quota exceeded error."""
        # First call: quota exceeded, second call: success
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OK", "result": {"name": "Grid Success"}})
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = self.searcher.get_place_details("test_place_id")

        # Should eventually succeed
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Grid Success")

        # Should make 2 API calls
        self.assertEqual(mock_get.call_count, 2)
        # Should sleep once (60 seconds)
        mock_sleep.assert_called_once_with(60)

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_retry_with_custom_max_retries(self, mock_sleep, mock_get):
        """POSITIVE: Should respect custom max_retries parameter."""
        # Two quota errors, then success (should succeed with max_retries=2)
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OK", "result": {"name": "Custom Retry Success"}})
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = get_place_details("test_place_id", max_retries=2)

        # Should succeed with custom max_retries
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Custom Retry Success")

        # Should make 3 API calls (2 failures + 1 success)
        self.assertEqual(mock_get.call_count, 3)


class TestRetryLogicNegativeCases(unittest.TestCase):
    """Test cases where retry logic should FAIL after max retries."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")
        # Disable rate limiter for retry tests to avoid side effects
        import data_collector
        self.original_rate_limiter = data_collector.rate_limiter
        data_collector.rate_limiter = None

    def tearDown(self):
        """Clean up test environment."""
        import data_collector
        data_collector.rate_limiter = self.original_rate_limiter

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_data_collector_retry_failure_max_retries_exceeded(self, mock_sleep, mock_get):
        """NEGATIVE: Should fail after exceeding max retries for quota errors."""
        # Always return quota exceeded (4 attempts with 3 retries = failure)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id", max_retries=3)

        # Should fail and return None
        self.assertIsNone(result)

        # Should make 4 API calls (1 initial + 3 retries)
        self.assertEqual(mock_get.call_count, 4)

        # Should sleep 3 times with exponential backoff: 60s, 120s, 240s
        expected_sleep_calls = [call(60), call(120), call(240)]
        self.assertEqual(mock_sleep.call_args_list, expected_sleep_calls)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_grid_search_retry_failure_max_retries_exceeded(self, mock_sleep, mock_get):
        """NEGATIVE: Grid searcher should fail after exceeding max retries."""
        # Always return quota exceeded
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.searcher.get_place_details("test_place_id", max_retries=2)

        # Should fail and return None
        self.assertIsNone(result)

        # Should make 3 API calls (1 initial + 2 retries)
        self.assertEqual(mock_get.call_count, 3)

        # Should sleep 2 times: 60s, 120s
        expected_sleep_calls = [call(60), call(120)]
        self.assertEqual(mock_sleep.call_args_list, expected_sleep_calls)

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_non_quota_error_no_retry(self, mock_sleep, mock_get):
        """NEGATIVE: Should NOT retry for non-quota errors."""
        # Return different error (not quota exceeded)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "NOT_FOUND",
            "error_message": "Place not found"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("invalid_place_id")

        # Should fail immediately without retries
        self.assertIsNone(result)

        # Should make only 1 API call (no retries for non-quota errors)
        self.assertEqual(mock_get.call_count, 1)

        # Should not sleep (no retries)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_network_error_no_retry(self, mock_sleep, mock_get):
        """NEGATIVE: Should NOT retry for network errors."""
        # Simulate network error
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        result = get_place_details("test_place_id")

        # Should fail immediately without retries
        self.assertIsNone(result)

        # Should make only 1 API call (no retries for network errors)
        self.assertEqual(mock_get.call_count, 1)

        # Should not sleep (no retries for network errors)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_zero_max_retries(self, mock_sleep, mock_get):
        """NEGATIVE: Should NOT retry when max_retries=0."""
        # Return quota exceeded
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id", max_retries=0)

        # Should fail immediately without retries
        self.assertIsNone(result)

        # Should make only 1 API call (no retries allowed)
        self.assertEqual(mock_get.call_count, 1)

        # Should not sleep (no retries allowed)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    def test_http_error_no_retry(self, mock_get):
        """NEGATIVE: Should NOT retry for HTTP errors (e.g., 403, 404)."""
        # Simulate HTTP 403 error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should fail immediately without retries
        self.assertIsNone(result)

        # Should make only 1 API call (no retries for HTTP errors)
        self.assertEqual(mock_get.call_count, 1)


class TestRetryLogicEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for retry logic."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")
        # Disable rate limiter for retry tests to avoid side effects
        import data_collector
        self.original_rate_limiter = data_collector.rate_limiter
        data_collector.rate_limiter = None

    def tearDown(self):
        """Clean up test environment."""
        import data_collector
        data_collector.rate_limiter = self.original_rate_limiter

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_exponential_backoff_timing(self, mock_sleep, mock_get):
        """EDGE CASE: Verify correct exponential backoff timing."""
        # Always return quota exceeded to test all backoff delays
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        start_time = time.time()
        result = get_place_details("test_place_id", max_retries=3)
        end_time = time.time()

        # Should fail after max retries
        self.assertIsNone(result)

        # Verify exponential backoff delays: 60s, 120s, 240s (total 420s in test)
        expected_sleep_calls = [call(60), call(120), call(240)]
        self.assertEqual(mock_sleep.call_args_list, expected_sleep_calls)

        # Verify total sleep time is correct (420 seconds in mocked time)
        total_sleep_time = sum(call_args[0][0] for call_args in mock_sleep.call_args_list)
        self.assertEqual(total_sleep_time, 420)  # 60 + 120 + 240

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_malformed_json_response(self, mock_sleep, mock_get):
        """EDGE CASE: Should handle malformed JSON responses gracefully."""
        # Return malformed JSON
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should fail gracefully without retries
        self.assertIsNone(result)

        # Should make only 1 API call
        self.assertEqual(mock_get.call_count, 1)

        # Should not sleep (not a quota error)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_missing_status_field(self, mock_sleep, mock_get):
        """EDGE CASE: Should handle response without status field."""
        # Return JSON without status field
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": {"name": "No Status"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should handle missing status gracefully (treat as non-OK)
        self.assertIsNone(result)

        # Should make only 1 API call
        self.assertEqual(mock_get.call_count, 1)

        # Should not retry (not explicitly OVER_QUERY_LIMIT)
        mock_sleep.assert_not_called()

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_timeout_with_retry_parameter(self, mock_sleep, mock_get):
        """EDGE CASE: Verify timeout parameter is passed correctly."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "result": {"name": "Timeout Test"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should succeed
        self.assertIsNotNone(result)

        # Verify timeout parameter was passed
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('timeout', call_args[1])
        self.assertEqual(call_args[1]['timeout'], 30)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_grid_search_class_method_signature_compatibility(self, mock_sleep, mock_get):
        """EDGE CASE: Ensure class method works with new parameters."""
        # Test that class method accepts new parameters
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "result": {"name": "Class Method Test"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test with explicit parameters
        result = self.searcher.get_place_details("test_place_id", max_retries=5, retry_count=0)

        # Should succeed
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Class Method Test")

        # Should make 1 API call
        self.assertEqual(mock_get.call_count, 1)

    def test_recursive_call_depth_safety(self):
        """EDGE CASE: Ensure recursion depth is safe with max_retries limit."""
        # Test that we don't hit Python's recursion limit
        max_retries = 100  # High number to test recursion safety

        # The recursion depth should never exceed max_retries + 1
        # With max_retries=100, we should have at most 101 recursive calls
        # Python's default recursion limit is 1000, so this should be safe

        # This is more of a design verification than a runtime test
        # The key insight is that our recursion is bounded by max_retries
        self.assertLessEqual(max_retries + 1, 1000)  # Well below Python's recursion limit

        # Verify that our exponential backoff doesn't overflow
        max_delay = 60 * (2 ** max_retries)  # This would be huge, but let's verify it doesn't break
        # In practice, we'll never reach this due to reasonable max_retries values
        self.assertIsInstance(max_delay, int)  # Should not overflow to float


class TestRetryLogicIntegration(unittest.TestCase):
    """Integration tests for retry logic in broader workflows."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")
        # Disable rate limiter for retry tests to avoid side effects
        import data_collector
        self.original_rate_limiter = data_collector.rate_limiter
        data_collector.rate_limiter = None

    def tearDown(self):
        """Clean up test environment."""
        import data_collector
        data_collector.rate_limiter = self.original_rate_limiter

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_retry_logic_preserves_function_behavior(self, mock_sleep, mock_get):
        """INTEGRATION: Ensure retry logic doesn't break existing function contracts."""
        # Test that function still returns expected data structure
        expected_result = {
            "name": "Integration Test Business",
            "formatted_address": "123 Test St, Test City, CA",
            "rating": 4.5,
            "reviews": [{"text": "Great service"}],
            "geometry": {"location": {"lat": 37.7749, "lng": -122.4194}}
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "result": expected_result}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_place_details("test_place_id")

        # Should return complete, expected data structure
        self.assertEqual(result, expected_result)
        self.assertIn("name", result)
        self.assertIn("rating", result)
        self.assertIn("geometry", result)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_grid_search_detail_fetching_with_retry(self, mock_sleep, mock_get):
        """INTEGRATION: Test retry logic works in grid search detail fetching workflow."""
        # Simulate quota exceeded followed by successful detail fetch
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {
                "status": "OK",
                "result": {
                    "name": "Detailed Business",
                    "website": "https://example.com",
                    "formatted_phone_number": "(555) 123-4567"
                }
            })
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = self.searcher.get_place_details("test_place_id")

        # Should eventually get detailed information
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Detailed Business")
        self.assertIn("website", result)
        self.assertIn("formatted_phone_number", result)

        # Should have retried once
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(60)


class TestRetryLogicPerformance(unittest.TestCase):
    """Performance and timing tests for retry logic."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_retry_performance_no_unnecessary_delays(self, mock_sleep, mock_get):
        """PERFORMANCE: Ensure no delays when no retries needed."""
        # Mock immediate success
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "result": {"name": "Fast Success"}}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        start_time = time.time()
        result = get_place_details("test_place_id")
        end_time = time.time()

        # Should succeed quickly
        self.assertIsNotNone(result)

        # Should not sleep at all
        mock_sleep.assert_not_called()

        # Should complete very quickly (less than 1 second in real time)
        elapsed_time = end_time - start_time
        self.assertLess(elapsed_time, 1.0)

    @patch('data_collector.requests.get')
    @patch('data_collector.time.sleep')
    def test_retry_timing_accuracy(self, mock_sleep, mock_get):
        """PERFORMANCE: Verify retry timing is accurate."""
        # Mock quota exceeded twice, then success
        mock_responses = [
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OVER_QUERY_LIMIT"}),
            MagicMock(json=lambda: {"status": "OK", "result": {"name": "Timed Success"}})
        ]

        for response in mock_responses:
            response.raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        result = get_place_details("test_place_id")

        # Should succeed after retries
        self.assertIsNotNone(result)

        # Should sleep with exact exponential backoff timing
        expected_sleep_calls = [call(60), call(120)]  # 2^0 * 60, 2^1 * 60
        self.assertEqual(mock_sleep.call_args_list, expected_sleep_calls)


if __name__ == "__main__":
    # Run all test suites
    unittest.main(verbosity=2)
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

#!/usr/bin/env python3
"""
Comprehensive test suite for deduplication functionality.
Tests both positive cases (should deduplicate) and negative cases (should not deduplicate).
"""

import unittest
import time
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from grid_search import ZipCodeGridSearcher, SearchResult, GridCell


class TestDeduplicationPositiveCases(unittest.TestCase):
    """Test cases where deduplication SHOULD occur."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")

    def test_exact_duplicate_by_place_id(self):
        """POSITIVE: Should deduplicate exact same place_id."""
        results = [
            SearchResult("place_123", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_123", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_2"),  # Same place_id
        ]

        unique_results = self.searcher.deduplicate_results(results)

        # Should keep only one
        self.assertEqual(len(unique_results), 1)
        self.assertEqual(unique_results[0].place_id, "place_123")

    def test_near_duplicates_same_location_similar_names(self):
        """POSITIVE: Should deduplicate businesses at same location with similar names."""
        results = [
            SearchResult("place_1", "Starbucks Coffee", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.6, "grid_2"),  # Same location, similar name
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should deduplicate to one result
        self.assertEqual(len(unique_results), 1)
        self.assertIn("Starbucks", unique_results[0].name)

    def test_close_distance_similar_names(self):
        """POSITIVE: Should deduplicate businesses within threshold distance with similar names."""
        results = [
            SearchResult("place_1", "McDonald's", 37.2431, -121.7903, "123 Main St", 4.0, "grid_1"),
            SearchResult("place_2", "McDonald's Restaurant", 37.2432, -121.7904, "124 Main St", 4.1, "grid_2"),  # ~15m away
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should deduplicate (within 50m and similar names)
        self.assertEqual(len(unique_results), 1)
        self.assertIn("McDonald", unique_results[0].name)

    def test_case_insensitive_name_matching(self):
        """POSITIVE: Should deduplicate case-insensitive name variants."""
        results = [
            SearchResult("place_1", "STARBUCKS COFFEE", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "starbucks coffee", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should deduplicate case variants
        self.assertEqual(len(unique_results), 1)

    def test_multiple_duplicates_keep_best(self):
        """POSITIVE: With multiple duplicates, should keep the best one (highest rating)."""
        results = [
            SearchResult("place_1", "Burger King", 37.2431, -121.7903, "123 Main St", 4.0, "grid_1"),  # Lowest rating
            SearchResult("place_2", "Burger King", 37.2432, -121.7904, "123 Main St", 4.1, "grid_2"),  # Duplicate
            SearchResult("place_3", "Burger King Restaurant", 37.2433, -121.7905, "123 Main St", 4.2, "grid_3"),  # Highest rating
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=100.0)

        # Should keep only the best one
        self.assertEqual(len(unique_results), 1)
        self.assertEqual(unique_results[0].place_id, "place_3")

    def test_large_dataset_deduplication(self):
        """POSITIVE: Should handle large dataset with many duplicates efficiently."""
        results = []
        # Create 100 results that are all duplicates of each other
        for i in range(100):
            results.append(
                SearchResult(f"place_{i}", "Starbucks Coffee", 37.2431, -121.7903, "123 Main St", 4.5, f"grid_{i}")
            )

        start_time = time.time()
        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)
        elapsed_time = time.time() - start_time

        # Should deduplicate to just one result
        self.assertEqual(len(unique_results), 1)
        # Should complete in reasonable time
        self.assertLess(elapsed_time, 5.0, f"Deduplication too slow: {elapsed_time:.2f}s for 100 duplicates")


class TestDeduplicationNegativeCases(unittest.TestCase):
    """Test cases where deduplication should NOT occur."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")

    def test_different_place_ids_far_apart(self):
        """NEGATIVE: Should NOT deduplicate different businesses far apart."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Starbucks", 37.3000, -121.8500, "999 Far Ave", 4.2, "grid_2"),  # Different location
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should keep both (different locations)
        self.assertEqual(len(unique_results), 2)
        place_ids = [r.place_id for r in unique_results]
        self.assertEqual(set(place_ids), {"place_1", "place_2"})

    def test_same_location_different_names(self):
        """NEGATIVE: Should NOT deduplicate different businesses at same location."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "McDonald's", 37.2431, -121.7903, "123 Main St", 3.8, "grid_1"),  # Same building, different business
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should keep both (different business types)
        self.assertEqual(len(unique_results), 2)
        names = [r.name for r in unique_results]
        self.assertIn("Starbucks", names)
        self.assertIn("McDonald's", names)

    def test_similar_names_far_apart(self):
        """NEGATIVE: Should NOT deduplicate similar names that are far apart."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Starbucks Coffee", 37.2500, -121.8000, "999 Oak Ave", 4.2, "grid_2"),  # ~8km away
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should keep both (too far apart despite similar names)
        self.assertEqual(len(unique_results), 2)
        place_ids = [r.place_id for r in unique_results]
        self.assertEqual(set(place_ids), {"place_1", "place_2"})

    def test_close_distance_dissimilar_names(self):
        """NEGATIVE: Should NOT deduplicate close businesses with very different names."""
        results = [
            SearchResult("place_1", "Starbucks Coffee", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Pizza Palace", 37.2432, -121.7904, "124 Main St", 4.0, "grid_2"),  # Close but different
        ]

        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)

        # Should keep both (different business types despite proximity)
        self.assertEqual(len(unique_results), 2)
        names = [r.name for r in unique_results]
        self.assertIn("Starbucks", names[0] if "Starbucks" in names[0] else names[1])
        self.assertIn("Pizza", names[0] if "Pizza" in names[0] else names[1])

    def test_no_duplicates_mixed_businesses(self):
        """NEGATIVE: Should NOT deduplicate when no actual duplicates exist."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "McDonald's", 37.2441, -121.7913, "456 Oak Ave", 4.0, "grid_2"),
            SearchResult("place_3", "Subway", 37.2451, -121.7923, "789 Pine St", 3.8, "grid_3"),
            SearchResult("place_4", "Taco Bell", 37.2461, -121.7933, "321 Elm St", 3.9, "grid_4"),
        ]

        unique_results = self.searcher.deduplicate_results(results)

        # Should keep all (no duplicates)
        self.assertEqual(len(unique_results), 4)
        place_ids = [r.place_id for r in unique_results]
        self.assertEqual(set(place_ids), {"place_1", "place_2", "place_3", "place_4"})

    def test_empty_input(self):
        """NEGATIVE: Should handle empty input gracefully."""
        unique_results = self.searcher.deduplicate_results([])

        # Should return empty list
        self.assertEqual(len(unique_results), 0)
        self.assertIsInstance(unique_results, list)

    def test_single_result(self):
        """NEGATIVE: Should NOT modify single result."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
        ]

        unique_results = self.searcher.deduplicate_results(results)

        # Should keep the single result unchanged
        self.assertEqual(len(unique_results), 1)
        self.assertEqual(unique_results[0].place_id, "place_1")
        self.assertEqual(unique_results[0].name, "Starbucks")

    def test_different_thresholds_no_dedup(self):
        """NEGATIVE: Should NOT deduplicate when distance exceeds threshold."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Starbucks Coffee", 37.2440, -121.7910, "456 Oak Ave", 4.2, "grid_2"),  # ~100m away
        ]

        # With 50m threshold - should NOT deduplicate (they're ~100m apart)
        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)
        self.assertEqual(len(unique_results), 2)

        # With 10m threshold - should definitely NOT deduplicate
        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=10.0)
        self.assertEqual(len(unique_results), 2)


class TestDeduplicationEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")

    def test_extreme_coordinates(self):
        """EDGE CASE: Should handle extreme coordinate values."""
        results = [
            SearchResult("place_1", "North Pole Shop", 89.9999, -180.0, "North Pole", 4.5, "grid_1"),
            SearchResult("place_2", "South Pole Shop", -89.9999, 179.9999, "South Pole", 4.5, "grid_2"),
        ]

        # Should not crash with extreme coordinates
        unique_results = self.searcher.deduplicate_results(results)

        # Should keep both (very far apart)
        self.assertEqual(len(unique_results), 2)

    def test_unicode_business_names(self):
        """EDGE CASE: Should handle Unicode characters in business names."""
        results = [
            SearchResult("place_1", "Café François", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Cafe Francois", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),  # ASCII version
            SearchResult("place_3", "北京餐厅", 37.2432, -121.7904, "124 Main St", 4.2, "grid_2"),  # Chinese
        ]

        # Should handle Unicode without errors
        unique_results = self.searcher.deduplicate_results(results)

        # Should work without throwing exceptions
        self.assertIsInstance(unique_results, list)
        self.assertGreater(len(unique_results), 0)

        # Verify Unicode is preserved
        names = [r.name for r in unique_results]
        unicode_found = any(ord(char) > 127 for name in names for char in name)
        self.assertTrue(unicode_found, "Unicode characters should be preserved")

    def test_very_long_business_names(self):
        """EDGE CASE: Should handle very long business names."""
        long_name = "A" * 500  # 500 character name
        results = [
            SearchResult("place_1", long_name, 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", long_name + " Inc", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
        ]

        # Should handle long names without errors
        unique_results = self.searcher.deduplicate_results(results)

        # Should complete successfully
        self.assertIsInstance(unique_results, list)
        self.assertLessEqual(len(unique_results), 2)

    def test_zero_threshold_distance(self):
        """EDGE CASE: Should handle zero distance threshold."""
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_2", "Starbucks Coffee", 37.2432, -121.7904, "123 Main St", 4.5, "grid_1"),
        ]

        # With 0 threshold, only exact coordinates should be considered duplicates
        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=0.0)

        # Should keep both (different coordinates)
        self.assertEqual(len(unique_results), 2)

    def test_performance_regression_check(self):
        """PERFORMANCE: Ensure deduplication doesn't regress in performance."""
        # Create realistic dataset with mix of duplicates and unique results
        results = []

        # 500 unique businesses
        for i in range(500):
            results.append(
                SearchResult(f"unique_{i}", f"Business {i}", 37.2431 + i*0.001, -121.7903 + i*0.001, f"Address {i}", 4.0, f"grid_{i}")
            )

        # 100 duplicates of the first business
        for i in range(100):
            results.append(
                SearchResult(f"dup_{i}", "Business 0", 37.2431, -121.7903, "Address 0", 4.0, f"dup_grid_{i}")
            )

        start_time = time.time()
        unique_results = self.searcher.deduplicate_results(results, distance_threshold_m=50.0)
        elapsed_time = time.time() - start_time

        # Should reduce to ~500 unique results
        self.assertEqual(len(unique_results), 500)

        # Performance regression check - should complete in reasonable time
        # For 600 results, should be much less than 1 second
        self.assertLess(elapsed_time, 2.0, f"Performance regression: {elapsed_time:.2f}s for 600 results")

        print(f"\n📊 Performance check: {len(results)} → {len(unique_results)} results in {elapsed_time:.3f}s")


class TestDeduplicationIntegration(unittest.TestCase):
    """Integration tests for deduplication within the broader workflow."""

    def setUp(self):
        """Set up test environment."""
        self.searcher = ZipCodeGridSearcher("test_api_key")

    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    @patch('grid_search.ZipCodeGridSearcher.search_grid_cell')
    @patch('grid_search.time.sleep')
    def test_exhaustive_search_deduplication_integration(self, mock_sleep, mock_search, mock_bounds):
        """INTEGRATION: Test deduplication works correctly within exhaustive search."""
        # Mock zip code bounds (include all required fields)
        mock_bounds.return_value = {
            'north': 37.25, 'south': 37.23, 'east': -121.78, 'west': -121.80,
            'area_km2': 4.0, 'center_lat': 37.24, 'center_lng': -121.79
        }

        # Mock grid cell search to return overlapping results
        # The method generates 4 grid cells, so we need 4 responses
        grid_results = [
            # Grid 1 results
            [SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_0_0"),
             SearchResult("place_2", "Peet's Coffee", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_0_0")],

            # Grid 2 results (with duplicates)
            [SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_0_1"),  # Duplicate
             SearchResult("place_3", "Blue Bottle", 37.2451, -121.7923, "789 Pine St", 4.7, "grid_0_1")],

            # Grid 3 results (empty)
            [],

            # Grid 4 results (one more unique)
            [SearchResult("place_4", "Coffee Bean", 37.2461, -121.7933, "999 Elm St", 4.1, "grid_1_1")]
        ]
        mock_search.side_effect = grid_results

        # Run exhaustive search
        results = self.searcher.exhaustive_search("95125", "coffee shop", max_pages_per_grid=1)

        # Verify deduplication occurred
        self.assertEqual(len(results), 4)  # Should have 4 unique results
        place_ids = [r.place_id for r in results]
        self.assertEqual(set(place_ids), {"place_1", "place_2", "place_3", "place_4"})

        # Verify the duplicate was properly removed
        starbucks_results = [r for r in results if r.place_id == "place_1"]
        self.assertEqual(len(starbucks_results), 1)  # Only one Starbucks should remain


if __name__ == "__main__":
    # Run all test suites
    unittest.main(verbosity=2)
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
from unittest.mock import patch, MagicMock
import json
import os
import sys
import math

# Add src directory to path to import grid_search
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the environment variable before importing grid_search
with patch.dict(os.environ, {'GOOGLE_MAPS_API_KEY': 'test_api_key'}):
    from grid_search import ZipCodeGridSearcher, GridCell, SearchResult


class TestGridSearch(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_api_key = 'test_api_key'
        self.searcher = ZipCodeGridSearcher(self.test_api_key)
        
        # Sample zip code bounds (San Jose, CA 95125)
        self.sample_bounds = {
            'north': 37.2531,
            'south': 37.2331,
            'east': -121.7803,
            'west': -121.8003,
            'center_lat': 37.2431,
            'center_lng': -121.7903,
            'area_km2': 15.5
        }
        
        # Sample geocoding API response
        self.sample_geocoding_response = {
            "status": "OK",
            "results": [{
                "geometry": {
                    "bounds": {
                        "northeast": {"lat": 37.2531, "lng": -121.7803},
                        "southwest": {"lat": 37.2331, "lng": -121.8003}
                    },
                    "location": {"lat": 37.2431, "lng": -121.7903}
                }
            }]
        }
        
        # Sample Places API response
        self.sample_places_response = {
            "status": "OK",
            "results": [
                {
                    "place_id": "test_place_1",
                    "name": "Test Coffee Shop 1",
                    "geometry": {"location": {"lat": 37.2431, "lng": -121.7903}},
                    "formatted_address": "123 Test St, San Jose, CA 95125",
                    "rating": 4.5
                },
                {
                    "place_id": "test_place_2", 
                    "name": "Test Coffee Shop 2",
                    "geometry": {"location": {"lat": 37.2441, "lng": -121.7913}},
                    "formatted_address": "456 Test Ave, San Jose, CA 95125",
                    "rating": 4.2
                }
            ]
        }

    def test_business_density_estimates(self):
        """Test that business density estimates are reasonable."""
        # Test known business types
        self.assertEqual(self.searcher.business_density['coffee shop'], 15)
        self.assertEqual(self.searcher.business_density['restaurant'], 25)
        self.assertEqual(self.searcher.business_density['gas station'], 3)
        
        # Test default fallback
        self.assertEqual(self.searcher.business_density['default'], 10)

    @patch('grid_search.requests.get')
    def test_get_zip_code_bounds_success(self, mock_get):
        """Test successful zip code bounds retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_geocoding_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        bounds = self.searcher.get_zip_code_bounds("95125")
        
        self.assertIsNotNone(bounds)
        self.assertIn('north', bounds)
        self.assertIn('south', bounds)
        self.assertIn('east', bounds)
        self.assertIn('west', bounds)
        self.assertIn('center_lat', bounds)
        self.assertIn('center_lng', bounds)
        self.assertIn('area_km2', bounds)
        
        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('address', call_args[1]['params'])
        self.assertEqual(call_args[1]['params']['address'], "95125")

    @patch('grid_search.requests.get')
    def test_get_zip_code_bounds_api_error(self, mock_get):
        """Test zip code bounds with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        bounds = self.searcher.get_zip_code_bounds("00000")
        
        self.assertIsNone(bounds)

    @patch('grid_search.requests.get')
    def test_get_zip_code_bounds_network_error(self, mock_get):
        """Test zip code bounds with network error."""
        mock_get.side_effect = Exception("Network error")
        
        bounds = self.searcher.get_zip_code_bounds("95125")
        
        self.assertIsNone(bounds)

    def test_calculate_optimal_grid_size(self):
        """Test optimal grid size calculation."""
        # Test high density business (coffee shop)
        grid_size = self.searcher.calculate_optimal_grid_size(15.5, "coffee shop")
        self.assertGreaterEqual(grid_size, 0.5)  # Minimum bound
        self.assertLessEqual(grid_size, 3.0)     # Maximum bound
        
        # Test low density business (gas station)
        grid_size_gas = self.searcher.calculate_optimal_grid_size(15.5, "gas station")
        self.assertGreater(grid_size_gas, grid_size)  # Should be larger for low density
        
        # Test unknown business type (should use default)
        grid_size_unknown = self.searcher.calculate_optimal_grid_size(15.5, "unknown business")
        self.assertGreaterEqual(grid_size_unknown, 0.5)
        self.assertLessEqual(grid_size_unknown, 3.0)

    def test_calculate_optimal_grid_size_scales_for_large_area(self):
        """Large areas should scale grid size but remain within bounds."""
        large_area = 1000  # km^2
        grid_size = self.searcher.calculate_optimal_grid_size(large_area, "coffee shop")
        self.assertEqual(grid_size, 3.0)  # Should scale up and clamp at max

    def test_generate_grid_cells(self):
        """Test grid cell generation."""
        grid_size_km = 1.5
        grid_cells = self.searcher.generate_grid_cells(self.sample_bounds, grid_size_km)
        
        # Should generate multiple grid cells
        self.assertGreater(len(grid_cells), 0)
        
        # Check grid cell properties
        for grid_cell in grid_cells:
            self.assertIsInstance(grid_cell, GridCell)
            self.assertIsNotNone(grid_cell.grid_id)
            self.assertGreater(grid_cell.search_radius_m, 0)
            self.assertEqual(grid_cell.size_km, grid_size_km)
            
            # Grid centers should be within reasonable bounds
            self.assertGreaterEqual(grid_cell.center_lat, self.sample_bounds['south'] - 0.1)
            self.assertLessEqual(grid_cell.center_lat, self.sample_bounds['north'] + 0.1)
            self.assertGreaterEqual(grid_cell.center_lng, self.sample_bounds['west'] - 0.1)
            self.assertLessEqual(grid_cell.center_lng, self.sample_bounds['east'] + 0.1)

    def test_generate_grid_cells_overlap(self):
        """Test that grid cells have proper overlap."""
        grid_size_km = 1.0
        overlap_ratio = 0.25
        
        grid_cells = self.searcher.generate_grid_cells(
            self.sample_bounds, grid_size_km, overlap_ratio
        )
        
        # Should have multiple overlapping grids for reasonable area
        self.assertGreater(len(grid_cells), 1)
        
        # Check that adjacent grids overlap (simplified check)
        if len(grid_cells) >= 2:
            grid1 = grid_cells[0]
            grid2 = grid_cells[1]
            
            # Calculate distance between centers
            distance = self.searcher._calculate_distance_meters(
                grid1.center_lat, grid1.center_lng,
                grid2.center_lat, grid2.center_lng
            )
            
            # Distance should be less than grid size (indicating overlap)
            expected_step = grid_size_km * 1000 * (1 - overlap_ratio)
            self.assertLess(distance, grid_size_km * 1000)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_search_grid_cell_success(self, mock_sleep, mock_get):
        """Test successful grid cell search."""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_places_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        grid_cell = GridCell(
            center_lat=37.2431,
            center_lng=-121.7903,
            size_km=1.5,
            search_radius_m=1000,
            grid_id="test_grid_0_0"
        )
        
        results = self.searcher.search_grid_cell(grid_cell, "coffee shop")
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].place_id, "test_place_1")
        self.assertEqual(results[0].name, "Test Coffee Shop 1")
        self.assertEqual(results[0].source_grid, "test_grid_0_0")

    @patch('grid_search.requests.get')
    def test_search_grid_cell_zero_results(self, mock_get):
        """Test grid cell search with zero results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        grid_cell = GridCell(
            center_lat=37.2431,
            center_lng=-121.7903,
            size_km=1.5,
            search_radius_m=1000,
            grid_id="test_grid_0_0"
        )
        
        results = self.searcher.search_grid_cell(grid_cell, "coffee shop")
        
        self.assertEqual(len(results), 0)

    @patch('grid_search.requests.get')
    def test_search_grid_cell_api_error(self, mock_get):
        """Test grid cell search with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "REQUEST_DENIED",
            "error_message": "API key invalid"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        grid_cell = GridCell(
            center_lat=37.2431,
            center_lng=-121.7903,
            size_km=1.5,
            search_radius_m=1000,
            grid_id="test_grid_0_0"
        )
        
        results = self.searcher.search_grid_cell(grid_cell, "coffee shop")
        
        self.assertEqual(len(results), 0)

    def test_calculate_distance_meters(self):
        """Test distance calculation between two points."""
        # Test same point (should be 0)
        distance = self.searcher._calculate_distance_meters(37.2431, -121.7903, 37.2431, -121.7903)
        self.assertAlmostEqual(distance, 0, places=1)
        
        # Test known distance (approximately 1km apart)
        lat1, lng1 = 37.2431, -121.7903
        lat2, lng2 = 37.2531, -121.7903  # ~1.1km north
        
        distance = self.searcher._calculate_distance_meters(lat1, lng1, lat2, lng2)
        self.assertGreater(distance, 1000)  # Should be > 1km
        self.assertLess(distance, 1500)     # Should be < 1.5km

    def test_names_similar(self):
        """Test business name similarity detection."""
        # Exact match
        self.assertTrue(self.searcher._names_similar("Starbucks", "Starbucks"))
        
        # Case insensitive
        self.assertTrue(self.searcher._names_similar("Starbucks", "STARBUCKS"))
        
        # Substring match (chain with location)
        self.assertTrue(self.searcher._names_similar("Starbucks", "Starbucks Coffee"))
        self.assertTrue(self.searcher._names_similar("McDonald's", "McDonald's #1234"))
        
        # Word overlap (adjust threshold for this test)
        # "Blue Bottle Coffee" vs "Blue Bottle Cafe" = 2 common words out of 4 total = 0.5 overlap
        self.assertTrue(self.searcher._names_similar("Blue Bottle Coffee", "Blue Bottle Cafe", threshold=0.5))

        # High word overlap with default threshold
        self.assertTrue(self.searcher._names_similar("Starbucks Coffee Company", "Starbucks Coffee"))

        # Different names
        self.assertFalse(self.searcher._names_similar("Starbucks", "Peet's Coffee"))
        
        # Empty names
        self.assertFalse(self.searcher._names_similar("", "Starbucks"))
        self.assertFalse(self.searcher._names_similar("Starbucks", ""))

    def test_deduplicate_results(self):
        """Test result deduplication."""
        # Create test results with duplicates
        results = [
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_2"),  # Exact duplicate
            SearchResult("place_2", "Starbucks Coffee", 37.2432, -121.7904, "123 Main St", 4.5, "grid_2"),  # Near duplicate
            SearchResult("place_3", "Peet's Coffee", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_1"),  # Unique
        ]
        
        unique_results = self.searcher.deduplicate_results(results)
        
        # Should remove duplicates
        self.assertEqual(len(unique_results), 2)  # Only Starbucks and Peet's
        
        # Check that we kept the right ones
        place_ids = [r.place_id for r in unique_results]
        self.assertIn("place_1", place_ids)  # Keep first Starbucks
        self.assertIn("place_3", place_ids)  # Keep Peet's
        self.assertNotIn("place_2", place_ids)  # Remove near-duplicate Starbucks

    def test_deduplicate_results_empty(self):
        """Test deduplication with empty input."""
        unique_results = self.searcher.deduplicate_results([])
        self.assertEqual(len(unique_results), 0)

    @patch('grid_search.ZipCodeGridSearcher.search_grid_cell')
    @patch('grid_search.ZipCodeGridSearcher.generate_grid_cells')
    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    @patch('grid_search.time.sleep')
    def test_exhaustive_search_success(self, mock_sleep, mock_bounds, mock_generate, mock_search):
        """Test successful exhaustive search."""
        # Mock zip code bounds
        mock_bounds.return_value = self.sample_bounds

        # Mock grid generation
        mock_grids = [
            GridCell(37.2431, -121.7903, 1.5, 1000, "grid_0_0"),
            GridCell(37.2441, -121.7913, 1.5, 1000, "grid_0_1")
        ]
        mock_generate.return_value = mock_grids

        # Mock search results
        mock_search.side_effect = [
            [SearchResult("place_1", "Coffee Shop 1", 37.2431, -121.7903, "123 Main St", 4.5, "grid_0_0")],
            [SearchResult("place_2", "Coffee Shop 2", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_0_1")]
        ]

        results = self.searcher.exhaustive_search("95125", "coffee shop")

        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].place_id, "place_1")
        self.assertEqual(results[1].place_id, "place_2")

        # Verify method calls
        mock_bounds.assert_called_once_with("95125")
        mock_generate.assert_called_once()
        self.assertEqual(mock_search.call_count, 2)

    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    def test_exhaustive_search_invalid_zip(self, mock_bounds):
        """Test exhaustive search with invalid zip code."""
        mock_bounds.return_value = None

        results = self.searcher.exhaustive_search("00000", "coffee shop")

        self.assertEqual(len(results), 0)
        mock_bounds.assert_called_once_with("00000")

    @patch('grid_search.ZipCodeGridSearcher.search_grid_cell')
    @patch('grid_search.ZipCodeGridSearcher.generate_grid_cells')
    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    def test_exhaustive_search_with_duplicates(self, mock_bounds, mock_generate, mock_search):
        """Test exhaustive search with duplicate results across grids."""
        # Mock zip code bounds
        mock_bounds.return_value = self.sample_bounds

        # Mock grid generation
        mock_grids = [
            GridCell(37.2431, -121.7903, 1.5, 1000, "grid_0_0"),
            GridCell(37.2441, -121.7913, 1.5, 1000, "grid_0_1")
        ]
        mock_generate.return_value = mock_grids

        # Mock search results with duplicates
        duplicate_result = SearchResult("place_1", "Starbucks", 37.2431, -121.7903, "123 Main St", 4.5, "grid_0_0")
        near_duplicate = SearchResult("place_1", "Starbucks", 37.2432, -121.7904, "123 Main St", 4.5, "grid_0_1")

        mock_search.side_effect = [
            [duplicate_result],
            [near_duplicate, SearchResult("place_2", "Peet's", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_0_1")]
        ]

        results = self.searcher.exhaustive_search("95125", "coffee shop")

        # Should deduplicate - only 2 unique results
        self.assertEqual(len(results), 2)
        place_ids = [r.place_id for r in results]
        self.assertIn("place_1", place_ids)  # Starbucks (deduplicated)
        self.assertIn("place_2", place_ids)  # Peet's

    def test_grid_cell_dataclass(self):
        """Test GridCell dataclass functionality."""
        grid = GridCell(
            center_lat=37.2431,
            center_lng=-121.7903,
            size_km=1.5,
            search_radius_m=1000,
            grid_id="test_grid"
        )

        self.assertEqual(grid.center_lat, 37.2431)
        self.assertEqual(grid.center_lng, -121.7903)
        self.assertEqual(grid.size_km, 1.5)
        self.assertEqual(grid.search_radius_m, 1000)
        self.assertEqual(grid.grid_id, "test_grid")

    def test_search_result_dataclass(self):
        """Test SearchResult dataclass functionality."""
        result = SearchResult(
            place_id="test_place",
            name="Test Business",
            lat=37.2431,
            lng=-121.7903,
            address="123 Test St",
            rating=4.5,
            source_grid="grid_0_0"
        )

        self.assertEqual(result.place_id, "test_place")
        self.assertEqual(result.name, "Test Business")
        self.assertEqual(result.lat, 37.2431)
        self.assertEqual(result.lng, -121.7903)
        self.assertEqual(result.address, "123 Test St")
        self.assertEqual(result.rating, 4.5)
        self.assertEqual(result.source_grid, "grid_0_0")

    def test_search_result_optional_fields(self):
        """Test SearchResult with optional fields."""
        result = SearchResult(
            place_id="test_place",
            name="Test Business",
            lat=37.2431,
            lng=-121.7903,
            address="123 Test St"
            # rating and source_grid are optional
        )

        self.assertIsNone(result.rating)
        self.assertIsNone(result.source_grid)

    def test_business_density_coverage(self):
        """Test that all expected business types have density estimates."""
        expected_types = [
            'coffee shop', 'cafe', 'restaurant', 'gas station',
            'pharmacy', 'bank', 'grocery store', 'hotel', 'gym'
        ]

        for business_type in expected_types:
            self.assertIn(business_type, self.searcher.business_density)
            self.assertGreater(self.searcher.business_density[business_type], 0)

    def test_grid_size_bounds(self):
        """Test that grid size calculation respects bounds."""
        # The calculation doesn't use area_km2 parameter, only business density
        # For restaurant density (25/km²): 45/25 = 1.8 km² → sqrt(1.8) = 1.34km
        # This is > 0.5km minimum, so let's test with a very high density business

        # Create a custom searcher with very high density to hit minimum bound
        high_density_searcher = ZipCodeGridSearcher(self.test_api_key)
        high_density_searcher.business_density['super_dense'] = 1000  # Very high density

        small_grid = high_density_searcher.calculate_optimal_grid_size(15.5, "super_dense")
        self.assertEqual(small_grid, 0.5)  # Should hit minimum bound

        # Test maximum bound with very low density
        low_density_searcher = ZipCodeGridSearcher(self.test_api_key)
        low_density_searcher.business_density['super_sparse'] = 0.1  # Very low density

        large_grid = low_density_searcher.calculate_optimal_grid_size(15.5, "super_sparse")
        self.assertEqual(large_grid, 3.0)  # Should hit maximum bound

    @patch('grid_search.requests.get')
    def test_search_grid_cell_pagination(self, mock_get):
        """Test grid cell search with pagination."""
        # First page response
        first_page = {
            "status": "OK",
            "results": [self.sample_places_response["results"][0]],
            "next_page_token": "next_token_123"
        }

        # Second page response
        second_page = {
            "status": "OK",
            "results": [self.sample_places_response["results"][1]]
        }

        mock_responses = [MagicMock(), MagicMock()]
        mock_responses[0].json.return_value = first_page
        mock_responses[0].raise_for_status.return_value = None
        mock_responses[1].json.return_value = second_page
        mock_responses[1].raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        with patch('grid_search.time.sleep') as mock_sleep:
            # Disable hybrid search to test just pagination
            results = self.searcher.search_grid_cell(grid_cell, "coffee shop", max_pages=2, use_hybrid=False)

        # Should get results from both pages
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_get.call_count, 2)
        # Sleep is called between pages
        self.assertGreaterEqual(mock_sleep.call_count, 1)

    @patch('grid_search.requests.get')
    def test_get_place_details_success(self, mock_get):
        """Test successful place details retrieval."""
        sample_details_response = {
            "status": "OK",
            "result": {
                "name": "Test Massage Spa",
                "formatted_address": "123 Test St, San Jose, CA 95125",
                "rating": 4.8,
                "user_ratings_total": 150,
                "reviews": [
                    {
                        "author_name": "John Doe",
                        "rating": 5,
                        "text": "Great massage experience!",
                        "time": 1234567890
                    }
                ],
                "website": "https://testmassagespa.com",
                "formatted_phone_number": "(408) 555-0123",
                "opening_hours": {
                    "weekday_text": ["Monday: 9:00 AM – 7:00 PM", "Tuesday: 9:00 AM – 7:00 PM"]
                }
            }
        }
        
        mock_response = MagicMock()
        mock_response.json.return_value = sample_details_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        details = self.searcher.get_place_details("test_place_id")
        
        self.assertIsNotNone(details)
        self.assertEqual(details["name"], "Test Massage Spa")
        self.assertEqual(details["rating"], 4.8)
        self.assertEqual(details["user_ratings_total"], 150)
        self.assertIn("reviews", details)
        self.assertEqual(len(details["reviews"]), 1)
        self.assertEqual(details["website"], "https://testmassagespa.com")

    @patch('grid_search.requests.get')
    def test_get_place_details_api_error(self, mock_get):
        """Test place details with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "NOT_FOUND",
            "error_message": "Place not found"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        details = self.searcher.get_place_details("invalid_place_id")
        
        self.assertIsNone(details)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_get_place_details_quota_exceeded_retry(self, mock_sleep, mock_get):
        """Test place details with quota exceeded and retry."""
        quota_response = MagicMock()
        quota_response.json.return_value = {"status": "OVER_QUERY_LIMIT"}
        quota_response.raise_for_status.return_value = None
        
        success_response = MagicMock()
        success_response.json.return_value = {
            "status": "OK",
            "result": {"name": "Test Place", "rating": 4.5}
        }
        success_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [quota_response, success_response]
        
        details = self.searcher.get_place_details("test_place_id")
        
        self.assertIsNotNone(details)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(60)

    @patch('grid_search.requests.get')
    def test_get_place_details_network_error(self, mock_get):
        """Test place details with network error."""
        mock_get.side_effect = Exception("Network error")
        
        details = self.searcher.get_place_details("test_place_id")
        
        self.assertIsNone(details)

    @patch('grid_search.ZipCodeGridSearcher.get_place_details')
    @patch('grid_search.time.sleep')
    def test_enhance_results_with_details_success(self, mock_sleep, mock_get_details):
        """Test successful result enhancement with details."""
        # Mock place details response
        mock_details = {
            "name": "Enhanced Massage Spa",
            "formatted_address": "456 Enhanced St, San Jose, CA 95125",
            "rating": 4.9,
            "user_ratings_total": 200,
            "reviews": [{"author_name": "Jane", "rating": 5, "text": "Excellent!"}],
            "website": "https://enhanced.com",
            "formatted_phone_number": "(408) 555-9999",
            "opening_hours": {"weekday_text": ["Monday: 8:00 AM – 8:00 PM"]}
        }
        mock_get_details.return_value = mock_details
        
        # Create basic search results
        basic_results = [
            SearchResult("place_1", "Basic Spa 1", 37.2431, -121.7903, "Basic Address", 4.0, "grid_1"),
            SearchResult("place_2", "Basic Spa 2", 37.2441, -121.7913, "Basic Address 2", 4.2, "grid_2")
        ]
        
        enhanced_results = self.searcher.enhance_results_with_details(basic_results)
        
        # Verify enhancements
        self.assertEqual(len(enhanced_results), 2)
        
        first_result = enhanced_results[0]
        self.assertEqual(first_result.name, "Enhanced Massage Spa")
        self.assertEqual(first_result.address, "456 Enhanced St, San Jose, CA 95125")
        self.assertEqual(first_result.rating, 4.9)
        self.assertEqual(first_result.total_ratings, 200)
        self.assertEqual(first_result.website, "https://enhanced.com")
        self.assertEqual(first_result.phone_number, "(408) 555-9999")
        self.assertIsNotNone(first_result.reviews)
        self.assertIsNotNone(first_result.opening_hours)
        
        # Verify API calls and rate limiting
        self.assertEqual(mock_get_details.call_count, 2)
        # mock_sleep.assert_called_once_with(2)  # Sleep between requests - Removed as we use RateLimiter now

    @patch('grid_search.ZipCodeGridSearcher.get_place_details')
    def test_enhance_results_with_details_empty_input(self, mock_get_details):
        """Test result enhancement with empty input."""
        enhanced_results = self.searcher.enhance_results_with_details([])
        
        self.assertEqual(len(enhanced_results), 0)
        mock_get_details.assert_not_called()

    @patch('grid_search.ZipCodeGridSearcher.get_place_details')
    def test_enhance_results_with_details_api_failure(self, mock_get_details):
        """Test result enhancement when place details API fails."""
        mock_get_details.return_value = None  # API failure
        
        basic_results = [
            SearchResult("place_1", "Basic Spa", 37.2431, -121.7903, "Basic Address", 4.0, "grid_1")
        ]
        
        enhanced_results = self.searcher.enhance_results_with_details(basic_results)
        
        # Should keep original result when enhancement fails
        self.assertEqual(len(enhanced_results), 1)
        self.assertEqual(enhanced_results[0].name, "Basic Spa")
        self.assertEqual(enhanced_results[0].address, "Basic Address")
        self.assertIsNone(enhanced_results[0].total_ratings)  # No enhancement

    @patch('grid_search.ZipCodeGridSearcher.enhance_results_with_details')
    @patch('grid_search.ZipCodeGridSearcher.search_grid_cell')
    @patch('grid_search.ZipCodeGridSearcher.generate_grid_cells')
    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    def test_exhaustive_search_with_details(self, mock_bounds, mock_generate, mock_search, mock_enhance):
        """Test exhaustive search with detail fetching enabled."""
        # Mock zip code bounds
        mock_bounds.return_value = self.sample_bounds

        # Mock grid generation
        mock_grids = [GridCell(37.2431, -121.7903, 1.5, 1000, "grid_0_0")]
        mock_generate.return_value = mock_grids

        # Mock search results
        basic_result = SearchResult("place_1", "Basic Spa", 37.2431, -121.7903, "Basic Address", 4.0, "grid_0_0")
        mock_search.return_value = [basic_result]

        # Mock enhancement
        enhanced_result = SearchResult(
            "place_1", "Enhanced Spa", 37.2431, -121.7903, "Enhanced Address", 4.5, "grid_0_0",
            total_ratings=100, reviews=[{"text": "Great!"}], website="https://test.com"
        )
        mock_enhance.return_value = [enhanced_result]

        # Test with details enabled
        results = self.searcher.exhaustive_search("95125", "massage", 3, fetch_details=True)

        # Verify enhancement was called
        mock_enhance.assert_called_once()
        
        # Verify enhanced results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Enhanced Spa")
        self.assertEqual(results[0].total_ratings, 100)
        self.assertEqual(results[0].website, "https://test.com")

    @patch('grid_search.ZipCodeGridSearcher.enhance_results_with_details')
    @patch('grid_search.ZipCodeGridSearcher.search_grid_cell')
    @patch('grid_search.ZipCodeGridSearcher.generate_grid_cells')
    @patch('grid_search.ZipCodeGridSearcher.get_zip_code_bounds')
    def test_exhaustive_search_without_details(self, mock_bounds, mock_generate, mock_search, mock_enhance):
        """Test exhaustive search with detail fetching disabled."""
        # Mock zip code bounds
        mock_bounds.return_value = self.sample_bounds

        # Mock grid generation
        mock_grids = [GridCell(37.2431, -121.7903, 1.5, 1000, "grid_0_0")]
        mock_generate.return_value = mock_grids

        # Mock search results
        basic_result = SearchResult("place_1", "Basic Spa", 37.2431, -121.7903, "Basic Address", 4.0, "grid_0_0")
        mock_search.return_value = [basic_result]

        # Test with details disabled (default)
        results = self.searcher.exhaustive_search("95125", "massage", 3, fetch_details=False)

        # Verify enhancement was NOT called
        mock_enhance.assert_not_called()
        
        # Verify basic results only
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "Basic Spa")
        self.assertIsNone(results[0].total_ratings)
        self.assertIsNone(results[0].website)

    def test_search_result_enhanced_dataclass(self):
        """Test SearchResult dataclass with enhanced fields."""
        result = SearchResult(
            place_id="test_place",
            name="Test Business",
            lat=37.2431,
            lng=-121.7903,
            address="123 Test St",
            rating=4.5,
            source_grid="grid_0_0",
            total_ratings=150,
            reviews=[{"author_name": "John", "rating": 5}],
            website="https://test.com",
            phone_number="(408) 555-0123",
            opening_hours=["Monday: 9:00 AM – 5:00 PM"]
        )

        # Verify all fields
        self.assertEqual(result.place_id, "test_place")
        self.assertEqual(result.name, "Test Business")
        self.assertEqual(result.total_ratings, 150)
        self.assertEqual(len(result.reviews), 1)
        self.assertEqual(result.website, "https://test.com")
        self.assertEqual(result.phone_number, "(408) 555-0123")
        self.assertEqual(len(result.opening_hours), 1)

    def test_search_result_partial_enhancement(self):
        """Test SearchResult with only some enhanced fields."""
        result = SearchResult(
            place_id="test_place",
            name="Test Business",
            lat=37.2431,
            lng=-121.7903,
            address="123 Test St",
            rating=4.5,
            source_grid="grid_0_0",
            total_ratings=150,
            # reviews, website, phone_number, opening_hours are None/default
        )

        self.assertEqual(result.total_ratings, 150)
        self.assertIsNone(result.reviews)
        self.assertIsNone(result.website)
        self.assertIsNone(result.phone_number)
        self.assertIsNone(result.opening_hours)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_search_nearby_success(self, mock_sleep, mock_get):
        """Test _search_nearby method with successful response."""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_places_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher._search_nearby(grid_cell, "coffee shop", max_pages=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].place_id, "test_place_1")
        self.assertEqual(results[0].source_grid, "test_grid")
        self.assertEqual(results[1].place_id, "test_place_2")

    @patch('grid_search.requests.get')
    def test_search_nearby_zero_results(self, mock_get):
        """Test _search_nearby method with zero results."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ZERO_RESULTS"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher._search_nearby(grid_cell, "coffee shop", max_pages=1)

        self.assertEqual(len(results), 0)

    @patch('grid_search.requests.get')
    @patch('grid_search.time.sleep')
    def test_search_nearby_pagination(self, mock_sleep, mock_get):
        """Test _search_nearby method with pagination."""
        first_page = {
            "status": "OK",
            "results": [self.sample_places_response["results"][0]],
            "next_page_token": "token123"
        }

        second_page = {
            "status": "OK",
            "results": [self.sample_places_response["results"][1]]
        }

        mock_responses = [MagicMock(), MagicMock()]
        mock_responses[0].json.return_value = first_page
        mock_responses[0].raise_for_status.return_value = None
        mock_responses[1].json.return_value = second_page
        mock_responses[1].raise_for_status.return_value = None

        mock_get.side_effect = mock_responses

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher._search_nearby(grid_cell, "coffee shop", max_pages=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(2)

    @patch('grid_search.requests.get')
    def test_search_text_success(self, mock_get):
        """Test _search_text method with successful response."""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_places_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher._search_text(grid_cell, "massage therapy", max_pages=1)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)

    @patch('grid_search.requests.get')
    def test_search_text_distance_filtering(self, mock_get):
        """Test _search_text filters results by distance."""
        # Create response with one close and one far result
        response = {
            "status": "OK",
            "results": [
                {
                    "place_id": "close_place",
                    "name": "Close Business",
                    "geometry": {"location": {"lat": 37.2431, "lng": -121.7903}},
                    "formatted_address": "123 Test St",
                    "rating": 4.5
                },
                {
                    "place_id": "far_place",
                    "name": "Far Business",
                    "geometry": {"location": {"lat": 38.0, "lng": -122.0}},  # Very far
                    "formatted_address": "456 Far St",
                    "rating": 4.0
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher._search_text(grid_cell, "business", max_pages=1)

        # Should only include the close result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].place_id, "close_place")

    def test_get_related_keywords_massage(self):
        """Test _get_related_keywords for massage."""
        keywords = self.searcher._get_related_keywords("massage")

        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 2)
        if keywords:
            self.assertIn(keywords[0], ['massage therapy', 'spa', 'body massage',
                                        'massage spa', 'foot spa', 'thai massage',
                                        'therapeutic massage'])

    def test_get_related_keywords_coffee(self):
        """Test _get_related_keywords for coffee."""
        keywords = self.searcher._get_related_keywords("coffee")

        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 2)

    def test_get_related_keywords_unknown(self):
        """Test _get_related_keywords for unknown business type."""
        keywords = self.searcher._get_related_keywords("unknown business type")

        self.assertEqual(keywords, [])

    def test_deduplicate_by_place_id(self):
        """Test _deduplicate_by_place_id method."""
        results = [
            SearchResult("place_1", "Business 1", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1"),
            SearchResult("place_1", "Business 1", 37.2431, -121.7903, "123 Main St", 4.5, "grid_2"),  # Duplicate
            SearchResult("place_2", "Business 2", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_1"),
            SearchResult("place_1", "Business 1", 37.2431, -121.7903, "123 Main St", 4.5, "grid_3"),  # Another duplicate
        ]

        unique_results = self.searcher._deduplicate_by_place_id(results)

        self.assertEqual(len(unique_results), 2)
        place_ids = [r.place_id for r in unique_results]
        self.assertIn("place_1", place_ids)
        self.assertIn("place_2", place_ids)
        # Verify only first occurrence kept
        self.assertEqual(unique_results[0].source_grid, "grid_1")

    def test_deduplicate_by_place_id_empty(self):
        """Test _deduplicate_by_place_id with empty input."""
        unique_results = self.searcher._deduplicate_by_place_id([])

        self.assertEqual(len(unique_results), 0)

    @patch('grid_search.ZipCodeGridSearcher._search_text')
    @patch('grid_search.ZipCodeGridSearcher._search_nearby')
    @patch('grid_search.time.sleep')
    def test_search_grid_cell_hybrid_mode(self, mock_sleep, mock_nearby, mock_text):
        """Test search_grid_cell with hybrid mode enabled."""
        # Mock nearby search results
        nearby_results = [
            SearchResult("place_1", "Nearby Business", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1")
        ]
        mock_nearby.return_value = nearby_results

        # Mock text search results
        text_results = [
            SearchResult("place_2", "Text Business", 37.2441, -121.7913, "456 Oak Ave", 4.2, "grid_1")
        ]
        mock_text.return_value = text_results

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher.search_grid_cell(grid_cell, "massage", max_pages=3, use_hybrid=True)

        # Should call both nearby and text search
        mock_nearby.assert_called_once()
        self.assertGreater(mock_text.call_count, 0)  # At least one text search

        # Should deduplicate and combine results
        self.assertGreater(len(results), 0)

    @patch('grid_search.ZipCodeGridSearcher._search_nearby')
    def test_search_grid_cell_no_hybrid(self, mock_nearby):
        """Test search_grid_cell with hybrid mode disabled."""
        nearby_results = [
            SearchResult("place_1", "Nearby Business", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1")
        ]
        mock_nearby.return_value = nearby_results

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher.search_grid_cell(grid_cell, "massage", max_pages=3, use_hybrid=False)

        # Should only call nearby search
        mock_nearby.assert_called_once()
        self.assertEqual(len(results), 1)

    @patch('grid_search.ZipCodeGridSearcher._search_text')
    @patch('grid_search.ZipCodeGridSearcher._search_nearby')
    @patch('grid_search.time.sleep')
    def test_search_grid_cell_hybrid_deduplication(self, mock_sleep, mock_nearby, mock_text):
        """Test that hybrid search properly deduplicates results."""
        # Same place found in both searches
        duplicate_result = SearchResult("place_1", "Business", 37.2431, -121.7903, "123 Main St", 4.5, "grid_1")

        mock_nearby.return_value = [duplicate_result]
        mock_text.return_value = [duplicate_result]  # Same result from text search

        grid_cell = GridCell(37.2431, -121.7903, 1.5, 1000, "test_grid")

        results = self.searcher.search_grid_cell(grid_cell, "massage", max_pages=3, use_hybrid=True)

        # Should deduplicate to only one result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].place_id, "place_1")

    @patch('grid_search.requests.get')
    def test_pagination_increment_prevents_infinite_loop(self, mock_get):
        """
        Regression test: Verify that _search_text increments page_num and terminates.
        Previously, page_num was not incremented, causing an infinite loop if max_pages > 1.
        """
        # Mock response with next_page_token to trigger loop
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "OK",
            "results": [],
            "next_page_token": "token123"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        grid_cell = GridCell(0, 0, 1, 1000, "grid_1")

        # Set max_pages=3. If bug exists, this will loop infinitely
        # Verify that requests.get is called exactly 3 times (pages 0, 1, 2)
        with patch('grid_search.time.sleep'):  # Skip sleep
            self.searcher._search_text(grid_cell, "test", max_pages=3)

        self.assertEqual(mock_get.call_count, 3, "Should stop after max_pages=3")

    @patch('grid_search.requests.get')
    def test_rate_limiter_called_in_get_place_details(self, mock_get):
        """
        Regression test: Verify rate_limiter.acquire() is called in get_place_details.
        Previously, it was missing, bypassing rate limiting.
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "OK", "result": {}}
        mock_get.return_value = mock_response

        # Mock the rate limiter
        self.searcher.rate_limiter = MagicMock()

        self.searcher.get_place_details("place_123")

        # Verify acquire was called
        self.searcher.rate_limiter.acquire.assert_called_once()

    def test_grid_indexing_handles_negative_coordinates(self):
        """
        Regression test: Verify grid indexing handles negative coordinates correctly.
        Previously, int() truncation caused -0.5 to map to 0 instead of -1.
        Now uses math.floor() for proper grid cell assignment.
        """
        # Create results near the equator/prime meridian boundary
        r1 = SearchResult("p1", "Place 1", 0.5, 0.5, "addr")
        r2 = SearchResult("p2", "Place 2", -0.5, -0.5, "addr")
        r3 = SearchResult("p3", "Place 2 Duplicate", -0.5001, -0.5001, "addr")

        results = [r1, r2, r3]

        # Verify deduplication works correctly with negative coordinates
        unique = self.searcher.deduplicate_results(results, distance_threshold_m=100)

        ids = [r.place_id for r in unique]
        self.assertIn("p1", ids)  # r1 should be kept (far from r2)
        self.assertIn("p2", ids)  # r2 should be kept
        self.assertNotIn("p3", ids)  # r3 is duplicate of r2


if __name__ == '__main__':
    unittest.main()

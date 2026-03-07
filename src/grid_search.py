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

# src/grid_search.py
"""
Grid Search Module for Exhaustive Business Discovery

This module implements geographic grid-based searching to overcome Google Places API's
60-result limit per query. It divides a zip code area into smaller grids and searches
each grid separately to achieve comprehensive coverage.

Key Features:
- Adaptive grid sizing based on business density
- Overlapping grids to prevent missed businesses
- Intelligent deduplication
- Cost optimization through smart grid placement
"""

import concurrent.futures
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from security_utils import safe_write_text
from rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

REQUEST_TIMEOUT = 30  # seconds

@dataclass
class GridCell:
    """Represents a single grid cell for searching"""
    center_lat: float
    center_lng: float
    size_km: float
    search_radius_m: int
    grid_id: str

@dataclass
class SearchResult:
    """Standardized search result from any API"""
    place_id: str
    name: str
    lat: float
    lng: float
    address: str
    rating: Optional[float] = None
    source_grid: Optional[str] = None
    # Detailed information (fetched separately)
    total_ratings: Optional[int] = None
    reviews: Optional[List[Dict]] = None
    website: Optional[str] = None
    phone_number: Optional[str] = None
    opening_hours: Optional[List[str]] = None

class ZipCodeGridSearcher:
    """
    Implements exhaustive business search using geographic grid strategy
    """
    
    def __init__(self, api_key: str, requests_per_second: int = 5):
        self.api_key = api_key
        self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.places_text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        self.places_nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        self.place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(max_calls=requests_per_second, period_seconds=1.0)
        
        # Business density estimates (businesses per km²)
        self.business_density = {
            'coffee shop': 15,
            'cafe': 15,
            'restaurant': 25,
            'gas station': 3,
            'pharmacy': 5,
            'bank': 8,
            'grocery store': 4,
            'hotel': 6,
            'gym': 7,
            'default': 10
        }
    
    def get_zip_code_bounds(self, zip_code: str) -> Optional[Dict[str, float]]:
        """
        Get geographic bounds for a zip code using Google Geocoding API
        
        Returns:
            Dict with keys: north, south, east, west, center_lat, center_lng, area_km2
        """
        params = {
            'address': zip_code,
            'key': self.api_key,
            'components': 'country:US'  # Restrict to US zip codes
        }
        
        try:
            response = requests.get(self.geocoding_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'OK' or not data['results']:
                print(f"❌ Could not geocode zip code {zip_code}: {data.get('status', 'Unknown error')}")
                return None
            
            result = data['results'][0]
            geometry = result['geometry']
            
            # Get bounds
            if 'bounds' in geometry:
                bounds = geometry['bounds']
                northeast = bounds['northeast']
                southwest = bounds['southwest']
            else:
                # Use viewport if bounds not available
                viewport = geometry['viewport']
                northeast = viewport['northeast']
                southwest = viewport['southwest']
            
            # Calculate center and area
            center_lat = (northeast['lat'] + southwest['lat']) / 2
            center_lng = (northeast['lng'] + southwest['lng']) / 2
            
            # Approximate area calculation (rough estimate)
            lat_diff = northeast['lat'] - southwest['lat']
            lng_diff = northeast['lng'] - southwest['lng']
            # Convert degrees to km (rough approximation)
            area_km2 = lat_diff * lng_diff * 111 * 111 * math.cos(math.radians(center_lat))
            
            return {
                'north': northeast['lat'],
                'south': southwest['lat'],
                'east': northeast['lng'],
                'west': southwest['lng'],
                'center_lat': center_lat,
                'center_lng': center_lng,
                'area_km2': abs(area_km2)
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error getting zip code bounds: {e}")
            return None
        except Exception as e:
            print(f"❌ Error processing zip code bounds: {e}")
            return None
    
    def calculate_optimal_grid_size(self, area_km2: float, business_type: str) -> float:
        """
        Calculate optimal grid size based on area and estimated business density
        
        Args:
            area_km2: Total area to search in square kilometers
            business_type: Type of business (for density estimation)
            
        Returns:
            Optimal grid side length in kilometers
        """
        # Get density estimate
        business_key = business_type.lower()
        estimated_density = self.business_density.get(business_key, self.business_density['default'])
        estimated_density = max(estimated_density, 0.1)  # Avoid divide-by-zero
        
        # Target: ~45 businesses per grid (buffer under 60 limit)
        target_businesses_per_grid = 45
        optimal_grid_area_km2 = target_businesses_per_grid / estimated_density
        
        # Convert to grid side length (assuming square grids)
        grid_side_km = math.sqrt(optimal_grid_area_km2)

        # Clamp between reasonable bounds
        min_grid_size = 0.5  # 500m minimum
        max_grid_size = 3.0  # 3km maximum
        base_grid_side = grid_side_km

        # Adjust grid size so that the total number of grids remains manageable
        # for very large zip codes, but only when the unconstrained grid size
        # already falls within our allowable range (so we don't undo the clamps).
        if (
            area_km2
            and area_km2 > 0
            and min_grid_size <= base_grid_side <= max_grid_size
        ):
            grid_area = grid_side_km ** 2
            estimated_grid_count = max(area_km2 / grid_area, 1e-6)
            max_grids = 40

            if estimated_grid_count > max_grids:
                scale = math.sqrt(estimated_grid_count / max_grids)
                grid_side_km *= scale

        return max(min_grid_size, min(grid_side_km, max_grid_size))
    
    def generate_grid_cells(self, bounds: Dict[str, float], grid_size_km: float,
                           overlap_ratio: float = 0.35) -> List[GridCell]:
        """
        Generate overlapping grid cells covering the zip code area

        Args:
            bounds: Geographic bounds from get_zip_code_bounds()
            grid_size_km: Size of each grid cell in kilometers
            overlap_ratio: Overlap between adjacent grids (0.35 = 35% overlap for better coverage)

        Returns:
            List of GridCell objects
        """
        grid_cells = []

        # Convert km to degrees (rough approximation)
        lat_per_km = 1 / 111.0  # 1 degree ≈ 111 km
        lng_per_km = 1 / (111.0 * math.cos(math.radians(bounds['center_lat'])))

        grid_size_lat = grid_size_km * lat_per_km
        grid_size_lng = grid_size_km * lng_per_km

        # Calculate step size with overlap
        step_lat = grid_size_lat * (1 - overlap_ratio)
        step_lng = grid_size_lng * (1 - overlap_ratio)

        # Generate grid centers
        current_lat = bounds['south'] + grid_size_lat / 2
        grid_row = 0

        while current_lat <= bounds['north'] + grid_size_lat / 2:
            current_lng = bounds['west'] + grid_size_lng / 2
            grid_col = 0

            while current_lng <= bounds['east'] + grid_size_lng / 2:
                # Check if grid center is within bounds (with some tolerance)
                if (bounds['south'] - grid_size_lat/2 <= current_lat <= bounds['north'] + grid_size_lat/2 and
                    bounds['west'] - grid_size_lng/2 <= current_lng <= bounds['east'] + grid_size_lng/2):

                    # Calculate search radius - use 90% of grid size for better coverage
                    search_radius_m = int(grid_size_km * 1000 * 0.9)  # 90% of grid size
                    search_radius_m = max(500, min(search_radius_m, 2000))  # Clamp 500m-2km

                    grid_cell = GridCell(
                        center_lat=current_lat,
                        center_lng=current_lng,
                        size_km=grid_size_km,
                        search_radius_m=search_radius_m,
                        grid_id=f"grid_{grid_row}_{grid_col}"
                    )
                    grid_cells.append(grid_cell)

                current_lng += step_lng
                grid_col += 1

            current_lat += step_lat
            grid_row += 1

        return grid_cells
    
    def search_grid_cell(self, grid_cell: GridCell, business_type: str,
                        max_pages: int = 3, use_hybrid: bool = True) -> List[SearchResult]:
        """
        Search a single grid cell using Google Places API with hybrid approach

        Args:
            grid_cell: GridCell to search
            business_type: Type of business to search for
            max_pages: Maximum pages to fetch (up to 3)
            use_hybrid: If True, combines Nearby + Text Search for better coverage

        Returns:
            List of SearchResult objects
        """
        all_results = []

        # Strategy 1: Nearby Search with primary keyword
        nearby_results = self._search_nearby(grid_cell, business_type, max_pages)
        all_results.extend(nearby_results)

        if use_hybrid:
            # Strategy 2: Text Search with location bias for related terms
            keywords = self._get_related_keywords(business_type)
            for keyword in keywords:
                time.sleep(1)  # Rate limiting between different searches
                text_results = self._search_text(grid_cell, keyword, max_pages=1)
                all_results.extend(text_results)

        # Deduplicate within grid cell results
        unique_results = self._deduplicate_by_place_id(all_results)

        print(f"  ✅ {grid_cell.grid_id} complete: {len(unique_results)} unique results "
              f"(from {len(all_results)} total)")
        return unique_results

    def _search_nearby(self, grid_cell: GridCell, keyword: str,
                       max_pages: int = 3) -> List[SearchResult]:
        """Perform Nearby Search API call"""
        results = []

        params = {
            'location': f"{grid_cell.center_lat},{grid_cell.center_lng}",
            'radius': grid_cell.search_radius_m,
            'keyword': keyword,
            'key': self.api_key
        }

        print(f"🔍 Nearby Search in {grid_cell.grid_id}: '{keyword}' "
              f"({grid_cell.center_lat:.4f}, {grid_cell.center_lng:.4f}) "
              f"radius {grid_cell.search_radius_m}m")

        page_num = 0
        while True:
            if page_num >= max_pages:
                break
            try:
                self.rate_limiter.acquire() # Rate limit check
                response = requests.get(self.places_nearby_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'OK':
                    places = data.get('results', [])
                    print(f"    📍 Page {page_num + 1}: Found {len(places)} places")

                    for place in places:
                        if 'place_id' in place and 'geometry' in place:
                            location = place['geometry']['location']
                            search_result = SearchResult(
                                place_id=place['place_id'],
                                name=place.get('name', 'Unknown'),
                                lat=location['lat'],
                                lng=location['lng'],
                                address=place.get('formatted_address', ''),
                                rating=place.get('rating'),
                                source_grid=grid_cell.grid_id
                            )
                            results.append(search_result)

                    next_page_token = data.get('next_page_token')
                    if next_page_token and page_num < max_pages - 1:
                        params['pagetoken'] = next_page_token
                        print(f"    ⏳ Waiting 2 seconds for next page...")
                        time.sleep(2)
                    else:
                        break

                elif data['status'] == 'ZERO_RESULTS':
                    print(f"    ❌ No results for '{keyword}'")
                    break
                else:
                    print(f"    ❌ API error: {data['status']}")
                    break

            except Exception as e:
                print(f"    ❌ Error: {e}")
                break
            page_num += 1

        return results

    def _search_text(self, grid_cell: GridCell, query: str,
                     max_pages: int = 1) -> List[SearchResult]:
        """Perform Text Search API call with location bias"""
        results = []

        # Create location-biased query
        location_query = f"{query} near {grid_cell.center_lat},{grid_cell.center_lng}"

        params = {
            'query': location_query,
            'location': f"{grid_cell.center_lat},{grid_cell.center_lng}",
            'radius': grid_cell.search_radius_m,
            'key': self.api_key
        }

        print(f"  🔎 Text Search: '{query}'")

        page_num = 0
        while True:
            if page_num >= max_pages:
                break
            try:
                self.rate_limiter.acquire() # Rate limit check
                response = requests.get(self.places_text_search_url, params=params, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if data['status'] == 'OK':
                    places = data.get('results', [])
                    print(f"    📍 Found {len(places)} places")

                    for place in places:
                        if 'place_id' in place and 'geometry' in place:
                            location = place['geometry']['location']

                            # Filter by distance to grid center
                            distance = self._calculate_distance_meters(
                                grid_cell.center_lat, grid_cell.center_lng,
                                location['lat'], location['lng']
                            )

                            # Only include if within search radius
                            if distance <= grid_cell.search_radius_m:
                                search_result = SearchResult(
                                    place_id=place['place_id'],
                                    name=place.get('name', 'Unknown'),
                                    lat=location['lat'],
                                    lng=location['lng'],
                                    address=place.get('formatted_address', ''),
                                    rating=place.get('rating'),
                                    source_grid=grid_cell.grid_id
                                )
                                results.append(search_result)

                    next_page_token = data.get('next_page_token')
                    if next_page_token and page_num < max_pages - 1:
                        params['pagetoken'] = next_page_token
                        time.sleep(2)
                    else:
                        break

                elif data['status'] == 'ZERO_RESULTS':
                    break
                else:
                    print(f"    ⚠️ API status: {data['status']}")
                    break

            except Exception as e:
                print(f"    ❌ Error: {e}")
                break
            
            page_num += 1

        return results

    def _get_related_keywords(self, business_type: str) -> List[str]:
        """Get related search keywords for better coverage"""
        keyword_map = {
            'massage': ['massage therapy', 'spa', 'body massage', 'massage spa',
                       'foot spa', 'thai massage', 'therapeutic massage'],
            'coffee': ['coffee shop', 'cafe', 'espresso bar', 'coffee house'],
            'restaurant': ['dining', 'eatery', 'food'],
            'pharmacy': ['drugstore', 'chemist'],
            'gym': ['fitness', 'health club', 'workout'],
        }

        # Find matching keywords
        business_lower = business_type.lower()
        for key, related in keyword_map.items():
            if key in business_lower:
                return related[:2]  # Limit to 2 related terms to control API usage

        return []  # No related keywords

    def _deduplicate_by_place_id(self, results: List[SearchResult]) -> List[SearchResult]:
        """Quick deduplication by place_id within a single search"""
        seen = set()
        unique = []
        for result in results:
            if result.place_id not in seen:
                seen.add(result.place_id)
                unique.append(result)
        return unique

    def deduplicate_results(self, all_results: List[SearchResult],
                           distance_threshold_m: float = 50.0) -> List[SearchResult]:
        """
        Remove duplicate businesses using spatial indexing for O(N) performance.
        
        Args:
            all_results: List of SearchResult objects
            distance_threshold_m: Distance in meters to consider as duplicate
            
        Returns:
            List of unique SearchResult objects
        """
        if not all_results:
            return []

        logger.info(f"🔄 Deduplicating {len(all_results)} results...")
        
        # 1. Quick deduplication by place_id
        unique_by_id = {}
        for result in all_results:
            if result.place_id not in unique_by_id:
                unique_by_id[result.place_id] = result
        
        initial_unique = list(unique_by_id.values())
        logger.info(f"  Reduced to {len(initial_unique)} results after place_id check")

        # 2. Spatial deduplication
        # Create a spatial grid index
        # Calculate grid size in degrees (approximate)
        # 1 degree lat ~= 111km. 1 degree lng ~= 111km * cos(lat)
        # We use a simplified conversion for grid indexing
        if distance_threshold_m <= 0:
            grid_size_deg = 0.00001  # Use a very small value for near-zero threshold
        else:
            # Grid size approx 2x threshold to ensure we check neighbors
            grid_size_deg = (distance_threshold_m / 111000.0) * 2  # 2x threshold for safety overlap
        spatial_index = defaultdict(list)
        
        final_unique = []
        
        # Sort by quality (rating + review count) so we keep the best version
        # If rating/reviews missing, treat as 0
        initial_unique.sort(
            key=lambda x: (x.rating or 0, x.total_ratings or 0), 
            reverse=True
        )

        for result in initial_unique:
            # Calculate grid key
            lat_idx = int(math.floor(result.lat / grid_size_deg))
            lng_idx = int(math.floor(result.lng / grid_size_deg))
            
            is_duplicate = False
            
            # Check current cell and all 8 neighbors
            for d_lat in [-1, 0, 1]:
                for d_lng in [-1, 0, 1]:
                    neighbor_key = (lat_idx + d_lat, lng_idx + d_lng)
                    
                    if neighbor_key in spatial_index:
                        for existing_result in spatial_index[neighbor_key]:
                            distance = self._calculate_distance_meters(
                                result.lat, result.lng,
                                existing_result.lat, existing_result.lng
                            )
                            
                            if (distance <= distance_threshold_m and 
                                self._names_similar(result.name, existing_result.name)):
                                is_duplicate = True
                                break
                    if is_duplicate:
                        break
                if is_duplicate:
                    break
            
            if not is_duplicate:
                final_unique.append(result)
                spatial_index[(lat_idx, lng_idx)].append(result)

        duplicates_removed = len(all_results) - len(final_unique)
        logger.info(f"✅ Deduplication complete: {len(final_unique)} unique results "
              f"({duplicates_removed} duplicates removed)")
        return final_unique

    def _calculate_distance_meters(self, lat1: float, lng1: float,
                                  lat2: float, lng2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371000  # Earth's radius in meters

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_place_details(self, place_id: str, max_retries: int = 3, retry_count: int = 0) -> Optional[Dict[str, Any]]:
        """
        Gets details for a specific place using Google Places API Place Details.

        Args:
            place_id: Google Place ID to fetch details for
            max_retries: Maximum number of retry attempts for quota exceeded errors
            retry_count: Current retry attempt (used internally for recursion)

        Returns:
            Dictionary of place details or None if an error occurs or max retries exceeded
        """
        params = {
            'place_id': place_id,
            'fields': 'name,formatted_address,rating,reviews,user_ratings_total,geometry,website,formatted_phone_number,opening_hours',
            'key': self.api_key
        }

        try:
            if self.rate_limiter:
                self.rate_limiter.acquire()
            response = requests.get(self.place_details_url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            details_json = response.json()

            status = details_json.get("status")

            if status == "OK":
                return details_json.get("result")
            elif status == "OVER_QUERY_LIMIT":
                if retry_count >= max_retries:
                    print(f"    ❌ Max retries ({max_retries}) exceeded for {place_id}. Quota still exceeded.")
                    return None

                # Exponential backoff: 60s, 120s, 240s
                delay = 60 * (2 ** retry_count)
                print(f"    ⚠️ API quota exceeded. Retry {retry_count + 1}/{max_retries} after {delay}s...")
                time.sleep(delay)
                return self.get_place_details(place_id, max_retries, retry_count + 1)
            else:
                print(f"    Error fetching details for place_id {place_id}: Status '{status}' - {details_json.get('error_message', 'No error message provided.')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"    Network or HTTP error fetching details for place_id {place_id}: {e}")
            return None
        except json.JSONDecodeError:
            print(f"    Error decoding JSON response fetching details for place_id {place_id}.")
            return None
        except Exception as e:
            print(f"    Unexpected error for place_id {place_id}: {str(e)}")
            return None

    def _names_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Check if two business names are similar"""
        if not name1 or not name2:
            return False

        name1_clean = name1.lower().strip()
        name2_clean = name2.lower().strip()

        if name1_clean == name2_clean:
            return True

        if name1_clean in name2_clean or name2_clean in name1_clean:
            return True

        words1 = set(name1_clean.split())
        words2 = set(name2_clean.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1.intersection(words2))
        total_unique = len(words1.union(words2))

        return (overlap / total_unique) >= threshold

    def enhance_results_with_details(self, results: List[SearchResult], max_workers: int = 5) -> List[SearchResult]:
        """
        Enhance search results with detailed place information including reviews, 
        contact info, and operating hours.
        
        Args:
            results: List of basic SearchResult objects
            max_workers: Maximum number of concurrent API requests
            
        Returns:
            List of enhanced SearchResult objects with detailed information
        """
        if not results:
            return results

        logger.info(f"\n🔍 Fetching detailed information for {len(results)} businesses...")
        enhanced_results = []
        
        # Helper function for parallel execution
        def fetch_detail(result):
            try:
                details = self.get_place_details(result.place_id)
                if details:
                    return SearchResult(
                        place_id=result.place_id,
                        name=details.get("name", result.name),
                        lat=result.lat,
                        lng=result.lng,
                        address=details.get("formatted_address", result.address),
                        rating=details.get("rating", result.rating),
                        source_grid=result.source_grid,
                        total_ratings=details.get("user_ratings_total"),
                        reviews=details.get("reviews", []),
                        website=details.get("website"),
                        phone_number=details.get("formatted_phone_number"),
                        opening_hours=details.get("opening_hours", {}).get("weekday_text", [])
                    )
                else:
                    logger.warning(f"    ⚠️ Could not get details for {result.name}, keeping basic info")
                    return result
            except Exception as e:
                logger.error(f"    ⚠️ Error enhancing {result.name}: {str(e)}")
                return result

        # Use ThreadPoolExecutor for parallel fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map returns results in order
            future_to_result = {executor.submit(fetch_detail, result): result for result in results}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_result):
                completed_count += 1
                if completed_count % 5 == 0:
                    logger.info(f"    Progress: {completed_count}/{len(results)} details fetched")
                
                try:
                    enhanced_result = future.result()
                    enhanced_results.append(enhanced_result)
                except Exception as e:
                    logger.error(f"    ❌ Critical error in detail fetch thread: {e}")
                    enhanced_results.append(future_to_result[future]) # Keep original on error
        
        logger.info(f"\n✅ Enhanced {len(enhanced_results)} results with detailed information")
        return enhanced_results

    def exhaustive_search(self, zip_code: str, business_type: str,
                         max_pages_per_grid: int = 3, fetch_details: bool = False,
                         use_hybrid: bool = True, max_workers: int = 3) -> List[SearchResult]:
        """
        Perform exhaustive search for businesses in a zip code using grid strategy
        
        Args:
            zip_code: US zip code to search
            business_type: Type of business (e.g., "coffee shop", "restaurant")
            max_pages_per_grid: Maximum pages to fetch per grid (1-3)
            fetch_details: Whether to fetch detailed information (reviews, hours, etc.)
            use_hybrid: Whether to use hybrid search (Nearby + Text Search) for better coverage
            max_workers: Maximum number of concurrent grid searches

        Returns:
            List of unique SearchResult objects (enhanced with details if fetch_details=True)
        """
        logger.info(f"🚀 Starting exhaustive search: {business_type} in {zip_code}")
        if use_hybrid:
            logger.info(f"   🔀 Hybrid search enabled (Nearby + Text Search)")

        # Step 1: Get zip code bounds
        bounds = self.get_zip_code_bounds(zip_code)
        if not bounds:
            logger.error("❌ Could not get zip code bounds. Aborting search.")
            return []

        logger.info(f"📍 Zip code {zip_code} bounds:")
        logger.info(f"   Area: ~{bounds['area_km2']:.2f} km²")
        logger.info(f"   Center: ({bounds['center_lat']:.4f}, {bounds['center_lng']:.4f})")

        # Step 2: Calculate optimal grid size
        grid_size_km = self.calculate_optimal_grid_size(bounds['area_km2'], business_type)
        logger.info(f"🔧 Calculated optimal grid size: {grid_size_km:.2f} km")

        # Step 3: Generate grid cells
        grid_cells = self.generate_grid_cells(bounds, grid_size_km)
        logger.info(f"🗂️  Generated {len(grid_cells)} grid cells with 35% overlap")

        # Step 4: Search each grid cell in parallel
        all_results = []
        total_api_calls = 0

        logger.info(f"📡 Searching {len(grid_cells)} grids with {max_workers} workers...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_grid = {
                executor.submit(self.search_grid_cell, grid, business_type, max_pages_per_grid, use_hybrid): grid 
                for grid in grid_cells
            }
            
            completed_grids = 0
            for future in concurrent.futures.as_completed(future_to_grid):
                grid = future_to_grid[future]
                completed_grids += 1
                try:
                    grid_results = future.result()
                    all_results.extend(grid_results)
                    
                    # Estimate API calls
                    if use_hybrid:
                        api_calls_this_grid = max_pages_per_grid + 2
                    else:
                        api_calls_this_grid = min(len(grid_results) // 20 + 1, max_pages_per_grid)
                    total_api_calls += api_calls_this_grid
                    
                    logger.info(f"    ✅ Grid {grid.grid_id} finished ({completed_grids}/{len(grid_cells)})")
                    
                except Exception as e:
                    logger.error(f"    ❌ Error searching grid {grid.grid_id}: {e}")

        logger.info(f"\n📊 Grid search complete:")
        logger.info(f"   Total results found: {len(all_results)}")
        logger.info(f"   Estimated API calls: {total_api_calls}")

        # Step 5: Deduplicate results
        unique_results = self.deduplicate_results(all_results)

        # Step 6: Optionally fetch detailed information
        if fetch_details:
            logger.info(f"\n📋 Detail fetching enabled - this will significantly increase API usage!")
            unique_results = self.enhance_results_with_details(unique_results, max_workers=max_workers*2)
            detail_api_calls = len(unique_results)
            logger.info(f"   Additional detail API calls: {detail_api_calls}")
            total_api_calls += detail_api_calls

        logger.info(f"\n🎯 Exhaustive search complete:")
        logger.info(f"   Unique businesses found: {len(unique_results)}")
        logger.info(f"   Coverage: {len(grid_cells)} grids searched")
        if fetch_details:
            logger.info(f"   Search API calls: {total_api_calls - len(unique_results)}")
            logger.info(f"   Detail API calls: {len(unique_results)}")
            logger.info(f"   Total API calls: {total_api_calls}")
            logger.info(f"   Estimated cost: ${total_api_calls * 0.017:.2f} (search + details)")
        else:
            logger.info(f"   Estimated cost: ${total_api_calls * 0.017:.2f} (search calls only)")
            logger.info(f"   💡 Add --details flag to get reviews, hours, and contact info")

        return unique_results


def main():
    """
    Example usage of the grid search functionality
    """
    import argparse

    parser = argparse.ArgumentParser(description='Exhaustive business search using grid strategy')
    parser.add_argument('--zip', required=True, help='US zip code to search')
    parser.add_argument('--business', required=True, help='Type of business (e.g., "massage")')
    parser.add_argument('--pages', type=int, default=3, help='Max pages per grid (1-3)')
    parser.add_argument('--details', action='store_true', help='Fetch detailed info (reviews, hours, contact) - increases API usage significantly')
    parser.add_argument('--hybrid', action='store_true', default=True, help='Use hybrid search (Nearby + Text Search) for better coverage (default: True)')
    parser.add_argument('--no-hybrid', dest='hybrid', action='store_false', help='Disable hybrid search, use only Nearby Search')
    parser.add_argument("--output", type=str, help="Output JSON filename")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum number of concurrent workers (default: 3)")
    parser.add_argument("--requests-per-second", type=int, default=5, help="API requests per second limit (default: 5)")
    
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_MAPS_API_KEY environment variable not set")
        print("   Please set your API key in .env file or environment")
        return

    # Create searcher and run exhaustive search
    try:
        searcher = ZipCodeGridSearcher(api_key, requests_per_second=args.requests_per_second)
        results = searcher.exhaustive_search(
            args.zip, 
            args.business,
            max_pages_per_grid=args.pages,
            fetch_details=args.details,
            use_hybrid=args.hybrid,
            max_workers=args.max_workers
        )
    except Exception as e:
        print(f"Error during search: {e}")
        return

    # Convert results to JSON format
    json_results = []
    for result in results:
        result_dict = {
            'place_id': result.place_id,
            'name': result.name,
            'latitude': result.lat,
            'longitude': result.lng,
            'address': result.address,
            'rating': result.rating,
            'source_grid': result.source_grid
        }
        
        # Add detailed information if available
        if args.details:
            result_dict.update({
                'total_ratings': result.total_ratings,
                'reviews': result.reviews,
                'website': result.website,
                'phone_number': result.phone_number,
                'opening_hours': result.opening_hours
            })
        
        json_results.append(result_dict)

    # Save results
    if args.output:
        output_file = args.output
    else:
        business_clean = args.business.replace(' ', '_').replace('/', '_')
        detail_suffix = "_detailed" if args.details else ""
        output_file = f"exhaustive_{business_clean}_{args.zip}{detail_suffix}.json"

    try:
        # Convert data to JSON string first
        json_content = json.dumps(json_results, indent=2, ensure_ascii=False)
        safe_path = safe_write_text(json_content, output_file)
        print(f"\n💾 Results saved to: {safe_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


if __name__ == "__main__":
    main()

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

# src/data_collector.py
import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time

import requests
from dotenv import load_dotenv  # Import load_dotenv

from security_utils import safe_write_text
from rate_limiter import RateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
REQUEST_TIMEOUT = 30  # seconds
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

BASE_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
BASE_PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"

# Global rate limiter
rate_limiter = None


def get_api_key():
    """
    Lazily resolve the Google Maps API key so that importing this module
    does not immediately fail when the key is injected later (e.g., tests).
    """
    global API_KEY

    if API_KEY:
        return API_KEY

    load_dotenv()
    API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not API_KEY:
        raise EnvironmentError(
            "GOOGLE_MAPS_API_KEY not found. "
            "Ensure it's set in your .env file or process environment."
        )
    return API_KEY


def confirm_execution(args, input_func=input, stdin=None):
    """
    Handle interactive confirmation logic. Returns True if execution should proceed.
    """
    stdin = stdin or sys.stdin
    auto_confirm = args.yes or not stdin.isatty()

    if auto_confirm:
        if args.yes:
            print("Proceeding without confirmation (--yes supplied).")
        else:
            print("Non-interactive session detected; proceeding without confirmation.")
        return True

    response = input_func("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Search cancelled.")
        return False
    return True

# --- API Interaction Functions ---

def search_places_by_query(query, max_pages=1, api_key=None):
    """
    Searches for places using Google Places API Text Search.
    Handles pagination up to max_pages.
    Returns a list of place dictionaries from the search results.
    """
    all_results = []
    api_key = api_key or get_api_key()
    params = {
        'query': query,
        'key': api_key
    }

    print(f"Initiating Text Search for: \"{query}\"")

    for page_num in range(max_pages):
        print(f"  Fetching page {page_num + 1}...")
        try:
            response = requests.get(
                BASE_TEXT_SEARCH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            results_json = response.json()

            status = results_json.get("status")

            if status == "OK":
                results_on_page = results_json.get("results", [])
                print(f"    Found {len(results_on_page)} results on this page.")
                all_results.extend(results_on_page)

                next_page_token = results_json.get("next_page_token")
                if next_page_token and (page_num < max_pages - 1):
                    params['pagetoken'] = next_page_token
                    # Google requires a short delay before using the next_page_token
                    print("    Waiting 2 seconds before fetching next page...")
                    time.sleep(2)
                else:
                    print("  No more pages or max pages reached.")
                    break # No more pages or max_pages reached
            elif status == "ZERO_RESULTS":
                print("  No results found for this query.")
                break
            else:
                # Handle other potential status codes like OVER_QUERY_LIMIT, REQUEST_DENIED, INVALID_REQUEST, UNKNOWN_ERROR
                print(f"  Error in Text Search API response: Status '{status}' - {results_json.get('error_message', 'No error message provided.')}")
                break # Stop processing on API error

        except requests.exceptions.RequestException as e:
            print(f"  Network or HTTP error during Text Search: {e}")
            break # Stop processing on request error
        except json.JSONDecodeError:
            print("  Error decoding JSON response from Text Search API.")
            break # Stop processing on JSON error

    print(f"Text Search complete. Total potential places found: {len(all_results)}")
    return all_results

def get_place_details(place_id, max_retries=3, retry_count=0, api_key=None):
    """
    Gets details for a specific place using Google Places API Place Details.

    Args:
        place_id: Google Place ID to fetch details for
        max_retries: Maximum number of retry attempts for quota exceeded errors
        retry_count: Current retry attempt (used internally for recursion)

    Returns:
        Dictionary of place details or None if an error occurs or max retries exceeded
    """
    api_key = api_key or get_api_key()
    params = {
        'place_id': place_id,
        'fields': 'name,formatted_address,rating,reviews,user_ratings_total,geometry,website,formatted_phone_number,opening_hours',
        'key': api_key
    }

    try:
        if rate_limiter:
            rate_limiter.acquire()
            
        response = requests.get(
            BASE_PLACE_DETAILS_URL,
            params=params,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        details_json = response.json()

        status = details_json.get("status")

        if status == "OK":
            return details_json.get("result")
        elif status == "OVER_QUERY_LIMIT":
            if retry_count >= max_retries:
                logger.error(f"    ❌ Max retries ({max_retries}) exceeded for {place_id}. Quota still exceeded.")
                return None

            # Exponential backoff: 60s, 120s, 240s
            delay = 60 * (2 ** retry_count)
            logger.warning(f"    ⚠️ API quota exceeded. Retry {retry_count + 1}/{max_retries} after {delay}s...")
            time.sleep(delay)
            return get_place_details(
                place_id,
                max_retries,
                retry_count + 1,
                api_key=api_key
            )
        else:
            logger.error(f"    Error fetching details for place_id {place_id}: Status '{status}' - {details_json.get('error_message', 'No error message provided.')}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"    Network or HTTP error fetching details for place_id {place_id}: {e}")
        return None
    except json.JSONDecodeError:
        logger.error(f"    Error decoding JSON response fetching details for place_id {place_id}.")
        return None
    except Exception as e:
        logger.error(f"    Unexpected error for place_id {place_id}: {str(e)}")
        return None


# --- Data Processing ---

def process_businesses_data(search_results, max_workers=5, requests_per_second=5):
    """
    Fetches detailed information for each place found in the search results.
    Returns a list of dictionaries containing processed business info.
    """
    global rate_limiter
    rate_limiter = RateLimiter(max_calls=requests_per_second, period_seconds=1.0)
    
    processed_businesses = []

    if not search_results:
        logger.warning("No search results to process.")
        return processed_businesses

    logger.info(f"\nProcessing details for {len(search_results)} potential places...")
    api_key = get_api_key()

    def process_single_place(place):
        place_id = place.get("place_id")
        place_name = place.get("name", "Unknown Place")

        if not place_id:
            logger.warning(f"    Skipping result ({place_name}) as it has no place_id.")
            return None

        details = get_place_details(place_id, api_key=api_key)

        if details:
            try:
                business_info = {
                    "place_id": place_id,
                    "name": details.get("name"),
                    "address": details.get("formatted_address"),
                    "latitude": details.get("geometry", {}).get("location", {}).get("lat"),
                    "longitude": details.get("geometry", {}).get("location", {}).get("lng"),
                    "rating": details.get("rating"),
                    "total_ratings": details.get("user_ratings_total"),
                    "reviews": details.get("reviews", []),
                    "website": details.get("website"),
                    "phone_number": details.get("formatted_phone_number"),
                    "opening_hours": details.get("opening_hours", {}).get("weekday_text", [])
                }
                logger.info(f"    ✓ Successfully processed {place_name}")
                return business_info
            except Exception as e:
                logger.error(f"    ✗ Error processing details for {place_name}: {str(e)}")
                return None
        else:
            logger.warning(f"    ✗ Could not retrieve details for {place_name} (ID: {place_id}).")
            return None

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_place = {executor.submit(process_single_place, place): place for place in search_results}
        
        for future in concurrent.futures.as_completed(future_to_place):
            result = future.result()
            if result:
                processed_businesses.append(result)

    logger.info("\nDetail processing complete.")
    return processed_businesses

# --- Data Saving ---

def save_data_to_json(data, filename="business_locations.json"):
    """
    Saves the collected business data to a JSON file.

    Returns:
        str: Path to the saved file, or None if saving failed or no data provided
    """
    if not data:
        print("No data to save.")
        return None

    try:
        # Convert data to JSON string first
        json_content = json.dumps(data, indent=4, ensure_ascii=False)
        safe_path = safe_write_text(json_content, filename)
        print(f"Successfully saved {len(data)} business entries to {safe_path}")
        return safe_path
    except Exception as e:
        print(f"Error saving data to JSON file {filename}: {e}")
        return None

# --- Main Execution ---

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Search for businesses using Google Places API')
    parser.add_argument('--city', default="San Francisco", help='City to search in (default: San Francisco)')
    parser.add_argument('--zip', default="94107", help='Zip code to search in (default: 94107)')
    parser.add_argument('--keywords', default="artisan bakery", help='Business keywords to search for (default: artisan bakery)')
    parser.add_argument('--pages', type=int, default=2,
                       help='Number of pages to fetch (default: 2, max: 3, each page has up to 20 results). '
                            'Google Places API returns maximum 60 results (3 pages) per single search query. '
                            'Note: Each page counts as 1 API call, and each place detail counts as 1 API call. '
                            'For example, 3 pages = 3 search calls + up to 60 detail calls.')
    parser.add_argument(
        '-y',
        '--yes',
        action='store_true',
        help='Skip the interactive confirmation prompt and proceed immediately.'
    )
    
    args = parser.parse_args()

    try:
        api_key = get_api_key()
    except EnvironmentError as exc:
        print(str(exc))
        sys.exit(1)

    print("--- Starting Store Location Research (Phase 1: Data Collection) ---")
    print(f"Search parameters:")
    print(f"  City: {args.city}")
    print(f"  Zip Code: {args.zip}")
    print(f"  Keywords: {args.keywords}")
    print(f"  Pages to fetch: {args.pages}")
    # Cap pages at 3 since Google Places API only returns max 3 pages (60 results) per query
    effective_pages = min(args.pages, 3)
    if args.pages > 3:
        print(f"⚠️  Note: Requested {args.pages} pages, but Google Places API returns maximum 3 pages per query.")
        print(f"  Will fetch {effective_pages} pages instead.")

    print(f"  Maximum results possible: {effective_pages * 20} (Google Places API limit: 60 per query)")
    print("\n⚠️  API Usage Warning:")
    print(f"  - Each page of search results = 1 API call")
    print(f"  - Each place detail = 1 API call")
    print(f"  - Total API calls will be: {effective_pages} (search) + up to {effective_pages * 20} (details)")
    print("  - This may impact your API quota and billing")

    if not confirm_execution(args):
        sys.exit(0)

    # Construct the search query
    search_query = f"{args.keywords} in {args.city} {args.zip}"

    # 1. Search for places using the query
    search_results = search_places_by_query(search_query, max_pages=effective_pages, api_key=api_key)

    # 2. Fetch detailed information for each place found
    businesses_data = process_businesses_data(search_results)

    # 3. Save the collected data to a JSON file
    output_filename = f"{args.keywords.replace(' ','_')}_{args.city.replace(' ','_')}_{args.zip}.json"
    output_path = None
    if businesses_data:
        output_path = save_data_to_json(businesses_data, filename=output_filename)
        if output_path:
            print(f"\nFound {len(businesses_data)} businesses out of {len(search_results)} search results")
    else:
        print("\nNo business data was successfully collected to save.")

    print("\n--- Store Location Research (Phase 1) Finished ---")
    if output_path:
        print(f"Look for the JSON output file at: {output_path}")
        print("This data is now ready for Phase 2: Map Visualization.")
    else:
        print(f"No output file was created due to lack of data.")
        print("Try adjusting your search parameters or check your API quota.")


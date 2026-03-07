import unittest
import time
import random
import string
from grid_search import ZipCodeGridSearcher, SearchResult
from dataclasses import dataclass
import logging

# Configure logging to show info during tests
logging.basicConfig(level=logging.INFO)

class TestPerformanceBenchmark(unittest.TestCase):
    def setUp(self):
        self.searcher = ZipCodeGridSearcher("fake_api_key")

    def generate_random_string(self, length=10):
        return ''.join(random.choices(string.ascii_letters, k=length))

    def test_deduplication_performance_large_dataset(self):
        """
        Benchmark deduplication performance with 10,000 items.
        Target: < 1.0 second (O(N) complexity)
        """
        print("\n🚀 Starting Deduplication Performance Benchmark (10,000 items)...")
        
        # Generate 10,000 results
        # Create clusters of duplicates to test the logic
        results = []
        num_unique_places = 8000
        num_duplicates = 2000
        
        # Base unique places
        for i in range(num_unique_places):
            lat = 37.0 + (random.random() * 0.5)
            lng = -122.0 + (random.random() * 0.5)
            results.append(SearchResult(
                place_id=f"place_{i}",
                name=f"UniqueBusiness_{i}_{self.generate_random_string(5)}",
                lat=lat,
                lng=lng,
                address=f"{i} Main St",
                rating=4.5,
                source_grid="grid_1"
            ))
            
        # Add duplicates (same place_id)
        for i in range(1000):
            original = results[i]
            results.append(SearchResult(
                place_id=original.place_id,
                name=original.name,
                lat=original.lat,
                lng=original.lng,
                address=original.address,
                rating=original.rating,
                source_grid="grid_2"
            ))
            
        # Add near duplicates (different place_id, close location, similar name)
        for i in range(1000, 2000):
            original = results[i]
            results.append(SearchResult(
                place_id=f"place_{i}_dup", # Different ID
                name=original.name, # Same name
                lat=original.lat + 0.00001, # Very close (~1m)
                lng=original.lng + 0.00001,
                address=original.address,
                rating=original.rating,
                source_grid="grid_2"
            ))
            
        random.shuffle(results)
        print(f"    Generated {len(results)} results.")
        
        start_time = time.time()
        unique_results = self.searcher.deduplicate_results(results)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"    ✅ Deduplication took {duration:.4f} seconds")
        print(f"    Result count: {len(unique_results)} (Expected ~{num_unique_places})")
        
        # Assertions
        self.assertLess(duration, 1.0, "Deduplication took too long! Target < 1.0s")
        self.assertEqual(len(unique_results), num_unique_places, "Deduplication logic failed to remove correct number of duplicates")

if __name__ == '__main__':
    unittest.main()

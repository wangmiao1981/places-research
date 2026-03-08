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
import time
import threading
from src.rate_limiter import RateLimiter

class TestRateLimiter(unittest.TestCase):
    def test_rate_limiting(self):
        """
        Verify that the rate limiter restricts calls to the specified rate.
        """
        max_calls = 5
        period = 1.0
        limiter = RateLimiter(max_calls=max_calls, period_seconds=period)
        
        start_time = time.time()
        
        # Make 10 calls. 
        # First 5 should be instant.
        # Next 5 should wait for ~1 second.
        for i in range(10):
            limiter.acquire()
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRate Limiter Test: 10 calls with limit {max_calls}/s took {duration:.4f}s")
        
        # Expect duration to be at least 1.0 second (since we need a second window)
        # But less than 2.0 seconds
        self.assertGreaterEqual(duration, 1.0)
        self.assertLess(duration, 2.0)

    def test_thread_safety(self):
        """
        Verify rate limiter works correctly with multiple threads.
        """
        max_calls = 10
        period = 1.0
        limiter = RateLimiter(max_calls=max_calls, period_seconds=period)
        
        def worker():
            for _ in range(2):
                limiter.acquire()
                
        threads = []
        start_time = time.time()
        
        # 10 threads, each making 2 calls = 20 calls total
        # Should take ~1 second (first 10 instant, next 10 wait)
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Thread Safety Test: 20 calls (10 threads) with limit {max_calls}/s took {duration:.4f}s")
        
        self.assertGreaterEqual(duration, 1.0)

if __name__ == '__main__':
    unittest.main()

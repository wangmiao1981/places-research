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

import time
import threading
from collections import deque

class RateLimiter:
    """
    Thread-safe rate limiter using a sliding window algorithm.
    Ensures that no more than `max_calls` are made within `period_seconds`.
    """
    def __init__(self, max_calls: int, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        """
        Blocks until a call is allowed.
        """
        while True:
            with self.lock:
                now = time.time()
                
                # Remove calls that are outside the window
                while self.calls and now - self.calls[0] > self.period_seconds:
                    self.calls.popleft()
                
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                
                # Calculate sleep time
                sleep_time = self.calls[0] + self.period_seconds - now
            
            # Sleep outside the lock
            if sleep_time > 0:
                time.sleep(sleep_time)

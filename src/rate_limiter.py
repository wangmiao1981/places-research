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

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

"""
web_searcher.py — Search API wrapper with disk-based caching and rate limiting.

Provides a simple interface over the Tavily search API for market research use.

Usage::

    from web_searcher import WebSearcher

    searcher = WebSearcher()                         # reads TAVILY_API_KEY from env
    results = searcher.search("massage therapy San Jose", max_results=5)
    for r in results:
        print(r.title, r.url, r.score)

    # Batch search
    all_results = searcher.search_multiple(["query A", "query B"])
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tavily import TavilyClient
except ImportError as _tavily_import_error:  # pragma: no cover
    raise ImportError(
        "The 'tavily-python' package is required but not installed.\n"
        "Install it with:  pip install tavily-python>=0.3.0"
    ) from _tavily_import_error

logger = logging.getLogger(__name__)

# Default cache directory relative to the caller's working directory.
_DEFAULT_CACHE_DIR = ".web_search_cache"


@dataclass
class SearchResult:
    """A single search result returned by the search API."""

    title: str
    url: str
    content: str
    score: float

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary (used for JSON caching)."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Deserialize from a plain dictionary (used when loading from cache)."""
        return cls(
            title=data["title"],
            url=data["url"],
            content=data["content"],
            score=float(data.get("score", 0.0)),
        )


class WebSearcher:
    """
    Abstraction over the Tavily web search API with disk caching and rate limiting.

    Parameters
    ----------
    api_key:
        Tavily API key.  If *None* the value of the ``TAVILY_API_KEY``
        environment variable is used.  Raises :class:`EnvironmentError` when
        no key is available.
    cache_dir:
        Directory in which search results are cached as JSON files.
        Created automatically if it does not exist.
        Defaults to ``.web_search_cache`` in the current working directory.
    requests_per_second:
        Maximum number of API requests issued per second.  Defaults to ``1``.
        Set to a larger number to speed up tests (cache hits are never rate
        limited).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        requests_per_second: int = 1,
    ) -> None:
        resolved_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "Tavily API key not found.\n"
                "Set the TAVILY_API_KEY environment variable or pass api_key= "
                "directly to WebSearcher().\n\n"
                "To obtain a key visit https://tavily.com and sign up for a "
                "free account.\n"
                "Then add to your shell:  export TAVILY_API_KEY='tvly-...'"
            )

        self._client = TavilyClient(api_key=resolved_key)
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.requests_per_second = requests_per_second
        self._min_interval: float = 1.0 / requests_per_second
        self._last_request_time: float = 0.0

        cache_path = Path(cache_dir) if cache_dir is not None else Path(_DEFAULT_CACHE_DIR)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Search the web for *query* and return up to *max_results* results.

        Results are cached on disk by a hash of ``(query, max_results)``.
        Subsequent calls with the same arguments return the cached results
        without making an API request.

        Parameters
        ----------
        query:       The search query string.
        max_results: Maximum number of results to return (default 5).

        Returns
        -------
        List[SearchResult]
        """
        cache_key = self._cache_key(query, max_results)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit for query=%r max_results=%d", query, max_results)
            return cached

        logger.debug("Cache miss for query=%r max_results=%d — calling API", query, max_results)
        self._rate_limit()

        try:
            response = self._client.search(query, max_results=max_results)
        except Exception as exc:
            raise RuntimeError(
                f"Tavily search failed for query {query!r}: {exc}"
            ) from exc

        results = self._parse_response(response)
        self._save_cache(cache_key, results)
        return results

    def search_multiple(
        self, queries: List[str], max_results: int = 5
    ) -> Dict[str, List[SearchResult]]:
        """
        Run :meth:`search` for each query in *queries* and return a mapping
        of ``{query: results}``.

        Parameters
        ----------
        queries:     List of query strings.
        max_results: Passed through to each :meth:`search` call.

        Returns
        -------
        Dict[str, List[SearchResult]]
        """
        return {query: self.search(query, max_results=max_results) for query in queries}

    def clear_cache(self) -> None:
        """Delete all cached JSON files from the cache directory."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except OSError as exc:  # pragma: no cover
                logger.warning("Could not delete cache file %s: %s", cache_file, exc)
        logger.debug("Cache cleared: %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(query: str, max_results: int) -> str:
        """Return the SHA-256 hex digest of ``'<query>:<max_results>'``."""
        raw = f"{query}:{max_results}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_cache(self, cache_key: str) -> Optional[List[SearchResult]]:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return [SearchResult.from_dict(item) for item in data]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to load cache file %s: %s", path, exc)
            return None

    def _save_cache(self, cache_key: str, results: List[SearchResult]) -> None:
        path = self._cache_path(cache_key)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump([r.to_dict() for r in results], fh, ensure_ascii=False, indent=2)
        except OSError as exc:  # pragma: no cover
            logger.warning("Failed to write cache file %s: %s", path, exc)

    def _rate_limit(self) -> None:
        """Sleep if needed to stay within *requests_per_second*."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _parse_response(response: dict) -> List[SearchResult]:
        """Convert a raw Tavily API response dict into SearchResult objects."""
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in response.get("results", [])
        ]

import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestSearchResult(unittest.TestCase):
    def test_to_dict(self):
        from web_searcher import SearchResult
        r = SearchResult(title="Test", url="http://example.com", content="body", score=0.9)
        d = r.to_dict()
        self.assertEqual(d["title"], "Test")
        self.assertEqual(d["url"], "http://example.com")
        self.assertEqual(d["content"], "body")
        self.assertEqual(d["score"], 0.9)

    def test_from_dict(self):
        from web_searcher import SearchResult
        d = {"title": "Test", "url": "http://example.com", "content": "body", "score": 0.8}
        r = SearchResult.from_dict(d)
        self.assertEqual(r.title, "Test")
        self.assertEqual(r.score, 0.8)

    def test_from_dict_default_score(self):
        from web_searcher import SearchResult
        d = {"title": "T", "url": "http://u.com", "content": "c"}
        r = SearchResult.from_dict(d)
        self.assertEqual(r.score, 0.0)

    def test_round_trip(self):
        from web_searcher import SearchResult
        original = SearchResult(title="A", url="http://a.com", content="c", score=0.5)
        restored = SearchResult.from_dict(original.to_dict())
        self.assertEqual(original.title, restored.title)
        self.assertEqual(original.score, restored.score)


class TestWebSearcherInit(unittest.TestCase):
    @patch("web_searcher.TavilyClient")
    def test_init_with_api_key(self, mock_tavily):
        from web_searcher import WebSearcher
        tmpdir = tempfile.mkdtemp()
        ws = WebSearcher(api_key="test_key", cache_dir=tmpdir)
        mock_tavily.assert_called_once_with(api_key="test_key")

    @patch("web_searcher.TavilyClient")
    @patch.dict(os.environ, {"TAVILY_API_KEY": "env_key"})
    def test_init_from_env(self, mock_tavily):
        from web_searcher import WebSearcher
        tmpdir = tempfile.mkdtemp()
        ws = WebSearcher(cache_dir=tmpdir)
        mock_tavily.assert_called_once_with(api_key="env_key")

    def test_init_no_key_raises(self):
        from web_searcher import WebSearcher
        env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                WebSearcher(cache_dir="/tmp/test_cache")
            self.assertIn("TAVILY_API_KEY", str(ctx.exception))

    @patch("web_searcher.TavilyClient")
    def test_custom_rate_limit(self, mock_tavily):
        from web_searcher import WebSearcher
        tmpdir = tempfile.mkdtemp()
        ws = WebSearcher(api_key="key", cache_dir=tmpdir, requests_per_second=2)
        self.assertEqual(ws.requests_per_second, 2)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tavily_patcher = patch("web_searcher.TavilyClient")
        self.mock_tavily_cls = self.tavily_patcher.start()
        self.mock_client = MagicMock()
        self.mock_tavily_cls.return_value = self.mock_client

    def tearDown(self):
        self.tavily_patcher.stop()

    def _make_searcher(self):
        from web_searcher import WebSearcher
        return WebSearcher(api_key="test_key", cache_dir=self.tmpdir)

    def test_search_returns_results(self):
        self.mock_client.search.return_value = {
            "results": [
                {"title": "Result 1", "url": "http://r1.com", "content": "body1", "score": 0.95},
                {"title": "Result 2", "url": "http://r2.com", "content": "body2", "score": 0.85},
            ]
        }
        ws = self._make_searcher()
        results = ws.search("test query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Result 1")
        self.assertEqual(results[1].score, 0.85)

    def test_search_caches_to_disk(self):
        self.mock_client.search.return_value = {
            "results": [{"title": "Cached", "url": "http://c.com", "content": "c", "score": 0.9}]
        }
        ws = self._make_searcher()
        ws.search("cache test")
        cache_files = list(Path(self.tmpdir).glob("*.json"))
        self.assertEqual(len(cache_files), 1)

    def test_cache_hit_skips_api(self):
        self.mock_client.search.return_value = {
            "results": [{"title": "First", "url": "http://f.com", "content": "f", "score": 0.9}]
        }
        ws = self._make_searcher()
        r1 = ws.search("same query")
        r2 = ws.search("same query")
        self.mock_client.search.assert_called_once()
        self.assertEqual(r1[0].title, r2[0].title)

    def test_search_passes_max_results(self):
        self.mock_client.search.return_value = {"results": []}
        ws = self._make_searcher()
        ws.search("q", max_results=3)
        self.mock_client.search.assert_called_once_with("q", max_results=3)

    def test_search_api_error_raises_runtime(self):
        self.mock_client.search.side_effect = Exception("API down")
        ws = self._make_searcher()
        with self.assertRaises(RuntimeError) as ctx:
            ws.search("fail query")
        self.assertIn("API down", str(ctx.exception))

    def test_search_empty_results(self):
        self.mock_client.search.return_value = {"results": []}
        ws = self._make_searcher()
        results = ws.search("empty")
        self.assertEqual(results, [])

    def test_search_missing_fields_in_result(self):
        self.mock_client.search.return_value = {"results": [{}]}
        ws = self._make_searcher()
        results = ws.search("partial")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "")
        self.assertEqual(results[0].score, 0.0)


class TestRateLimiting(unittest.TestCase):
    @patch("web_searcher.time.sleep")
    @patch("web_searcher.time.monotonic")
    @patch("web_searcher.TavilyClient")
    def test_rate_limiting_sleeps(self, mock_tavily_cls, mock_monotonic, mock_sleep):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {"results": []}
        # Simulate two rapid calls at same time
        mock_monotonic.side_effect = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        tmpdir = tempfile.mkdtemp()
        from web_searcher import WebSearcher
        ws = WebSearcher(api_key="key", cache_dir=tmpdir, requests_per_second=1)
        ws.search("q1")
        ws.search("q2")
        self.assertTrue(mock_sleep.called)


class TestSearchMultiple(unittest.TestCase):
    @patch("web_searcher.TavilyClient")
    def test_search_multiple_returns_dict(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [{"title": "R", "url": "http://r.com", "content": "c", "score": 0.5}]
        }
        tmpdir = tempfile.mkdtemp()
        from web_searcher import WebSearcher
        ws = WebSearcher(api_key="key", cache_dir=tmpdir)
        results = ws.search_multiple(["q1", "q2"])
        self.assertIn("q1", results)
        self.assertIn("q2", results)
        self.assertEqual(len(results["q1"]), 1)


class TestClearCache(unittest.TestCase):
    @patch("web_searcher.TavilyClient")
    def test_clear_cache_removes_files(self, mock_tavily_cls):
        tmpdir = tempfile.mkdtemp()
        for i in range(3):
            with open(os.path.join(tmpdir, f"cache_{i}.json"), "w") as f:
                json.dump({}, f)
        from web_searcher import WebSearcher
        ws = WebSearcher(api_key="key", cache_dir=tmpdir)
        ws.clear_cache()
        json_files = list(Path(tmpdir).glob("*.json"))
        self.assertEqual(len(json_files), 0)


class TestCacheCorruption(unittest.TestCase):
    @patch("web_searcher.TavilyClient")
    def test_corrupted_cache_refetches(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [{"title": "Fresh", "url": "http://f.com", "content": "c", "score": 0.8}]
        }
        tmpdir = tempfile.mkdtemp()
        from web_searcher import WebSearcher
        ws = WebSearcher(api_key="key", cache_dir=tmpdir)
        # Write corrupted cache
        import hashlib
        cache_key = hashlib.sha256("corrupt:5".encode()).hexdigest()
        with open(os.path.join(tmpdir, f"{cache_key}.json"), "w") as f:
            f.write("NOT VALID JSON{{{")
        results = ws.search("corrupt")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Fresh")


@unittest.skip("Requires TAVILY_API_KEY — run manually for integration testing")
class TestIntegrationSearch(unittest.TestCase):
    def test_real_search(self):
        from web_searcher import WebSearcher
        ws = WebSearcher()
        results = ws.search("average commercial rent San Jose 95125", max_results=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(all(r.title for r in results))


if __name__ == "__main__":
    unittest.main()

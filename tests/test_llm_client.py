"""
Tests for LLM client module — Claude wrapper with retry and token tracking.

Unit tests use mocked API calls. Integration tests are skipped by default
(require a real ANTHROPIC_API_KEY or AWS credentials).
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, call

# ---------------------------------------------------------------------------
# Helpers to build mock response objects
# ---------------------------------------------------------------------------

def _make_usage(input_tokens: int, output_tokens: int) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_content(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    return block


def _make_message(text: str, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    msg = MagicMock()
    msg.content = [_make_content(text)]
    msg.usage = _make_usage(input_tokens, output_tokens)
    return msg


# ---------------------------------------------------------------------------
# Fake exceptions that mimic anthropic SDK error hierarchy
# ---------------------------------------------------------------------------

class _FakeAPIStatusError(Exception):
    """Simulates anthropic.APIStatusError."""
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
        self.response = MagicMock()
        self.response.status_code = status_code
        self.body = {"error": {"message": message}}


class _FakeRateLimitError(_FakeAPIStatusError):
    """Simulates anthropic.RateLimitError (429)."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429)


class _FakeInternalServerError(_FakeAPIStatusError):
    """Simulates anthropic.InternalServerError (500)."""
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, 500)


class _FakeBadRequestError(_FakeAPIStatusError):
    """Simulates anthropic.BadRequestError (400)."""
    def __init__(self, message: str = "Bad request"):
        super().__init__(message, 400)


# ---------------------------------------------------------------------------
# Module import under controlled patch
# ---------------------------------------------------------------------------

# We patch anthropic at import time so the module can import even without
# the real SDK installed.
_anthropic_mock = MagicMock()
_anthropic_bedrock_mock = MagicMock()

# Wire up exception classes on the mock
_anthropic_mock.RateLimitError = _FakeRateLimitError
_anthropic_mock.InternalServerError = _FakeInternalServerError
_anthropic_mock.BadRequestError = _FakeBadRequestError
_anthropic_mock.APIStatusError = _FakeAPIStatusError

with patch.dict("sys.modules", {
    "anthropic": _anthropic_mock,
    "anthropic_bedrock": _anthropic_bedrock_mock,
}):
    import llm_client
    from llm_client import LLMClient, LLMError


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

class TestLLMClientInit(unittest.TestCase):

    def _make_client(self, env=None):
        env = env or {}
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            with patch.dict(os.environ, env, clear=False):
                return LLMClient.__new__(LLMClient)

    def test_init_with_api_key_env(self):
        """Client initialises with ANTHROPIC_API_KEY from environment."""
        _anthropic_mock.Anthropic.reset_mock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-123"}, clear=False):
            with patch.dict("sys.modules", {
                "anthropic": _anthropic_mock,
                "anthropic_bedrock": _anthropic_bedrock_mock,
            }):
                client = LLMClient(api_key="test-key-123")
        self.assertIsNotNone(client)
        self.assertEqual(client.auth_method, "api_key")

    def test_init_with_explicit_api_key(self):
        """Client accepts explicit api_key parameter."""
        _anthropic_mock.Anthropic.reset_mock()
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            client = LLMClient(api_key="explicit-key")
        self.assertEqual(client.auth_method, "api_key")

    def test_init_with_bedrock(self):
        """Client initialises with Bedrock auth when force_bedrock=True."""
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            client = LLMClient(force_bedrock=True)
        self.assertEqual(client.auth_method, "bedrock")

    def test_auto_detect_prefers_api_key(self):
        """Auto-detection prefers API key over Bedrock when both available."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "some-key"}, clear=False):
            with patch.dict("sys.modules", {
                "anthropic": _anthropic_mock,
                "anthropic_bedrock": _anthropic_bedrock_mock,
            }):
                client = LLMClient()
        self.assertEqual(client.auth_method, "api_key")

    def test_auto_detect_falls_back_to_bedrock(self):
        """Auto-detection falls back to Bedrock when no API key present."""
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {
                "anthropic": _anthropic_mock,
                "anthropic_bedrock": _anthropic_bedrock_mock,
            }):
                client = LLMClient()
        self.assertEqual(client.auth_method, "bedrock")

    def test_default_model(self):
        """Default model is claude-sonnet-4-6."""
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            client = LLMClient(api_key="k")
        self.assertEqual(client.default_model, "claude-sonnet-4-6")

    def test_custom_default_model(self):
        """Custom default model is respected."""
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            client = LLMClient(api_key="k", default_model="claude-opus-4-6")
        self.assertEqual(client.default_model, "claude-opus-4-6")

    def test_initial_usage_is_zero(self):
        """Usage counters start at zero."""
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            client = LLMClient(api_key="k")
        summary = client.get_usage_summary()
        self.assertEqual(summary["total_input_tokens"], 0)
        self.assertEqual(summary["total_output_tokens"], 0)
        self.assertEqual(summary["total_calls"], 0)
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.0)


# ---------------------------------------------------------------------------
# Test: call() — happy path
# ---------------------------------------------------------------------------

class TestLLMClientCall(unittest.TestCase):

    def setUp(self):
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            self.client = LLMClient(api_key="test-key")
        # Reset the mock client each time
        self.mock_messages = MagicMock()
        self.client._client = MagicMock()
        self.client._client.messages = self.mock_messages

    def test_call_returns_text(self):
        """call() returns the assistant text content."""
        self.mock_messages.create.return_value = _make_message("Hello, world!")
        result = self.client.call(system="You are helpful.", user="Say hi.")
        self.assertEqual(result, "Hello, world!")

    def test_call_uses_default_model(self):
        """call() uses the client's default model when none specified."""
        self.mock_messages.create.return_value = _make_message("ok")
        self.client.call(system="sys", user="usr")
        _, kwargs = self.mock_messages.create.call_args
        self.assertEqual(kwargs.get("model") or self.mock_messages.create.call_args[0][0]
                         if self.mock_messages.create.call_args[0]
                         else kwargs["model"],
                         self.client.default_model)

    def test_call_uses_custom_model(self):
        """call() accepts an explicit model parameter."""
        self.mock_messages.create.return_value = _make_message("ok")
        self.client.call(system="sys", user="usr", model="claude-haiku-4-6")
        _, kwargs = self.mock_messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-haiku-4-6")

    def test_call_uses_custom_max_tokens(self):
        """call() accepts a custom max_tokens parameter."""
        self.mock_messages.create.return_value = _make_message("ok")
        self.client.call(system="sys", user="usr", max_tokens=512)
        _, kwargs = self.mock_messages.create.call_args
        self.assertEqual(kwargs["max_tokens"], 512)

    def test_call_tracks_token_usage(self):
        """call() accumulates token usage after each call."""
        self.mock_messages.create.return_value = _make_message("resp", 120, 60)
        self.client.call(system="sys", user="usr")
        summary = self.client.get_usage_summary()
        self.assertEqual(summary["total_input_tokens"], 120)
        self.assertEqual(summary["total_output_tokens"], 60)
        self.assertEqual(summary["total_calls"], 1)

    def test_call_accumulates_usage_across_calls(self):
        """Token usage accumulates across multiple calls."""
        self.mock_messages.create.side_effect = [
            _make_message("first", 100, 50),
            _make_message("second", 200, 80),
        ]
        self.client.call(system="sys", user="first")
        self.client.call(system="sys", user="second")
        summary = self.client.get_usage_summary()
        self.assertEqual(summary["total_input_tokens"], 300)
        self.assertEqual(summary["total_output_tokens"], 130)
        self.assertEqual(summary["total_calls"], 2)

    def test_call_passes_system_and_user_message(self):
        """call() correctly passes system prompt and user message to the API."""
        self.mock_messages.create.return_value = _make_message("ok")
        self.client.call(system="Be concise.", user="What is 2+2?")
        _, kwargs = self.mock_messages.create.call_args
        self.assertEqual(kwargs["system"], "Be concise.")
        messages = kwargs["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "What is 2+2?")


# ---------------------------------------------------------------------------
# Test: Retry logic
# ---------------------------------------------------------------------------

class TestLLMClientRetry(unittest.TestCase):

    def setUp(self):
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            self.client = LLMClient(api_key="test-key")
        self.client._client = MagicMock()
        self.mock_messages = MagicMock()
        self.client._client.messages = self.mock_messages

    @patch("time.sleep", return_value=None)
    def test_retry_on_rate_limit(self, mock_sleep):
        """Retries on 429 RateLimitError with exponential backoff."""
        self.mock_messages.create.side_effect = [
            _FakeRateLimitError(),
            _FakeRateLimitError(),
            _make_message("success"),
        ]
        result = self.client.call(system="s", user="u")
        self.assertEqual(result, "success")
        self.assertEqual(self.mock_messages.create.call_count, 3)
        # Should have slept twice (between retries)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("time.sleep", return_value=None)
    def test_exponential_backoff_sleep_durations(self, mock_sleep):
        """Backoff sleep durations grow exponentially (1, 2 seconds for attempts 0,1)."""
        self.mock_messages.create.side_effect = [
            _FakeRateLimitError(),
            _FakeRateLimitError(),
            _make_message("ok"),
        ]
        self.client.call(system="s", user="u")
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # First sleep >= 1, second sleep >= 2
        self.assertGreaterEqual(sleep_calls[0], 1)
        self.assertGreaterEqual(sleep_calls[1], sleep_calls[0])

    @patch("time.sleep", return_value=None)
    def test_retry_on_server_error(self, mock_sleep):
        """Retries on 5xx InternalServerError."""
        self.mock_messages.create.side_effect = [
            _FakeInternalServerError(),
            _make_message("recovered"),
        ]
        result = self.client.call(system="s", user="u")
        self.assertEqual(result, "recovered")
        self.assertEqual(self.mock_messages.create.call_count, 2)

    @patch("time.sleep", return_value=None)
    def test_max_retries_exceeded_raises(self, mock_sleep):
        """Raises LLMError after max retries are exhausted."""
        self.mock_messages.create.side_effect = _FakeRateLimitError()
        with self.assertRaises(LLMError):
            self.client.call(system="s", user="u")
        # 1 original + 3 retries = 4 total attempts
        self.assertEqual(self.mock_messages.create.call_count, 4)

    @patch("time.sleep", return_value=None)
    def test_server_error_max_retries_raises(self, mock_sleep):
        """Raises LLMError after max retries on InternalServerError."""
        self.mock_messages.create.side_effect = _FakeInternalServerError()
        with self.assertRaises(LLMError):
            self.client.call(system="s", user="u")
        self.assertEqual(self.mock_messages.create.call_count, 4)

    @patch("time.sleep", return_value=None)
    def test_no_retry_on_client_error(self, mock_sleep):
        """Does NOT retry on 400 BadRequestError."""
        self.mock_messages.create.side_effect = _FakeBadRequestError()
        with self.assertRaises(LLMError):
            self.client.call(system="s", user="u")
        # Only 1 attempt — no retry
        self.assertEqual(self.mock_messages.create.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("time.sleep", return_value=None)
    def test_no_retry_on_400_status(self, mock_sleep):
        """Does NOT retry on any 4xx error (except 429)."""
        self.mock_messages.create.side_effect = _FakeAPIStatusError("Forbidden", 403)
        with self.assertRaises(LLMError):
            self.client.call(system="s", user="u")
        self.assertEqual(self.mock_messages.create.call_count, 1)

    @patch("time.sleep", return_value=None)
    def test_retry_429_not_other_4xx(self, mock_sleep):
        """429 is retried but other 4xx are not."""
        self.mock_messages.create.side_effect = [
            _FakeRateLimitError(),
            _make_message("ok"),
        ]
        result = self.client.call(system="s", user="u")
        self.assertEqual(result, "ok")
        self.assertEqual(self.mock_messages.create.call_count, 2)


# ---------------------------------------------------------------------------
# Test: Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation(unittest.TestCase):

    def setUp(self):
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            self.client = LLMClient(api_key="k")
        self.client._client = MagicMock()
        self.mock_messages = MagicMock()
        self.client._client.messages = self.mock_messages

    def test_sonnet_cost_estimation(self):
        """Sonnet pricing: $3/$15 per 1M tokens."""
        self.mock_messages.create.return_value = _make_message(
            "ok", input_tokens=1_000_000, output_tokens=1_000_000
        )
        self.client.call(system="s", user="u", model="claude-sonnet-4-6")
        summary = self.client.get_usage_summary()
        # $3 input + $15 output = $18
        self.assertAlmostEqual(summary["estimated_cost_usd"], 18.0, places=2)

    def test_opus_cost_estimation(self):
        """Opus pricing: $5/$25 per 1M tokens."""
        self.mock_messages.create.return_value = _make_message(
            "ok", input_tokens=1_000_000, output_tokens=1_000_000
        )
        self.client.call(system="s", user="u", model="claude-opus-4-6")
        summary = self.client.get_usage_summary()
        # $5 input + $25 output = $30
        self.assertAlmostEqual(summary["estimated_cost_usd"], 30.0, places=2)

    def test_haiku_cost_estimation(self):
        """Haiku pricing: $1/$5 per 1M tokens."""
        self.mock_messages.create.return_value = _make_message(
            "ok", input_tokens=1_000_000, output_tokens=1_000_000
        )
        self.client.call(system="s", user="u", model="claude-haiku-4-6")
        summary = self.client.get_usage_summary()
        # $1 input + $5 output = $6
        self.assertAlmostEqual(summary["estimated_cost_usd"], 6.0, places=2)

    def test_cost_accumulates_across_models(self):
        """Cost accumulates correctly across calls with different models."""
        self.mock_messages.create.side_effect = [
            _make_message("a", 1_000_000, 0),   # 1M input sonnet
            _make_message("b", 0, 1_000_000),   # 1M output haiku
        ]
        self.client.call(system="s", user="u", model="claude-sonnet-4-6")
        self.client.call(system="s", user="u", model="claude-haiku-4-6")
        summary = self.client.get_usage_summary()
        # $3 (sonnet input) + $5 (haiku output) = $8
        self.assertAlmostEqual(summary["estimated_cost_usd"], 8.0, places=2)

    def test_unknown_model_uses_sonnet_pricing(self):
        """Unknown model falls back to Sonnet pricing."""
        self.mock_messages.create.return_value = _make_message(
            "ok", input_tokens=1_000_000, output_tokens=0
        )
        self.client.call(system="s", user="u", model="claude-unknown-99")
        summary = self.client.get_usage_summary()
        self.assertAlmostEqual(summary["estimated_cost_usd"], 3.0, places=2)


# ---------------------------------------------------------------------------
# Test: get_usage_summary()
# ---------------------------------------------------------------------------

class TestGetUsageSummary(unittest.TestCase):

    def setUp(self):
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            self.client = LLMClient(api_key="k")
        self.client._client = MagicMock()
        self.mock_messages = MagicMock()
        self.client._client.messages = self.mock_messages

    def test_summary_keys_present(self):
        """get_usage_summary() returns all required keys."""
        summary = self.client.get_usage_summary()
        self.assertIn("total_input_tokens", summary)
        self.assertIn("total_output_tokens", summary)
        self.assertIn("total_calls", summary)
        self.assertIn("estimated_cost_usd", summary)

    def test_summary_after_calls(self):
        """Summary reflects actual usage after calls."""
        self.mock_messages.create.side_effect = [
            _make_message("a", 500, 200),
            _make_message("b", 300, 100),
        ]
        self.client.call(system="s", user="u1")
        self.client.call(system="s", user="u2")
        summary = self.client.get_usage_summary()
        self.assertEqual(summary["total_input_tokens"], 800)
        self.assertEqual(summary["total_output_tokens"], 300)
        self.assertEqual(summary["total_calls"], 2)
        self.assertGreater(summary["estimated_cost_usd"], 0)

    def test_call_history_tracked(self):
        """Individual call records are stored in history."""
        self.mock_messages.create.return_value = _make_message("resp", 100, 50)
        self.client.call(system="s", user="u")
        self.assertEqual(len(self.client.call_history), 1)
        record = self.client.call_history[0]
        self.assertEqual(record["input_tokens"], 100)
        self.assertEqual(record["output_tokens"], 50)
        self.assertIn("model", record)
        self.assertIn("cost_usd", record)


# ---------------------------------------------------------------------------
# Test: stream()
# ---------------------------------------------------------------------------

class TestLLMClientStream(unittest.TestCase):

    def setUp(self):
        with patch.dict("sys.modules", {
            "anthropic": _anthropic_mock,
            "anthropic_bedrock": _anthropic_bedrock_mock,
        }):
            self.client = LLMClient(api_key="k")
        self.client._client = MagicMock()

    def _make_stream_chunks(self, texts):
        """Build fake streaming event objects."""
        chunks = []
        for t in texts:
            chunk = MagicMock()
            chunk.type = "content_block_delta"
            chunk.delta = MagicMock()
            chunk.delta.type = "text_delta"
            chunk.delta.text = t
            chunks.append(chunk)
        return chunks

    def test_stream_yields_text_chunks(self):
        """stream() yields text chunks from the streaming response."""
        chunks = self._make_stream_chunks(["Hello", ", ", "world!"])

        # Mock the context manager returned by client.messages.stream()
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=iter(chunks))
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        self.client._client.messages.stream = MagicMock(return_value=mock_stream_cm)

        result = list(self.client.stream(system="s", user="u"))
        self.assertEqual(result, ["Hello", ", ", "world!"])

    def test_stream_uses_default_model(self):
        """stream() uses the default model when none specified."""
        chunks = self._make_stream_chunks(["ok"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=iter(chunks))
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        self.client._client.messages.stream = MagicMock(return_value=mock_stream_cm)

        list(self.client.stream(system="s", user="u"))
        _, kwargs = self.client._client.messages.stream.call_args
        self.assertEqual(kwargs["model"], self.client.default_model)

    def test_stream_uses_custom_model(self):
        """stream() accepts an explicit model parameter."""
        chunks = self._make_stream_chunks(["ok"])
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=iter(chunks))
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        self.client._client.messages.stream = MagicMock(return_value=mock_stream_cm)

        list(self.client.stream(system="s", user="u", model="claude-haiku-4-6"))
        _, kwargs = self.client._client.messages.stream.call_args
        self.assertEqual(kwargs["model"], "claude-haiku-4-6")

    def test_stream_filters_non_text_events(self):
        """stream() only yields text_delta events."""
        chunk_text = MagicMock()
        chunk_text.type = "content_block_delta"
        chunk_text.delta = MagicMock()
        chunk_text.delta.type = "text_delta"
        chunk_text.delta.text = "real text"

        chunk_other = MagicMock()
        chunk_other.type = "message_start"

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=iter([chunk_other, chunk_text]))
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        self.client._client.messages.stream = MagicMock(return_value=mock_stream_cm)

        result = list(self.client.stream(system="s", user="u"))
        self.assertEqual(result, ["real text"])


# ---------------------------------------------------------------------------
# Integration test stubs (skipped — require real credentials)
# ---------------------------------------------------------------------------

@unittest.skip("requires ANTHROPIC_API_KEY environment variable")
class TestLLMClientIntegration(unittest.TestCase):
    """Integration tests that hit the real Anthropic API."""

    def setUp(self):
        self.client = LLMClient()

    def test_real_call(self):
        """Make a real API call and verify a response is returned."""
        result = self.client.call(
            system="You are a helpful assistant. Be very brief.",
            user="Say 'hello' and nothing else.",
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_real_token_tracking(self):
        """Verify real API call tracks token usage."""
        self.client.call(
            system="You are a helpful assistant.",
            user="What is 1+1? Answer with just the number.",
        )
        summary = self.client.get_usage_summary()
        self.assertGreater(summary["total_input_tokens"], 0)
        self.assertGreater(summary["total_output_tokens"], 0)
        self.assertGreater(summary["estimated_cost_usd"], 0)

    def test_real_stream(self):
        """Verify streaming returns chunks from the real API."""
        chunks = list(self.client.stream(
            system="You are a helpful assistant.",
            user="Count from 1 to 3, one number per line.",
        ))
        self.assertGreater(len(chunks), 0)
        full_text = "".join(chunks)
        self.assertIn("1", full_text)


@unittest.skip("requires AWS credentials and Bedrock access")
class TestLLMClientBedrockIntegration(unittest.TestCase):
    """Integration tests for Bedrock auth path."""

    def setUp(self):
        self.client = LLMClient(force_bedrock=True)

    def test_bedrock_call(self):
        """Make a real API call via Bedrock."""
        result = self.client.call(
            system="You are a helpful assistant.",
            user="Say 'hello' in one word.",
        )
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


if __name__ == "__main__":
    unittest.main()

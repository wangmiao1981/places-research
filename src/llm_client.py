"""
LLM Client — Claude wrapper with retry and token tracking.

Provides a thin wrapper around the Anthropic SDK supporting:
- API key auth (ANTHROPIC_API_KEY env var) as primary method
- Bedrock auth as fallback (via AnthropicBedrock)
- Auto-detection of available auth method
- Retry with exponential backoff on rate limits (429) and server errors (5xx)
- Token usage tracking and cost estimation
- Streaming support
"""

import os
import time
from typing import Generator, List, Optional, Dict, Any

import anthropic

# Try to import AnthropicBedrock — optional dependency
try:
    from anthropic import AnthropicBedrock
    _BEDROCK_AVAILABLE = True
except ImportError:
    _BEDROCK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0

# Pricing per 1M tokens: {model_prefix: (input_price, output_price)}
# Sonnet: $3/$15, Opus: $5/$25, Haiku: $1/$5
_PRICING: Dict[str, tuple] = {
    "sonnet": (3.0, 15.0),
    "opus": (5.0, 25.0),
    "haiku": (1.0, 5.0),
}
_DEFAULT_PRICING = _PRICING["sonnet"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when an LLM API call fails after all retries are exhausted,
    or when a non-retryable error occurs."""
    pass


# ---------------------------------------------------------------------------
# Helper: determine pricing tier from model name
# ---------------------------------------------------------------------------

def _get_pricing(model: str) -> tuple:
    """Return (input_price, output_price) per 1M tokens for the given model."""
    model_lower = model.lower()
    for key, pricing in _PRICING.items():
        if key in model_lower:
            return pricing
    return _DEFAULT_PRICING


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a given number of tokens."""
    input_price, output_price = _get_pricing(model)
    return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around the Anthropic SDK for calling Claude models.

    Authentication is auto-detected:
    1. If ``api_key`` is provided explicitly, use it.
    2. If ``ANTHROPIC_API_KEY`` env var is set, use it (API key auth).
    3. Otherwise, fall back to Bedrock auth (requires AWS credentials).

    Pass ``force_bedrock=True`` to skip auto-detection and always use Bedrock.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
        force_bedrock: bool = False,
    ) -> None:
        self.default_model = default_model
        self.call_history: List[Dict[str, Any]] = []
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._estimated_cost: float = 0.0

        if force_bedrock:
            self.auth_method = "bedrock"
            self._client = AnthropicBedrock()
        elif api_key is not None:
            self.auth_method = "api_key"
            self._client = anthropic.Anthropic(api_key=api_key)
        elif os.environ.get("ANTHROPIC_API_KEY"):
            self.auth_method = "api_key"
            self._client = anthropic.Anthropic()
        else:
            self.auth_method = "bedrock"
            self._client = AnthropicBedrock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Call the Claude API and return the assistant's text response.

        Retries on rate limits (429) and server errors (5xx) with exponential
        backoff, up to MAX_RETRIES times. Raises LLMError on failure.

        Args:
            system: System prompt.
            user: User message.
            model: Model to use (defaults to self.default_model).
            max_tokens: Maximum tokens to generate.

        Returns:
            The assistant's text response.

        Raises:
            LLMError: If the call fails after all retries.
        """
        resolved_model = model or self.default_model
        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=resolved_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                self._record_usage(resolved_model, response.usage)
                return response.content[0].text

            except anthropic.RateLimitError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    self._backoff(attempt)
                    continue
                raise LLMError(
                    f"Rate limit exceeded after {MAX_RETRIES} retries: {exc}"
                ) from exc

            except anthropic.InternalServerError as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    self._backoff(attempt)
                    continue
                raise LLMError(
                    f"Server error after {MAX_RETRIES} retries: {exc}"
                ) from exc

            except anthropic.APIStatusError as exc:
                # 4xx errors (except 429 which is RateLimitError) are not retried
                raise LLMError(f"API error {exc.status_code}: {exc}") from exc

    def stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """
        Stream a Claude response, yielding text chunks as they arrive.

        Args:
            system: System prompt.
            user: User message.
            model: Model to use (defaults to self.default_model).
            max_tokens: Maximum tokens to generate.

        Yields:
            Text chunks from the streaming response.
        """
        resolved_model = model or self.default_model

        with self._client.messages.stream(
            model=resolved_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Return a summary of token usage and estimated cost across all calls.

        Returns:
            Dict with keys:
                - total_input_tokens (int)
                - total_output_tokens (int)
                - total_calls (int)
                - estimated_cost_usd (float)
        """
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_calls": len(self.call_history),
            "estimated_cost_usd": self._estimated_cost,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> None:
        """Sleep for an exponentially growing duration."""
        sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** attempt)
        time.sleep(sleep_seconds)

    def _record_usage(self, model: str, usage: Any) -> None:
        """Update cumulative token counters and cost, and append to history."""
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cost = _estimate_cost(model, input_tokens, output_tokens)

        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._estimated_cost += cost

        self.call_history.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
        })

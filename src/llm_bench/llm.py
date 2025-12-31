"""LiteLLM integration for LLM-Bench."""

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
)

from llm_bench.models import LatencyMetrics, TokenUsage

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_JITTER = 0.1  # 10% jitter

# Rate limiting configuration
DEFAULT_RATE_LIMIT = 100  # calls per period
DEFAULT_RATE_PERIOD = 60.0  # seconds


class RateLimiter:
    """Simple rate limiter using sliding window algorithm.

    Limits the number of calls within a time period to prevent API abuse.
    """

    def __init__(
        self,
        max_calls: int = DEFAULT_RATE_LIMIT,
        period: float = DEFAULT_RATE_PERIOD,
    ) -> None:
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the period.
            period: Time period in seconds.
        """
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire rate limit slot, waiting if necessary."""
        async with self._lock:
            now = time.time()

            # Remove calls outside the window
            self._calls = [t for t in self._calls if now - t < self.period]

            if len(self._calls) >= self.max_calls:
                # Wait until oldest call expires
                sleep_time = self.period - (now - self._calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                # Remove expired calls after sleeping
                now = time.time()
                self._calls = [t for t in self._calls if now - t < self.period]

            self._calls.append(time.time())


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(
    max_calls: int = DEFAULT_RATE_LIMIT,
    period: float = DEFAULT_RATE_PERIOD,
) -> RateLimiter:
    """Get or create the global rate limiter.

    Args:
        max_calls: Maximum calls per period.
        period: Time period in seconds.

    Returns:
        RateLimiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_calls, period)
    return _rate_limiter


class LLMError(Exception):
    """Error raised when LLM call fails."""

    def __init__(
        self,
        message: str,
        model: str,
        error_type: str,
        details: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.message = message
        self.model = model
        self.error_type = error_type
        self.details = details
        self.suggestion = suggestion
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with helpful suggestions."""
        msg = f"[{self.error_type}] {self.message} (model: {self.model})"
        if self.details:
            msg += f"\nDetails: {self.details}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


def is_local_model(model: str) -> bool:
    """Check if a model is a local model (Ollama, LM Studio, vLLM, etc.).

    Args:
        model: Model identifier.

    Returns:
        True if the model is a local model provider.
    """
    model_lower = model.lower()
    local_prefixes = ("ollama/", "ollama_chat/", "lm_studio/", "hosted_vllm/")
    return any(model_lower.startswith(prefix) for prefix in local_prefixes)


def _get_api_key_suggestion(model: str) -> str:
    """Get a suggestion for setting the API key based on model name."""
    model_lower = model.lower()

    if is_local_model(model):
        return "Local models don't require API keys. Ensure your local server is running."
    elif "openai" in model_lower or model_lower.startswith("gpt"):
        return "Set OPENAI_API_KEY environment variable or add it to your .env file"
    elif "anthropic" in model_lower or "claude" in model_lower:
        return "Set ANTHROPIC_API_KEY environment variable or add it to your .env file"
    elif "gemini" in model_lower or "google" in model_lower:
        return "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable"
    elif "openrouter" in model_lower:
        return "Set OPENROUTER_API_KEY environment variable"
    elif "groq" in model_lower:
        return "Set GROQ_API_KEY environment variable"
    elif "mistral" in model_lower:
        return "Set MISTRAL_API_KEY environment variable"
    else:
        return "Check that the correct API key is set for this provider"


def _get_rate_limit_suggestion(_model: str) -> str:
    """Get a suggestion for handling rate limits."""
    return (
        "Options: 1) Wait and retry, 2) Reduce --concurrency, "
        "3) Use a different model, or 4) Upgrade your API plan"
    )


def _calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: float = DEFAULT_JITTER,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current attempt number (0-indexed).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay in seconds.
        jitter: Jitter factor (0.0 to 1.0).

    Returns:
        Delay in seconds.
    """
    delay = min(base_delay * (2**attempt), max_delay)
    # Add random jitter
    jitter_amount = delay * jitter * (2 * random.random() - 1)
    return float(max(0, delay + jitter_amount))


@dataclass
class LLMResponse:
    """Response from an LLM call with metrics."""

    content: str
    latency: LatencyMetrics
    token_usage: TokenUsage
    cost_usd: float
    model: str
    raw_response: dict[str, Any] | None = None


async def call_llm(
    model: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.1,
    stream: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    rate_limit: bool = True,
    api_base: str | None = None,
) -> LLMResponse:
    """Call an LLM provider via LiteLLM and track metrics.

    Includes automatic retry with exponential backoff for rate limits
    and transient connection errors. Rate limiting prevents API abuse.

    Args:
        model: Model identifier in format provider/model-name.
        system_prompt: System prompt for the LLM.
        user_input: User input to process.
        temperature: Generation temperature.
        stream: Whether to use streaming (enables TTFT tracking).
        max_retries: Maximum number of retry attempts for retryable errors.
        rate_limit: Whether to apply rate limiting (default True).
        api_base: Custom API base URL for local models or custom endpoints.

    Returns:
        LLMResponse with content and metrics.

    Raises:
        LLMError: If the API call fails after all retries.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        # Apply rate limiting to prevent API abuse
        if rate_limit:
            await get_rate_limiter().acquire()

        start_time = time.perf_counter()
        ttft: float | None = None

        try:
            if stream:
                content, ttft, token_usage, raw_response = await _call_streaming(
                    model, messages, temperature, start_time, api_base
                )
            else:
                content, token_usage, raw_response = await _call_non_streaming(
                    model, messages, temperature, api_base
                )

            total_time = time.perf_counter() - start_time

            # Calculate cost
            cost_usd = calculate_cost(
                model, token_usage.prompt_tokens, token_usage.completion_tokens
            )

            return LLMResponse(
                content=content,
                latency=LatencyMetrics(
                    total_seconds=total_time,
                    time_to_first_token_seconds=ttft,
                ),
                token_usage=token_usage,
                cost_usd=cost_usd,
                model=model,
                raw_response=raw_response,
            )

        except AuthenticationError as e:
            # Don't retry authentication errors
            raise LLMError(
                "Authentication failed - check your API key",
                model=model,
                error_type="AuthenticationError",
                details=str(e),
                suggestion=_get_api_key_suggestion(model),
            ) from None

        except RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                delay = _calculate_backoff_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise LLMError(
                f"Rate limit exceeded after {max_retries + 1} attempts",
                model=model,
                error_type="RateLimitError",
                details=str(e),
                suggestion=_get_rate_limit_suggestion(model),
            ) from None

        except APIConnectionError as e:
            last_error = e
            if attempt < max_retries:
                delay = _calculate_backoff_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise LLMError(
                f"Failed to connect to API after {max_retries + 1} attempts",
                model=model,
                error_type="ConnectionError",
                details=str(e),
                suggestion="Check your internet connection and try again",
            ) from None

        except APIError as e:
            # Only retry on 5xx errors (server errors)
            error_code = getattr(e, "status_code", None)
            if error_code and 500 <= error_code < 600 and attempt < max_retries:
                last_error = e
                delay = _calculate_backoff_delay(attempt)
                await asyncio.sleep(delay)
                continue
            raise LLMError(
                "API error occurred",
                model=model,
                error_type="APIError",
                details=str(e),
                suggestion="Check the API status page or try again later",
            ) from None

        except Exception as e:
            raise LLMError(
                f"Unexpected error: {type(e).__name__}",
                model=model,
                error_type="UnexpectedError",
                details=str(e),
                suggestion="Please report this issue if it persists",
            ) from None

    # This should not be reached, but just in case
    raise LLMError(
        f"Failed after {max_retries + 1} attempts",
        model=model,
        error_type="RetryExhausted",
        details=str(last_error) if last_error else "Unknown error",
        suggestion="Try again later or reduce concurrency",
    )


async def _call_streaming(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    start_time: float,
    api_base: str | None = None,
) -> tuple[str, float | None, TokenUsage, dict[str, Any] | None]:
    """Make a streaming LLM call and collect response.

    Returns:
        Tuple of (content, ttft, token_usage, raw_response).
    """
    chunks: list[str] = []
    ttft: float | None = None
    prompt_tokens = 0
    completion_tokens = 0

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = await litellm.acompletion(**kwargs)

    async for chunk in response:
        if ttft is None:
            ttft = time.perf_counter() - start_time

        delta = chunk.choices[0].delta
        if delta.content:
            chunks.append(delta.content)

        # Get token counts from final chunk if available
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0

    content = "".join(chunks)

    # If we didn't get usage from streaming, estimate tokens
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens, completion_tokens = _estimate_tokens(messages, content, model)

    return (
        content,
        ttft,
        TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        None,
    )


async def _call_non_streaming(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    api_base: str | None = None,
) -> tuple[str, TokenUsage, dict[str, Any]]:
    """Make a non-streaming LLM call.

    Returns:
        Tuple of (content, token_usage, raw_response).
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if api_base:
        kwargs["api_base"] = api_base

    response = await litellm.acompletion(**kwargs)

    content = response.choices[0].message.content or ""

    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
    completion_tokens = response.usage.completion_tokens if response.usage else 0

    # If usage is missing, estimate
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens, completion_tokens = _estimate_tokens(messages, content, model)

    return (
        content,
        TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        response.model_dump(),
    )


def _estimate_tokens(
    messages: list[dict[str, str]], completion: str, model: str
) -> tuple[int, int]:
    """Estimate token counts when not provided by API.

    Uses LiteLLM's token counting utilities.
    """
    try:
        prompt_text = " ".join(msg["content"] for msg in messages)
        prompt_tokens: int = litellm.token_counter(model=model, text=prompt_text)  # type: ignore
        completion_tokens: int = litellm.token_counter(model=model, text=completion)  # type: ignore
        return prompt_tokens, completion_tokens
    except Exception:
        # Fallback to rough estimate (4 chars per token)
        prompt_text = " ".join(msg["content"] for msg in messages)
        return len(prompt_text) // 4, len(completion) // 4


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost for a request using LiteLLM cost utilities.

    Args:
        model: Model identifier.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of completion tokens.

    Returns:
        Total cost in USD.
    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(  # type: ignore
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return float(prompt_cost + completion_cost)
    except Exception:
        # Return 0 if cost calculation fails (unknown model, etc.)
        return 0.0


def get_supported_models() -> list[str]:
    """Get list of models supported by LiteLLM.

    Returns:
        List of model identifiers.
    """
    return list(litellm.model_list)


def check_missing_api_keys(models: list[str]) -> list[str]:
    """Check if API keys are set for the requested models.

    Args:
        models: List of model identifiers.

    Returns:
        List of missing environment variable names with descriptions.
    """
    import os

    missing = []

    # Filter out local models - they don't need API keys
    cloud_models = [m for m in models if not is_local_model(m)]

    # OpenAI
    if any(
        m.startswith("openai/") or m.startswith("gpt-") for m in cloud_models
    ) and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY (for OpenAI models)")

    # Anthropic
    if any(
        m.startswith("anthropic/") or "claude" in m for m in cloud_models
    ) and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY (for Anthropic models)")

    # Google Gemini
    if (
        any(m.startswith("gemini/") or "gemini" in m for m in cloud_models)
        and not os.getenv("GEMINI_API_KEY")
        and not os.getenv("GOOGLE_API_KEY")
    ):
        missing.append("GEMINI_API_KEY or GOOGLE_API_KEY (for Google Gemini models)")

    # OpenRouter
    if any(m.startswith("openrouter/") for m in cloud_models) and not os.getenv(
        "OPENROUTER_API_KEY"
    ):
        missing.append("OPENROUTER_API_KEY (for OpenRouter models)")

    return missing

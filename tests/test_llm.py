"""Tests for LLM integration module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_bench.llm import (
    LLMError,
    LLMResponse,
    calculate_cost,
    call_llm,
)
from llm_bench.models import LatencyMetrics, TokenUsage


class TestLLMError:
    """Tests for LLMError class."""

    def test_error_format(self) -> None:
        """Test error message formatting."""
        error = LLMError(
            message="Test error",
            model="openai/gpt-4",
            error_type="TestError",
        )
        assert "Test error" in str(error)
        assert "openai/gpt-4" in str(error)
        assert "TestError" in str(error)

    def test_error_with_details(self) -> None:
        """Test error message with details."""
        error = LLMError(
            message="Test error",
            model="openai/gpt-4",
            error_type="TestError",
            details="Additional info",
        )
        assert "Additional info" in str(error)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello!",
            latency=LatencyMetrics(total_seconds=1.0, time_to_first_token_seconds=0.1),
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            cost_usd=0.001,
            model="openai/gpt-4",
        )
        assert response.content == "Hello!"
        assert response.latency.total_seconds == 1.0
        assert response.token_usage.total_tokens == 15
        assert response.cost_usd == 0.001


class TestCallLLM:
    """Tests for call_llm function."""

    @pytest.fixture
    def mock_streaming_response(self) -> list[MagicMock]:
        """Create mock streaming response chunks."""
        chunks = []

        # First chunk with content
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta = MagicMock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None
        chunks.append(chunk1)

        # Second chunk with content
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta = MagicMock()
        chunk2.choices[0].delta.content = " World!"
        chunk2.usage = None
        chunks.append(chunk2)

        # Final chunk with usage
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta = MagicMock()
        chunk3.choices[0].delta.content = None
        chunk3.usage = MagicMock()
        chunk3.usage.prompt_tokens = 20
        chunk3.usage.completion_tokens = 10
        chunks.append(chunk3)

        return chunks

    @pytest.fixture
    def mock_non_streaming_response(self) -> MagicMock:
        """Create mock non-streaming response."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "Hello World!"
        response.usage = MagicMock()
        response.usage.prompt_tokens = 20
        response.usage.completion_tokens = 10
        response.model_dump.return_value = {"id": "test"}
        return response

    @pytest.mark.asyncio
    async def test_call_llm_streaming(
        self, mock_streaming_response: list[MagicMock]
    ) -> None:
        """Test streaming LLM call."""

        async def mock_stream() -> Any:
            for chunk in mock_streaming_response:
                yield chunk

        with (
            patch("llm_bench.llm.litellm.acompletion") as mock_completion,
            patch("llm_bench.llm.calculate_cost", return_value=0.001),
        ):
            mock_completion.return_value = mock_stream()

            response = await call_llm(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                stream=True,
            )

            assert response.content == "Hello World!"
            assert response.latency.total_seconds > 0
            assert response.latency.time_to_first_token_seconds is not None
            assert response.token_usage.prompt_tokens == 20
            assert response.token_usage.completion_tokens == 10
            assert response.model == "openai/gpt-4"

    @pytest.mark.asyncio
    async def test_call_llm_non_streaming(
        self, mock_non_streaming_response: MagicMock
    ) -> None:
        """Test non-streaming LLM call."""
        with (
            patch("llm_bench.llm.litellm.acompletion") as mock_completion,
            patch("llm_bench.llm.calculate_cost", return_value=0.001),
        ):
            mock_completion.return_value = mock_non_streaming_response

            response = await call_llm(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                stream=False,
            )

            assert response.content == "Hello World!"
            assert response.latency.total_seconds > 0
            assert response.latency.time_to_first_token_seconds is None
            assert response.token_usage.prompt_tokens == 20
            assert response.token_usage.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_call_llm_authentication_error(self) -> None:
        """Test handling of authentication errors."""
        from litellm.exceptions import AuthenticationError

        with patch("llm_bench.llm.litellm.acompletion") as mock_completion:
            mock_completion.side_effect = AuthenticationError(
                message="Invalid API key",
                llm_provider="openai",
                model="gpt-4",
            )

            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="openai/gpt-4",
                    system_prompt="Test",
                    user_input="Test",
                )

            assert "Authentication" in str(exc_info.value)
            assert exc_info.value.error_type == "AuthenticationError"

    @pytest.mark.asyncio
    async def test_call_llm_rate_limit_error(self) -> None:
        """Test handling of rate limit errors."""
        from litellm.exceptions import RateLimitError

        with patch("llm_bench.llm.litellm.acompletion") as mock_completion:
            mock_completion.side_effect = RateLimitError(
                message="Rate limit exceeded",
                llm_provider="openai",
                model="gpt-4",
            )

            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="openai/gpt-4",
                    system_prompt="Test",
                    user_input="Test",
                )

            assert "Rate limit" in str(exc_info.value)
            assert exc_info.value.error_type == "RateLimitError"

    @pytest.mark.asyncio
    async def test_call_llm_connection_error(self) -> None:
        """Test handling of connection errors."""
        from litellm.exceptions import APIConnectionError

        with patch("llm_bench.llm.litellm.acompletion") as mock_completion:
            mock_completion.side_effect = APIConnectionError(
                message="Connection failed",
                llm_provider="openai",
                model="gpt-4",
            )

            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="openai/gpt-4",
                    system_prompt="Test",
                    user_input="Test",
                )

            assert "connect" in str(exc_info.value).lower()
            assert exc_info.value.error_type == "ConnectionError"

    @pytest.mark.asyncio
    async def test_call_llm_api_error(self) -> None:
        """Test handling of API errors."""
        from litellm.exceptions import APIError

        with patch("llm_bench.llm.litellm.acompletion") as mock_completion:
            mock_completion.side_effect = APIError(
                message="API error",
                status_code=500,
                llm_provider="openai",
                model="gpt-4",
            )

            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="openai/gpt-4",
                    system_prompt="Test",
                    user_input="Test",
                )

            assert exc_info.value.error_type == "APIError"

    @pytest.mark.asyncio
    async def test_call_llm_unexpected_error(self) -> None:
        """Test handling of unexpected errors."""
        with patch("llm_bench.llm.litellm.acompletion") as mock_completion:
            mock_completion.side_effect = ValueError("Unexpected error")

            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="openai/gpt-4",
                    system_prompt="Test",
                    user_input="Test",
                )

            assert exc_info.value.error_type == "UnexpectedError"
            assert "ValueError" in str(exc_info.value)


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_calculate_cost_known_model(self) -> None:
        """Test cost calculation for known model."""
        with patch("llm_bench.llm.litellm.cost_per_token") as mock_cost:
            mock_cost.return_value = (0.01, 0.02)

            cost = calculate_cost(
                "openai/gpt-4", prompt_tokens=100, completion_tokens=50
            )

            assert cost == 0.03
            mock_cost.assert_called_once_with(
                model="openai/gpt-4",
                prompt_tokens=100,
                completion_tokens=50,
            )

    def test_calculate_cost_unknown_model(self) -> None:
        """Test cost calculation for unknown model returns 0."""
        with patch("llm_bench.llm.litellm.cost_per_token") as mock_cost:
            mock_cost.side_effect = Exception("Unknown model")

            cost = calculate_cost(
                "unknown/model", prompt_tokens=100, completion_tokens=50
            )

            assert cost == 0.0

    def test_calculate_cost_zero_tokens(self) -> None:
        """Test cost calculation with zero tokens."""
        with patch("llm_bench.llm.litellm.cost_per_token") as mock_cost:
            mock_cost.return_value = (0.0, 0.0)

            cost = calculate_cost("openai/gpt-4", prompt_tokens=0, completion_tokens=0)

            assert cost == 0.0

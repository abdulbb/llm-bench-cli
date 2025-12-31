"""Tests for retry logic with exponential backoff."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import APIConnectionError, RateLimitError

from llm_bench.llm import (
    LLMError,
    _calculate_backoff_delay,
    call_llm,
)


class TestCalculateBackoffDelay:
    """Tests for _calculate_backoff_delay."""

    def test_first_attempt_base_delay(self) -> None:
        """Test first attempt uses base delay."""
        delay = _calculate_backoff_delay(0, base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert delay == 1.0

    def test_exponential_growth(self) -> None:
        """Test delay grows exponentially."""
        delay_1 = _calculate_backoff_delay(
            1, base_delay=1.0, max_delay=60.0, jitter=0.0
        )
        delay_2 = _calculate_backoff_delay(
            2, base_delay=1.0, max_delay=60.0, jitter=0.0
        )
        delay_3 = _calculate_backoff_delay(
            3, base_delay=1.0, max_delay=60.0, jitter=0.0
        )

        assert delay_1 == 2.0  # 1 * 2^1
        assert delay_2 == 4.0  # 1 * 2^2
        assert delay_3 == 8.0  # 1 * 2^3

    def test_max_delay_cap(self) -> None:
        """Test delay is capped at max_delay."""
        delay = _calculate_backoff_delay(10, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 30.0

    def test_jitter_range(self) -> None:
        """Test jitter stays within expected range."""
        delays = [
            _calculate_backoff_delay(0, base_delay=10.0, max_delay=60.0, jitter=0.1)
            for _ in range(100)
        ]
        # With 10% jitter on base 10s, delays should be in range [9, 11]
        assert all(9.0 <= d <= 11.0 for d in delays)

    def test_zero_jitter(self) -> None:
        """Test zero jitter gives consistent results."""
        delays = [
            _calculate_backoff_delay(0, base_delay=1.0, max_delay=60.0, jitter=0.0)
            for _ in range(10)
        ]
        assert all(d == 1.0 for d in delays)


class TestRetryBehavior:
    """Tests for retry behavior in call_llm."""

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        """Test that successful calls don't retry."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}

        with patch("llm_bench.llm.litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            result = await call_llm(
                model="test-model",
                system_prompt="test",
                user_input="test",
                stream=False,
                max_retries=3,
            )
            assert mock.call_count == 1
            assert result.content == "test response"

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self) -> None:
        """Test that rate limit errors trigger retry."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "success after retry"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}

        with (
            patch("llm_bench.llm.litellm.acompletion", new_callable=AsyncMock) as mock,
            patch("llm_bench.llm.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
        ):
            mock.side_effect = [
                RateLimitError(
                    "Rate limit exceeded",
                    model="test-model",
                    response=MagicMock(),
                    llm_provider="openai",
                ),
                mock_response,
            ]
            result = await call_llm(
                model="test-model",
                system_prompt="test",
                user_input="test",
                stream=False,
                max_retries=3,
            )
            assert mock.call_count == 2
            assert sleep_mock.called
            assert result.content == "success after retry"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_rate_limit(self) -> None:
        """Test that exceeding max retries raises LLMError."""
        with (
            patch("llm_bench.llm.litellm.acompletion", new_callable=AsyncMock) as mock,
            patch("llm_bench.llm.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock.side_effect = RateLimitError(
                "Rate limit exceeded",
                model="test-model",
                response=MagicMock(),
                llm_provider="openai",
            )
            with pytest.raises(LLMError) as exc_info:
                await call_llm(
                    model="test-model",
                    system_prompt="test",
                    user_input="test",
                    stream=False,
                    max_retries=2,
                )
                assert mock.call_count == 3  # initial + 2 retries
                assert "after 3 attempts" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self) -> None:
        """Test that connection errors trigger retry."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "success"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}

        with (
            patch("llm_bench.llm.litellm.acompletion", new_callable=AsyncMock) as mock,
            patch("llm_bench.llm.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock.side_effect = [
                APIConnectionError(
                    "Connection failed",
                    model="test-model",
                    request=MagicMock(),
                    llm_provider="openai",
                ),
                mock_response,
            ]
            result = await call_llm(
                model="test-model",
                system_prompt="test",
                user_input="test",
                stream=False,
                max_retries=3,
            )
            assert mock.call_count == 2
            assert result.content == "success"

    @pytest.mark.asyncio
    async def test_zero_retries_no_retry(self) -> None:
        """Test that max_retries=0 means no retry attempts."""
        with patch("llm_bench.llm.litellm.acompletion", new_callable=AsyncMock) as mock:
            mock.side_effect = RateLimitError(
                "Rate limit exceeded",
                model="test-model",
                response=MagicMock(),
                llm_provider="openai",
            )
            with pytest.raises(LLMError):
                await call_llm(
                    model="test-model",
                    system_prompt="test",
                    user_input="test",
                    stream=False,
                    max_retries=0,
                )
            assert mock.call_count == 1

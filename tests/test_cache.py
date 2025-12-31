"""Tests for caching module."""

import tempfile
from pathlib import Path

import pytest

from llm_bench.cache import (
    ResponseCache,
    generate_cache_key,
)
from llm_bench.llm import LLMResponse
from llm_bench.models import LatencyMetrics, TokenUsage


class TestGenerateCacheKey:
    """Tests for cache key generation."""

    def test_deterministic_key(self) -> None:
        """Test that same inputs produce same key."""
        key1 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        key2 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        assert key1 == key2

    def test_different_model_different_key(self) -> None:
        """Test that different model produces different key."""
        key1 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        key2 = generate_cache_key(
            model="openai/gpt-3.5-turbo",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        assert key1 != key2

    def test_different_prompt_different_key(self) -> None:
        """Test that different system prompt produces different key."""
        key1 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        key2 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are a pirate.",
            user_input="Hello",
            temperature=0.1,
        )
        assert key1 != key2

    def test_different_input_different_key(self) -> None:
        """Test that different user input produces different key."""
        key1 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        key2 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Goodbye",
            temperature=0.1,
        )
        assert key1 != key2

    def test_different_temperature_different_key(self) -> None:
        """Test that different temperature produces different key."""
        key1 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        key2 = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.5,
        )
        assert key1 != key2

    def test_key_is_sha256_hex(self) -> None:
        """Test that key is a valid SHA-256 hex string."""
        key = generate_cache_key(
            model="openai/gpt-4",
            system_prompt="You are helpful.",
            user_input="Hello",
            temperature=0.1,
        )
        assert len(key) == 64  # SHA-256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in key)


class TestResponseCache:
    """Tests for ResponseCache class."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create a temporary directory for cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_response(self) -> LLMResponse:
        """Create a sample LLM response."""
        return LLMResponse(
            content='{"result": "success"}',
            latency=LatencyMetrics(total_seconds=1.0, time_to_first_token_seconds=0.1),
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20),
            cost_usd=0.001,
            model="openai/gpt-4",
        )

    def test_cache_miss_returns_none(self, temp_cache_dir: Path) -> None:
        """Test that cache miss returns None."""
        with ResponseCache(temp_cache_dir) as cache:
            result = cache.get(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
            )
            assert result is None

    def test_cache_set_and_get(
        self, temp_cache_dir: Path, sample_response: LLMResponse
    ) -> None:
        """Test storing and retrieving from cache."""
        with ResponseCache(temp_cache_dir) as cache:
            # Store in cache
            cache.set(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                response=sample_response,
            )

            # Retrieve from cache
            result = cache.get(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
            )

            assert result is not None
            assert result.content == sample_response.content
            assert result.latency.total_seconds == sample_response.latency.total_seconds
            assert (
                result.token_usage.prompt_tokens
                == sample_response.token_usage.prompt_tokens
            )
            assert result.cost_usd == sample_response.cost_usd
            assert result.model == sample_response.model

    def test_cache_persists_across_instances(
        self, temp_cache_dir: Path, sample_response: LLMResponse
    ) -> None:
        """Test that cache persists across cache instances."""
        # Store in first cache instance
        with ResponseCache(temp_cache_dir) as cache1:
            cache1.set(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                response=sample_response,
            )

        # Retrieve from second cache instance
        with ResponseCache(temp_cache_dir) as cache2:
            result = cache2.get(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
            )

            assert result is not None
            assert result.content == sample_response.content

    def test_different_params_different_entries(
        self, temp_cache_dir: Path, sample_response: LLMResponse
    ) -> None:
        """Test that different parameters create different cache entries."""
        with ResponseCache(temp_cache_dir) as cache:
            # Store for one set of params
            cache.set(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                response=sample_response,
            )

            # Different input should not hit cache
            result = cache.get(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Goodbye",
                temperature=0.1,
            )

            assert result is None

    def test_cache_clear(
        self, temp_cache_dir: Path, sample_response: LLMResponse
    ) -> None:
        """Test clearing the cache."""
        with ResponseCache(temp_cache_dir) as cache:
            # Store in cache
            cache.set(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
                response=sample_response,
            )

            # Clear cache
            cache.clear()

            # Should be cache miss now
            result = cache.get(
                model="openai/gpt-4",
                system_prompt="You are helpful.",
                user_input="Hello",
                temperature=0.1,
            )

            assert result is None

    def test_cache_preserves_latency_metrics(self, temp_cache_dir: Path) -> None:
        """Test that latency metrics including TTFT are preserved."""
        response = LLMResponse(
            content='{"test": true}',
            latency=LatencyMetrics(
                total_seconds=2.5,
                time_to_first_token_seconds=0.25,
            ),
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
            cost_usd=0.005,
            model="openai/gpt-4",
        )

        with ResponseCache(temp_cache_dir) as cache:
            cache.set(
                model="openai/gpt-4",
                system_prompt="Test",
                user_input="Test",
                temperature=0.0,
                response=response,
            )

            result = cache.get(
                model="openai/gpt-4",
                system_prompt="Test",
                user_input="Test",
                temperature=0.0,
            )

            assert result is not None
            assert result.latency.total_seconds == 2.5
            assert result.latency.time_to_first_token_seconds == 0.25


class TestCacheIntegration:
    """Integration tests for cache with runner."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create a temporary directory for cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_response(self) -> LLMResponse:
        """Create a sample LLM response."""
        return LLMResponse(
            content='{"result": "success"}',
            latency=LatencyMetrics(total_seconds=1.0),
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20),
            cost_usd=0.001,
            model="test-model",
        )

"""Caching layer for LLM-Bench using DiskCache."""

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from diskcache import Cache

from llm_bench.llm import LLMResponse
from llm_bench.models import LatencyMetrics, TokenUsage

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "llm-bench"

# Secure permissions: user read/write/execute only (700)
SECURE_DIR_MODE = stat.S_IRWXU


def generate_cache_key(
    model: str,
    system_prompt: str,
    user_input: str,
    temperature: float,
) -> str:
    """Generate a deterministic cache key from request parameters.

    The cache key is a SHA-256 hash of the normalized request parameters,
    ensuring consistent hashing regardless of input formatting.

    Args:
        model: Model identifier (e.g., "openai/gpt-4").
        system_prompt: System prompt for the LLM.
        user_input: User input/query.
        temperature: Generation temperature.

    Returns:
        Hex string of the SHA-256 hash.
    """
    # Create a canonical representation for hashing
    key_data = {
        "model": model,
        "system_prompt": system_prompt,
        "user_input": user_input,
        "temperature": temperature,
    }

    # Use JSON with sorted keys for deterministic serialization
    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True)

    # Generate SHA-256 hash
    return hashlib.sha256(key_string.encode("utf-8")).hexdigest()


class ResponseCache:
    """Cache for LLM responses using DiskCache.

    Provides persistent caching of LLM responses to avoid redundant API calls.
    Cache entries are keyed by a hash of the request parameters.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize the response cache with secure permissions.

        Args:
            cache_dir: Directory to store cache files. Defaults to ~/.cache/llm-bench.
        """
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR

        # Create directory with secure permissions (700 = rwx------)
        self._cache_dir.mkdir(parents=True, exist_ok=True, mode=SECURE_DIR_MODE)

        # Ensure existing directory has correct permissions
        # This handles cases where the directory already existed with wrong permissions
        try:
            current_mode = self._cache_dir.stat().st_mode & 0o777
            if current_mode != SECURE_DIR_MODE:
                os.chmod(self._cache_dir, SECURE_DIR_MODE)
        except OSError:
            pass  # Best effort - may fail on some systems

        self._cache: Cache = Cache(str(self._cache_dir))

    def get(
        self,
        model: str,
        system_prompt: str,
        user_input: str,
        temperature: float,
    ) -> LLMResponse | None:
        """Retrieve a cached response if available.

        Args:
            model: Model identifier.
            system_prompt: System prompt for the LLM.
            user_input: User input/query.
            temperature: Generation temperature.

        Returns:
            Cached LLMResponse if found, None otherwise.
        """
        key = generate_cache_key(model, system_prompt, user_input, temperature)
        cached_data = self._cache.get(key)

        if cached_data is None:
            return None

        return self._deserialize_response(cached_data)

    def set(
        self,
        model: str,
        system_prompt: str,
        user_input: str,
        temperature: float,
        response: LLMResponse,
    ) -> None:
        """Store a response in the cache.

        Args:
            model: Model identifier.
            system_prompt: System prompt for the LLM.
            user_input: User input/query.
            temperature: Generation temperature.
            response: LLM response to cache.
        """
        key = generate_cache_key(model, system_prompt, user_input, temperature)
        serialized = self._serialize_response(response)
        self._cache.set(key, serialized)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def close(self) -> None:
        """Close the cache connection."""
        self._cache.close()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache path, entry count, and size in bytes.
        """
        # Use diskcache's built-in volume() method for efficient size calculation
        return {
            "path": str(self._cache_dir),
            "entry_count": len(self._cache),
            "size_bytes": self._cache.volume(),
        }

    def __enter__(self) -> "ResponseCache":
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()

    @staticmethod
    def _serialize_response(response: LLMResponse) -> dict[str, Any]:
        """Serialize an LLMResponse for cache storage.

        Args:
            response: LLM response to serialize.

        Returns:
            Dictionary representation suitable for caching.
        """
        return {
            "content": response.content,
            "latency": response.latency.model_dump(),
            "token_usage": response.token_usage.model_dump(),
            "cost_usd": response.cost_usd,
            "model": response.model,
        }

    @staticmethod
    def _deserialize_response(data: dict[str, Any]) -> LLMResponse:
        """Deserialize a cached response back to LLMResponse.

        Args:
            data: Cached dictionary representation.

        Returns:
            Reconstructed LLMResponse.
        """
        return LLMResponse(
            content=data["content"],
            latency=LatencyMetrics(**data["latency"]),
            token_usage=TokenUsage(**data["token_usage"]),
            cost_usd=data["cost_usd"],
            model=data["model"],
        )


# Global cache instance (lazy initialization)
_cache_instance: ResponseCache | None = None


def get_cache(cache_dir: Path | None = None) -> ResponseCache:
    """Get or create the global cache instance.

    Args:
        cache_dir: Optional custom cache directory.

    Returns:
        ResponseCache instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache(cache_dir)
    return _cache_instance


def close_cache() -> None:
    """Close the global cache instance."""
    global _cache_instance
    if _cache_instance is not None:
        _cache_instance.close()
        _cache_instance = None

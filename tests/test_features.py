"""Tests for new features: Cost Limits."""

from unittest.mock import patch

import pytest

from llm_bench.llm import LLMResponse
from llm_bench.models import (
    BenchConfig,
    LatencyMetrics,
    RunConfig,
    TestCase,
    TokenUsage,
)
from llm_bench.runner import run_benchmark


class TestCostLimits:
    """Tests for max_cost feature."""

    @pytest.mark.asyncio
    async def test_stops_after_cost_limit(self) -> None:
        """Test that execution stops when cost limit is exceeded."""
        # Config with 2 models, 2 tests each = 4 tasks
        config = BenchConfig(
            name="Cost Test",
            system_prompt="sys",
            models=["model1", "model2"],
            test_cases=[
                TestCase(input="t1", expected=None),
                TestCase(input="t2", expected=None),
            ],
            config=RunConfig(max_cost=0.05, concurrency=1, temperature=0.1),
        )

        # Mock LLM response with cost
        # First call costs 0.04 (total 0.04 < 0.05) -> OK
        # Second call costs 0.04 (total 0.08 > 0.05) -> OK
        # Third call -> Should be skipped (checked before execution)

        mock_response = LLMResponse(
            content="{}",
            latency=LatencyMetrics(total_seconds=0.1),
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
            cost_usd=0.04,
            model="model1",
        )

        with patch("llm_bench.runner.call_llm", return_value=mock_response) as mock_llm:
            result = await run_benchmark(config, use_cache=False)

            # We expect some tests to run, but not all
            # Since concurrency=1, they run sequentially.
            # 1. Run task 1: Cost += 0.04 (Total 0.04)
            # 2. Run task 2: Cost += 0.04 (Total 0.08)
            # 3. Run task 3: Check (0.08 > 0.05) -> Skip
            # 4. Run task 4: Check (0.08 > 0.05) -> Skip

            # So mock_llm should be called 2 times.
            assert mock_llm.call_count == 2

            # The result object should still contain results for the ones that ran
            total_tests_ran = sum(len(m.test_results) for m in result.model_results)
            assert total_tests_ran == 2

    @pytest.mark.asyncio
    async def test_high_limit_runs_all(self) -> None:
        """Test that execution runs all if limit is high."""
        config = BenchConfig(
            name="Cost Test",
            system_prompt="sys",
            models=["model1"],
            test_cases=[TestCase(input="t1", expected=None)],
            config=RunConfig(max_cost=10.0, concurrency=5, temperature=0.1),
        )

        mock_response = LLMResponse(
            content="{}",
            latency=LatencyMetrics(total_seconds=0.1),
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
            cost_usd=0.01,
            model="model1",
        )

        with patch("llm_bench.runner.call_llm", return_value=mock_response) as mock_llm:
            await run_benchmark(config, use_cache=False)
            assert mock_llm.call_count == 1

"""Tests for async execution engine."""

import asyncio
from unittest.mock import patch

import pytest

from llm_bench.llm import LLMError, LLMResponse
from llm_bench.models import (
    BenchConfig,
    LatencyMetrics,
    RunConfig,
    TestCase,
    TestResult,
    TokenUsage,
    ValidationStatus,
)
from llm_bench.runner import (
    aggregate_results,
    execute_single_test,
    generate_task_specs,
    run_benchmark,
    run_with_progress,
)

# Note: Common fixtures like sample_test_case, sample_latency, sample_token_usage,
# sample_llm_response, and minimal_bench_config are available from conftest.py


class TestGenerateTaskSpecs:
    """Tests for task specification generation."""

    def test_single_model_single_test(self) -> None:
        """Test with one model and one test case."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            models=["model-a"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[TestCase(input="test", expected={"key": "value"})],
        )

        specs = generate_task_specs(config)

        assert len(specs) == 1
        assert specs[0].model == "model-a"
        assert specs[0].test_case.input == "test"
        assert specs[0].index == 0

    def test_multiple_models_multiple_tests(self) -> None:
        """Test with multiple models and test cases."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            models=["model-a", "model-b"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[
                TestCase(input="test1", expected={}),
                TestCase(input="test2", expected={}),
            ],
        )

        specs = generate_task_specs(config)

        # Should generate 2 models × 2 tests = 4 specs
        assert len(specs) == 4

        # Check all combinations exist
        combinations = [(s.model, s.test_case.input) for s in specs]
        assert ("model-a", "test1") in combinations
        assert ("model-a", "test2") in combinations
        assert ("model-b", "test1") in combinations
        assert ("model-b", "test2") in combinations

    def test_index_increments(self) -> None:
        """Test that indices increment correctly."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            models=["model-a", "model-b"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[
                TestCase(input="test1", expected={}),
                TestCase(input="test2", expected={}),
            ],
        )

        specs = generate_task_specs(config)

        indices = [s.index for s in specs]
        assert indices == [0, 1, 2, 3]


class TestAggregateResults:
    """Tests for result aggregation."""

    # Uses sample_test_result_passed fixture from conftest.py for cleaner tests
    @pytest.fixture
    def make_test_result(
        self, sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
    ) -> TestResult:
        """Create a test result for aggregation tests."""
        return TestResult(
            test_case=TestCase(input="test", expected={}),
            passed=True,
            status=ValidationStatus.PASSED,
            actual_output={},
            raw_output="{}",
            latency=sample_latency,
            token_usage=sample_token_usage,
            cost_usd=0.001,
            error_message=None,
            used_fuzzy_match=False,
        )

    def test_aggregate_single_model(self, make_test_result: TestResult) -> None:
        """Test aggregating results for a single model."""
        results = [("model-a", make_test_result)]
        models = ["model-a"]

        aggregated = aggregate_results(results, models)

        assert len(aggregated) == 1
        assert aggregated[0].model_name == "model-a"
        assert len(aggregated[0].test_results) == 1

    def test_aggregate_multiple_models(self, make_test_result: TestResult) -> None:
        """Test aggregating results for multiple models."""
        results = [
            ("model-a", make_test_result),
            ("model-b", make_test_result),
        ]
        models = ["model-a", "model-b"]

        aggregated = aggregate_results(results, models)

        assert len(aggregated) == 2
        assert aggregated[0].model_name == "model-a"
        assert aggregated[1].model_name == "model-b"

    def test_preserves_model_order(self, make_test_result: TestResult) -> None:
        """Test that model order is preserved."""
        # Results in different order than models list
        results = [
            ("model-c", make_test_result),
            ("model-a", make_test_result),
            ("model-b", make_test_result),
        ]
        models = ["model-a", "model-b", "model-c"]

        aggregated = aggregate_results(results, models)

        assert [m.model_name for m in aggregated] == ["model-a", "model-b", "model-c"]

    def test_groups_results_by_model(self, make_test_result: TestResult) -> None:
        """Test that results are grouped by model."""
        results = [
            ("model-a", make_test_result),
            ("model-a", make_test_result),
            ("model-b", make_test_result),
        ]
        models = ["model-a", "model-b"]

        aggregated = aggregate_results(results, models)

        assert len(aggregated[0].test_results) == 2  # model-a has 2 results
        assert len(aggregated[1].test_results) == 1  # model-b has 1 result


class TestExecuteSingleTest:
    """Tests for single test execution."""

    # Uses sample_llm_response fixture from conftest.py
    @pytest.fixture
    def mock_llm_response(self, sample_token_usage: TokenUsage) -> LLMResponse:
        """Create a mock LLM response."""
        return LLMResponse(
            content='{"result": "success"}',
            latency=LatencyMetrics(total_seconds=1.0, time_to_first_token_seconds=0.1),
            token_usage=sample_token_usage,
            cost_usd=0.001,
            model="test-model",
        )

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_llm_response: LLMResponse) -> None:
        """Test successful test execution."""
        test_case = TestCase(input="test input", expected={"result": "success"})

        with patch("llm_bench.runner.call_llm") as mock_call:
            mock_call.return_value = mock_llm_response

            result = await execute_single_test(
                model="test-model",
                test_case=test_case,
                system_prompt="You are helpful",
                temperature=0.1,
            )

            assert result.passed is True
            assert result.status == ValidationStatus.PASSED
            assert result.actual_output == {"result": "success"}
            assert result.latency.total_seconds == 1.0
            assert result.cost_usd == 0.001

    @pytest.mark.asyncio
    async def test_validation_failure(self, mock_llm_response: LLMResponse) -> None:
        """Test execution with validation failure."""
        test_case = TestCase(input="test input", expected={"result": "different"})

        with patch("llm_bench.runner.call_llm") as mock_call:
            mock_call.return_value = mock_llm_response

            result = await execute_single_test(
                model="test-model",
                test_case=test_case,
                system_prompt="You are helpful",
                temperature=0.1,
            )

            assert result.passed is False
            assert result.status == ValidationStatus.FAILED_EQUALITY

    @pytest.mark.asyncio
    async def test_llm_error_handling(self) -> None:
        """Test handling of LLM errors."""
        test_case = TestCase(input="test input", expected={"result": "success"})

        with patch("llm_bench.runner.call_llm") as mock_call:
            mock_call.side_effect = LLMError(
                message="API failed",
                model="test-model",
                error_type="APIError",
            )

            result = await execute_single_test(
                model="test-model",
                test_case=test_case,
                system_prompt="You are helpful",
                temperature=0.1,
            )

            assert result.passed is False
            assert "API failed" in (result.error_message or "")
            assert result.cost_usd == 0.0


class TestRunBenchmark:
    """Tests for full benchmark execution."""

    @pytest.fixture
    def simple_config(self) -> BenchConfig:
        """Create a simple benchmark config."""
        return BenchConfig(
            name="Test Benchmark",
            system_prompt="You are helpful",
            models=["model-a", "model-b"],
            config=RunConfig(concurrency=2, temperature=0.1),
            test_cases=[
                TestCase(input="test1", expected={"key": "value1"}),
                TestCase(input="test2", expected={"key": "value2"}),
            ],
        )

    @pytest.fixture
    def mock_response(self) -> LLMResponse:
        """Create a mock LLM response."""
        return LLMResponse(
            content='{"key": "value1"}',
            latency=LatencyMetrics(total_seconds=0.5),
            token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
            cost_usd=0.001,
            model="test",
        )

    @pytest.mark.asyncio
    async def test_runs_all_combinations(
        self, simple_config: BenchConfig, mock_response: LLMResponse
    ) -> None:
        """Test that all model/test combinations are run."""
        call_count = 0

        async def mock_call(*_args: object, **_kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch("llm_bench.runner.call_llm", side_effect=mock_call):
            result = await run_benchmark(simple_config, use_cache=False)

            # 2 models × 2 tests = 4 calls
            assert call_count == 4
            assert len(result.model_results) == 2

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(
        self, simple_config: BenchConfig, mock_response: LLMResponse
    ) -> None:
        """Test that concurrency limit is respected."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_call(*_args: object, **_kwargs: object) -> LLMResponse:
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)

            await asyncio.sleep(0.01)  # Simulate some work

            async with lock:
                current_concurrent -= 1

            return mock_response

        # Set concurrency to 1 to make it easy to test
        simple_config.config = RunConfig(concurrency=1, temperature=0.1)

        with patch("llm_bench.runner.call_llm", side_effect=mock_call):
            await run_benchmark(simple_config, use_cache=False)

            # With concurrency=1, max should never exceed 1
            assert max_concurrent == 1

    @pytest.mark.asyncio
    async def test_handles_individual_failures(
        self, simple_config: BenchConfig, mock_response: LLMResponse
    ) -> None:
        """Test that individual task failures don't crash the pipeline."""
        call_count = 0

        async def mock_call(*_args: object, **_kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise LLMError(
                    message="Simulated failure",
                    model="test",
                    error_type="TestError",
                )
            return mock_response

        with patch("llm_bench.runner.call_llm", side_effect=mock_call):
            result = await run_benchmark(simple_config, use_cache=False)

            # Should still complete all 4 tasks
            assert call_count == 4
            # Results should include both successes and failures
            total_results = sum(len(mr.test_results) for mr in result.model_results)
            assert total_results == 4

    @pytest.mark.asyncio
    async def test_progress_callback(
        self, simple_config: BenchConfig, mock_response: LLMResponse
    ) -> None:
        """Test that progress callback is called."""
        with patch("llm_bench.runner.call_llm", return_value=mock_response):
            _result, progress = await run_with_progress(simple_config, use_cache=False)

            # Should have 4 progress updates (one per task)
            assert len(progress) == 4

            # Final update should show 4/4 complete
            assert progress[-1] == (4, 4)

            # Progress should be monotonically increasing
            completed_counts = [p[0] for p in progress]
            assert completed_counts == sorted(completed_counts)


class TestConcurrencyControl:
    """Tests specifically for concurrency limiting behavior."""

    @pytest.mark.asyncio
    async def test_high_concurrency(self) -> None:
        """Test with high concurrency limit."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test",
            models=["model-a"],
            config=RunConfig(concurrency=10, temperature=0.1),
            test_cases=[TestCase(input=f"test{i}", expected={}) for i in range(5)],
        )

        mock_response = LLMResponse(
            content="{}",
            latency=LatencyMetrics(total_seconds=0.1),
            token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
            cost_usd=0.0001,
            model="test",
        )

        with patch("llm_bench.runner.call_llm", return_value=mock_response):
            result = await run_benchmark(config, use_cache=False)

            assert len(result.model_results) == 1
            assert len(result.model_results[0].test_results) == 5

    @pytest.mark.asyncio
    async def test_semaphore_releases_on_error(self) -> None:
        """Test that semaphore is released even on errors."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test",
            models=["model-a"],
            config=RunConfig(concurrency=1, temperature=0.1),
            test_cases=[TestCase(input=f"test{i}", expected={}) for i in range(3)],
        )

        call_count = 0

        async def mock_call(*_args: object, **_kwargs: object) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise LLMError(
                    message="First call fails",
                    model="test",
                    error_type="TestError",
                )
            return LLMResponse(
                content="{}",
                latency=LatencyMetrics(total_seconds=0.1),
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                cost_usd=0.0001,
                model="test",
            )

        with patch("llm_bench.runner.call_llm", side_effect=mock_call):
            result = await run_benchmark(config, use_cache=False)

            # All 3 tasks should complete despite first one failing
            assert call_count == 3
            assert len(result.model_results[0].test_results) == 3

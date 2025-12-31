"""Tests for benchmark comparison functionality."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from llm_bench.compare import (
    compare_runs,
    format_diff,
    load_benchmark_run,
)
from llm_bench.models import (
    BenchConfig,
    BenchmarkRun,
    GitInfo,
    LatencyMetrics,
    ModelResult,
    RunConfig,
    TestCase,
    TestResult,
    TokenUsage,
    ValidationStatus,
)


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case."""
    return TestCase(input="Test input", expected={"result": "test"})


@pytest.fixture
def sample_test_result(sample_test_case: TestCase) -> TestResult:
    """Create a sample test result."""
    return TestResult(
        test_case=sample_test_case,
        passed=True,
        status=ValidationStatus.PASSED,
        actual_output={"result": "test"},
        raw_output='{"result": "test"}',
        latency=LatencyMetrics(total_seconds=1.5, time_to_first_token_seconds=0.1),
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        cost_usd=0.001,
    )


@pytest.fixture
def sample_model_result(sample_test_result: TestResult) -> ModelResult:
    """Create a sample model result."""
    return ModelResult(
        model_name="test-model",
        test_results=[sample_test_result],
    )


@pytest.fixture
def sample_config() -> BenchConfig:
    """Create a sample benchmark config."""
    return BenchConfig(
        name="Test Benchmark",
        system_prompt="Test prompt",
        models=["test-model"],
        config=RunConfig(concurrency=1, temperature=0.1),
        test_cases=[TestCase(input="Test input", expected={"result": "test"})],
    )


@pytest.fixture
def sample_benchmark_run(sample_config: BenchConfig, sample_model_result: ModelResult) -> BenchmarkRun:
    """Create a sample benchmark run."""
    return BenchmarkRun(
        config=sample_config,
        model_results=[sample_model_result],
        git_info=GitInfo(
            commit_hash="abc123",
            commit_short="abc",
            branch="main",
            is_dirty=False,
        ),
        run_timestamp="2024-01-01T00:00:00Z",
    )


class TestLoadBenchmarkRun:
    """Tests for load_benchmark_run."""

    def test_load_valid_json(self, sample_benchmark_run: BenchmarkRun) -> None:
        """Test loading a valid benchmark JSON file."""
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(sample_benchmark_run.model_dump_json())
            f.flush()

            loaded = load_benchmark_run(Path(f.name))
            assert loaded.config.name == sample_benchmark_run.config.name

    def test_load_missing_file(self) -> None:
        """Test loading a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_benchmark_run(Path("/nonexistent/file.json"))

    def test_load_invalid_json(self) -> None:
        """Test loading invalid JSON raises error."""
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()

            with pytest.raises(ValueError, match="Invalid JSON"):
                load_benchmark_run(Path(f.name))

    def test_load_invalid_schema(self) -> None:
        """Test loading JSON with invalid schema raises error."""
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": "schema"}')
            f.flush()

            with pytest.raises(ValueError, match="Failed to parse"):
                load_benchmark_run(Path(f.name))


class TestCompareRuns:
    """Tests for compare_runs."""

    def test_compare_identical_runs(
        self, sample_benchmark_run: BenchmarkRun
    ) -> None:
        """Test comparing identical runs shows no differences."""
        result = compare_runs(sample_benchmark_run, sample_benchmark_run)

        assert len(result.model_comparisons) == 1
        comp = result.model_comparisons[0]
        assert comp.pass_rate_diff == 0.0
        assert comp.latency_diff == 0.0
        assert comp.cost_diff == 0.0

    def test_compare_different_models(
        self, sample_config: BenchConfig, sample_test_result: TestResult
    ) -> None:
        """Test comparing runs with different models."""
        run_a = BenchmarkRun(
            config=sample_config,
            model_results=[
                ModelResult(model_name="model-a", test_results=[sample_test_result]),
            ],
        )
        run_b = BenchmarkRun(
            config=sample_config,
            model_results=[
                ModelResult(model_name="model-b", test_results=[sample_test_result]),
            ],
        )

        result = compare_runs(run_a, run_b)

        assert len(result.model_comparisons) == 0
        assert result.models_only_in_a == ["model-a"]
        assert result.models_only_in_b == ["model-b"]

    def test_compare_with_common_and_unique_models(
        self, sample_config: BenchConfig, sample_test_result: TestResult
    ) -> None:
        """Test comparing runs with some common and some unique models."""
        run_a = BenchmarkRun(
            config=sample_config,
            model_results=[
                ModelResult(model_name="common", test_results=[sample_test_result]),
                ModelResult(model_name="only-a", test_results=[sample_test_result]),
            ],
        )
        run_b = BenchmarkRun(
            config=sample_config,
            model_results=[
                ModelResult(model_name="common", test_results=[sample_test_result]),
                ModelResult(model_name="only-b", test_results=[sample_test_result]),
            ],
        )

        result = compare_runs(run_a, run_b)

        assert len(result.model_comparisons) == 1
        assert result.model_comparisons[0].model_name == "common"
        assert result.models_only_in_a == ["only-a"]
        assert result.models_only_in_b == ["only-b"]


class TestFormatDiff:
    """Tests for format_diff."""

    def test_positive_higher_is_better(self) -> None:
        """Test positive diff when higher is better shows green."""
        result = format_diff(5.0, is_percentage=True, higher_is_better=True)
        assert "[green]" in result
        assert "+5.00%" in result

    def test_negative_higher_is_better(self) -> None:
        """Test negative diff when higher is better shows red."""
        result = format_diff(-5.0, is_percentage=True, higher_is_better=True)
        assert "[red]" in result
        assert "-5.00%" in result

    def test_positive_lower_is_better(self) -> None:
        """Test positive diff when lower is better shows red."""
        result = format_diff(5.0, is_percentage=True, higher_is_better=False)
        assert "[red]" in result
        assert "+5.00%" in result

    def test_near_zero_shows_dim(self) -> None:
        """Test near-zero values show as dim."""
        result = format_diff(0.0001, is_percentage=False)
        assert "[dim]" in result
        assert "~0" in result

    def test_non_percentage_format(self) -> None:
        """Test non-percentage formatting."""
        result = format_diff(2.5, is_percentage=False, higher_is_better=True)
        assert "%" not in result
        assert "+2.50" in result

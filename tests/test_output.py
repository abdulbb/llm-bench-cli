"""Tests for Rich terminal output module."""

import pytest
from rich.panel import Panel
from rich.table import Table

from llm_bench.models import (
    BenchConfig,
    BenchmarkRun,
    LatencyMetrics,
    ModelResult,
    RunConfig,
    TestCase,
    TestResult,
    TokenUsage,
    ValidationStatus,
)
from llm_bench.output import (
    calculate_metrics,
    count_failures_by_type,
    create_diff_table,
    create_progress_bar,
    create_summary_table,
    format_cost,
    format_error_type,
    format_failure_detail,
    format_json_for_display,
    format_latency,
    format_pass_rate,
    format_throughput,
    get_failed_tests,
    sort_results_by_pass_rate,
)


class TestFormatPassRate:
    """Tests for pass rate formatting."""

    def test_high_pass_rate_green(self) -> None:
        """Test that high pass rate (>=90%) is green."""
        result = format_pass_rate(95.0)
        assert "[bold green]" in result
        assert "95.0%" in result

    def test_medium_pass_rate_yellow(self) -> None:
        """Test that medium pass rate (70-89%) is yellow."""
        result = format_pass_rate(75.0)
        assert "[yellow]" in result
        assert "75.0%" in result

    def test_low_pass_rate_red(self) -> None:
        """Test that low pass rate (<70%) is red."""
        result = format_pass_rate(50.0)
        assert "[bold red]" in result
        assert "50.0%" in result

    def test_boundary_90_is_green(self) -> None:
        """Test that exactly 90% is green."""
        result = format_pass_rate(90.0)
        assert "[bold green]" in result

    def test_boundary_70_is_yellow(self) -> None:
        """Test that exactly 70% is yellow."""
        result = format_pass_rate(70.0)
        assert "[yellow]" in result

    def test_zero_pass_rate(self) -> None:
        """Test zero pass rate."""
        result = format_pass_rate(0.0)
        assert "[bold red]" in result
        assert "0.0%" in result

    def test_100_pass_rate(self) -> None:
        """Test 100% pass rate."""
        result = format_pass_rate(100.0)
        assert "[bold green]" in result
        assert "100.0%" in result


class TestFormatLatency:
    """Tests for latency formatting."""

    def test_milliseconds(self) -> None:
        """Test latency under 1 second shows as milliseconds."""
        assert format_latency(0.5) == "500ms"
        assert format_latency(0.123) == "123ms"
        assert format_latency(0.001) == "1ms"

    def test_seconds(self) -> None:
        """Test latency 1 second or more shows as seconds."""
        assert format_latency(1.0) == "1.00s"
        assert format_latency(2.5) == "2.50s"
        assert format_latency(10.123) == "10.12s"

    def test_zero_latency(self) -> None:
        """Test zero latency."""
        assert format_latency(0.0) == "0ms"


class TestFormatThroughput:
    """Tests for throughput formatting."""

    def test_typical_throughput(self) -> None:
        """Test typical throughput values."""
        assert format_throughput(100.0) == "100.0 tok/s"
        assert format_throughput(50.5) == "50.5 tok/s"

    def test_zero_throughput(self) -> None:
        """Test zero throughput."""
        assert format_throughput(0.0) == "0.0 tok/s"

    def test_high_throughput(self) -> None:
        """Test high throughput values."""
        assert format_throughput(1000.0) == "1000.0 tok/s"


class TestFormatCost:
    """Tests for cost formatting."""

    def test_small_cost(self) -> None:
        """Test small costs show 4 decimal places."""
        assert format_cost(0.001) == "$0.0010"
        assert format_cost(0.0001) == "$0.0001"

    def test_larger_cost(self) -> None:
        """Test larger costs show 2 decimal places."""
        assert format_cost(0.01) == "$0.01"
        assert format_cost(1.23) == "$1.23"
        assert format_cost(10.50) == "$10.50"

    def test_zero_cost(self) -> None:
        """Test zero cost."""
        assert format_cost(0.0) == "$0.0000"


class TestSortResultsByPassRate:
    """Tests for sorting results by pass rate."""

    @pytest.fixture
    def sample_results(self) -> list[ModelResult]:
        """Create sample model results with different pass rates."""

        def make_result(name: str, passed: int, total: int) -> ModelResult:
            results = []
            for i in range(total):
                results.append(
                    TestResult(
                        test_case=TestCase(input=f"test{i}", expected={}),
                        passed=i < passed,
                        status=(
                            ValidationStatus.PASSED
                            if i < passed
                            else ValidationStatus.FAILED_EQUALITY
                        ),
                        actual_output={},
                        raw_output="{}",
                        latency=LatencyMetrics(total_seconds=0.1),
                        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                        cost_usd=0.001,
                        error_message=None if i < passed else "mismatch",
                        used_fuzzy_match=False,
                    )
                )
            return ModelResult(model_name=name, test_results=results)

        return [
            make_result("model-low", 2, 10),  # 20%
            make_result("model-high", 9, 10),  # 90%
            make_result("model-mid", 5, 10),  # 50%
        ]

    def test_sort_descending(self, sample_results: list[ModelResult]) -> None:
        """Test that results are sorted by pass rate descending."""
        sorted_results = sort_results_by_pass_rate(sample_results)

        assert sorted_results[0].model_name == "model-high"
        assert sorted_results[1].model_name == "model-mid"
        assert sorted_results[2].model_name == "model-low"

    def test_empty_list(self) -> None:
        """Test sorting empty list."""
        assert sort_results_by_pass_rate([]) == []


class TestCreateSummaryTable:
    """Tests for summary table creation."""

    @pytest.fixture
    def sample_benchmark_run(self) -> BenchmarkRun:
        """Create a sample benchmark run."""
        test_result = TestResult(
            test_case=TestCase(input="test", expected={}),
            passed=True,
            status=ValidationStatus.PASSED,
            actual_output={},
            raw_output="{}",
            latency=LatencyMetrics(total_seconds=1.0),
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
            cost_usd=0.001,
            error_message=None,
            used_fuzzy_match=False,
        )

        model_result = ModelResult(
            model_name="test-model",
            test_results=[test_result, test_result],
        )

        config = BenchConfig(
            name="Test Benchmark",
            system_prompt="You are helpful.",
            models=["test-model"],
            config=RunConfig(concurrency=1, temperature=0.1),
            test_cases=[TestCase(input="test", expected={})],
        )

        return BenchmarkRun(config=config, model_results=[model_result])

    def test_creates_table(self, sample_benchmark_run: BenchmarkRun) -> None:
        """Test that a Rich Table is created."""
        table = create_summary_table(sample_benchmark_run)
        assert isinstance(table, Table)

    def test_table_has_correct_columns(
        self, sample_benchmark_run: BenchmarkRun
    ) -> None:
        """Test that table has the expected columns."""
        table = create_summary_table(sample_benchmark_run)
        column_names = [col.header for col in table.columns]

        assert "Model" in column_names
        assert "Pass Rate" in column_names
        assert "P95 Latency" in column_names
        assert "Throughput" in column_names
        assert "Est. Cost" in column_names

    def test_table_includes_benchmark_name(
        self, sample_benchmark_run: BenchmarkRun
    ) -> None:
        """Test that table title includes benchmark name."""
        table = create_summary_table(sample_benchmark_run)
        assert table.title is not None
        assert "Test Benchmark" in table.title


class TestCalculateMetrics:
    """Tests for metrics calculation."""

    @pytest.fixture
    def sample_model_result(self) -> ModelResult:
        """Create a sample model result."""
        results = [
            TestResult(
                test_case=TestCase(input=f"test{i}", expected={}),
                passed=True,
                status=ValidationStatus.PASSED,
                actual_output={},
                raw_output="{}",
                latency=LatencyMetrics(total_seconds=0.5 + i * 0.1),
                token_usage=TokenUsage(prompt_tokens=50, completion_tokens=25),
                cost_usd=0.001,
                error_message=None,
                used_fuzzy_match=False,
            )
            for i in range(10)
        ]

        return ModelResult(model_name="test-model", test_results=results)

    def test_returns_all_metrics(self, sample_model_result: ModelResult) -> None:
        """Test that all expected metrics are returned."""
        metrics = calculate_metrics(sample_model_result)

        assert "pass_rate" in metrics
        assert "pass_rate_formatted" in metrics
        assert "p95_latency" in metrics
        assert "p95_latency_formatted" in metrics
        assert "throughput" in metrics
        assert "throughput_formatted" in metrics
        assert "cost" in metrics
        assert "cost_formatted" in metrics

    def test_pass_rate_calculation(self, sample_model_result: ModelResult) -> None:
        """Test pass rate is calculated correctly."""
        metrics = calculate_metrics(sample_model_result)
        assert metrics["pass_rate"] == 100.0

    def test_cost_calculation(self, sample_model_result: ModelResult) -> None:
        """Test cost is calculated correctly."""
        metrics = calculate_metrics(sample_model_result)
        assert metrics["cost"] == pytest.approx(0.01)


class TestCreateProgressBar:
    """Tests for progress bar creation."""

    def test_creates_progress(self) -> None:
        """Test that a Progress instance is created."""
        progress = create_progress_bar()
        assert progress is not None

    def test_progress_can_be_used_as_context_manager(self) -> None:
        """Test that progress can be used as context manager."""
        with create_progress_bar() as progress:
            task_id = progress.add_task("Test", total=10)
            progress.update(task_id, completed=5)


class TestFormatErrorType:
    """Tests for error type formatting."""

    def test_json_parse_error(self) -> None:
        """Test JSON parse error formatting."""
        result = format_error_type(ValidationStatus.FAILED_JSON_PARSE)
        assert "JSON Parse Error" in result
        assert "[red]" in result

    def test_schema_error(self) -> None:
        """Test schema error formatting."""
        result = format_error_type(ValidationStatus.FAILED_SCHEMA)
        assert "Schema Validation Error" in result
        assert "[yellow]" in result

    def test_equality_error(self) -> None:
        """Test equality error formatting."""
        result = format_error_type(ValidationStatus.FAILED_EQUALITY)
        assert "Value Mismatch" in result
        assert "[red]" in result

    def test_fuzzy_error(self) -> None:
        """Test fuzzy match error formatting."""
        result = format_error_type(ValidationStatus.FAILED_FUZZY)
        assert "Fuzzy Match Failed" in result
        assert "[yellow]" in result

    def test_passed(self) -> None:
        """Test passed status formatting."""
        result = format_error_type(ValidationStatus.PASSED)
        assert "Passed" in result
        assert "[green]" in result


class TestFormatJsonForDisplay:
    """Tests for JSON formatting."""

    def test_formats_dict(self) -> None:
        """Test formatting a dictionary."""
        result = format_json_for_display({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_none_returns_placeholder(self) -> None:
        """Test None returns placeholder text."""
        result = format_json_for_display(None)
        assert "<no output>" in result

    def test_sorts_keys(self) -> None:
        """Test that keys are sorted."""
        result = format_json_for_display({"z": 1, "a": 2})
        # "a" should appear before "z"
        assert result.index('"a"') < result.index('"z"')


class TestCreateDiffTable:
    """Tests for diff table creation."""

    def test_creates_table(self) -> None:
        """Test that a table is created."""
        table = create_diff_table({"key": "expected"}, {"key": "actual"})
        assert isinstance(table, Table)

    def test_handles_none_actual(self) -> None:
        """Test handling None as actual value."""
        table = create_diff_table({"key": "expected"}, None)
        assert isinstance(table, Table)

    def test_has_expected_and_actual_columns(self) -> None:
        """Test that table has both columns."""
        table = create_diff_table({"key": "value"}, {"key": "value"})
        column_names = [col.header for col in table.columns]
        assert "Expected" in column_names
        assert "Actual" in column_names


class TestFormatFailureDetail:
    """Tests for failure detail formatting."""

    @pytest.fixture
    def failed_test_result(self) -> TestResult:
        """Create a failed test result."""
        return TestResult(
            test_case=TestCase(input="test input", expected={"key": "expected"}),
            passed=False,
            status=ValidationStatus.FAILED_EQUALITY,
            actual_output={"key": "actual"},
            raw_output='{"key": "actual"}',
            latency=LatencyMetrics(total_seconds=1.0),
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20),
            cost_usd=0.001,
            error_message="Values do not match",
            used_fuzzy_match=False,
        )

    def test_creates_panel(self, failed_test_result: TestResult) -> None:
        """Test that a Panel is created."""
        panel = format_failure_detail(failed_test_result, 0)
        assert isinstance(panel, Panel)

    def test_panel_has_failure_title(self, failed_test_result: TestResult) -> None:
        """Test that panel title indicates failure."""
        panel = format_failure_detail(failed_test_result, 0)
        assert panel.title is not None
        assert "Failed" in panel.title

    def test_includes_test_number(self, failed_test_result: TestResult) -> None:
        """Test that panel includes test number."""
        panel = format_failure_detail(failed_test_result, 5)
        assert panel.title is not None
        assert "#6" in panel.title  # index 5 = test #6


class TestGetFailedTests:
    """Tests for getting failed tests."""

    def test_returns_only_failed(self) -> None:
        """Test that only failed tests are returned."""
        results = [
            TestResult(
                test_case=TestCase(input="test1", expected={}),
                passed=True,
                status=ValidationStatus.PASSED,
                actual_output={},
                raw_output="{}",
                latency=LatencyMetrics(total_seconds=0.1),
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                cost_usd=0.001,
                error_message=None,
                used_fuzzy_match=False,
            ),
            TestResult(
                test_case=TestCase(input="test2", expected={"key": "value"}),
                passed=False,
                status=ValidationStatus.FAILED_EQUALITY,
                actual_output={"key": "wrong"},
                raw_output='{"key": "wrong"}',
                latency=LatencyMetrics(total_seconds=0.1),
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                cost_usd=0.001,
                error_message="mismatch",
                used_fuzzy_match=False,
            ),
        ]
        model_result = ModelResult(model_name="test-model", test_results=results)

        failed = get_failed_tests(model_result)

        assert len(failed) == 1
        assert failed[0][0] == 1  # index of failed test
        assert failed[0][1].passed is False

    def test_empty_when_all_pass(self) -> None:
        """Test that empty list when all tests pass."""
        results = [
            TestResult(
                test_case=TestCase(input="test", expected={}),
                passed=True,
                status=ValidationStatus.PASSED,
                actual_output={},
                raw_output="{}",
                latency=LatencyMetrics(total_seconds=0.1),
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                cost_usd=0.001,
                error_message=None,
                used_fuzzy_match=False,
            )
        ]
        model_result = ModelResult(model_name="test-model", test_results=results)

        failed = get_failed_tests(model_result)

        assert len(failed) == 0


class TestCountFailuresByType:
    """Tests for counting failures by type."""

    @pytest.fixture
    def benchmark_with_failures(self) -> BenchmarkRun:
        """Create a benchmark run with various failure types."""

        def make_result(passed: bool, status: ValidationStatus) -> TestResult:
            return TestResult(
                test_case=TestCase(input="test", expected={}),
                passed=passed,
                status=status,
                actual_output={} if passed else None,
                raw_output="{}",
                latency=LatencyMetrics(total_seconds=0.1),
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                cost_usd=0.001,
                error_message=None if passed else "error",
                used_fuzzy_match=False,
            )

        model_result = ModelResult(
            model_name="test-model",
            test_results=[
                make_result(True, ValidationStatus.PASSED),
                make_result(False, ValidationStatus.FAILED_JSON_PARSE),
                make_result(False, ValidationStatus.FAILED_JSON_PARSE),
                make_result(False, ValidationStatus.FAILED_EQUALITY),
            ],
        )

        config = BenchConfig(
            name="Test",
            system_prompt="Test",
            models=["test-model"],
            config=RunConfig(concurrency=1, temperature=0.1),
            test_cases=[TestCase(input="test", expected={})],
        )

        return BenchmarkRun(config=config, model_results=[model_result])

    def test_counts_by_type(self, benchmark_with_failures: BenchmarkRun) -> None:
        """Test that failures are counted by type."""
        counts = count_failures_by_type(benchmark_with_failures)

        assert counts[ValidationStatus.FAILED_JSON_PARSE] == 2
        assert counts[ValidationStatus.FAILED_EQUALITY] == 1
        assert ValidationStatus.PASSED not in counts  # Only counts failures

    def test_empty_for_all_passing(self) -> None:
        """Test that empty dict when all tests pass."""
        model_result = ModelResult(
            model_name="test-model",
            test_results=[
                TestResult(
                    test_case=TestCase(input="test", expected={}),
                    passed=True,
                    status=ValidationStatus.PASSED,
                    actual_output={},
                    raw_output="{}",
                    latency=LatencyMetrics(total_seconds=0.1),
                    token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
                    cost_usd=0.001,
                    error_message=None,
                    used_fuzzy_match=False,
                )
            ],
        )

        config = BenchConfig(
            name="Test",
            system_prompt="Test",
            models=["test-model"],
            config=RunConfig(concurrency=1, temperature=0.1),
            test_cases=[TestCase(input="test", expected={})],
        )

        results = BenchmarkRun(config=config, model_results=[model_result])
        counts = count_failures_by_type(results)

        assert len(counts) == 0

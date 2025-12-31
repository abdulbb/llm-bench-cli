"""Tests for Pydantic models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

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


class TestRunConfig:
    """Tests for RunConfig model."""

    def test_default_values(self) -> None:
        """Test that RunConfig has sensible defaults."""
        config = RunConfig()
        assert config.concurrency == 5
        assert config.temperature is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RunConfig(concurrency=10, temperature=0.5)
        assert config.concurrency == 10
        assert config.temperature == 0.5

    def test_invalid_concurrency_zero(self) -> None:
        """Test that zero concurrency is rejected."""
        with pytest.raises(ValidationError):
            RunConfig(concurrency=0, temperature=0.1)

    def test_invalid_concurrency_negative(self) -> None:
        """Test that negative concurrency is rejected."""
        with pytest.raises(ValidationError):
            RunConfig(concurrency=-1, temperature=0.1)

    def test_invalid_concurrency_too_high(self) -> None:
        """Test that concurrency over 100 is rejected."""
        with pytest.raises(ValidationError):
            RunConfig(concurrency=101, temperature=0.1)

    def test_invalid_temperature_negative(self) -> None:
        """Test that negative temperature is rejected."""
        with pytest.raises(ValidationError):
            RunConfig(concurrency=5, temperature=-0.1)

    def test_invalid_temperature_too_high(self) -> None:
        """Test that temperature over 2.0 is rejected."""
        with pytest.raises(ValidationError):
            RunConfig(concurrency=5, temperature=2.1)


class TestTestCase:
    """Tests for TestCase model."""

    def test_valid_test_case(self) -> None:
        """Test valid test case creation."""
        tc = TestCase(input="Hello", expected={"response": "Hi"})
        assert tc.input == "Hello"
        assert tc.expected == {"response": "Hi"}

    def test_empty_expected(self) -> None:
        """Test test case with empty expected dict."""
        tc = TestCase(input="Hello")
        assert tc.expected is None

    def test_empty_input_rejected(self) -> None:
        """Test that empty input is rejected."""
        with pytest.raises(ValidationError):
            TestCase(input="", expected=None)

    def test_whitespace_only_input_rejected(self) -> None:
        """Test that whitespace-only input is rejected."""
        with pytest.raises(ValidationError):
            TestCase(input="   ", expected=None)

    def test_is_freeform_true_when_no_validation(self) -> None:
        """Test is_freeform is True when no validation criteria defined."""
        tc = TestCase(input="What is 2+2?")
        assert tc.is_freeform is True

    def test_is_freeform_false_with_expected(self) -> None:
        """Test is_freeform is False when expected is set."""
        tc = TestCase(input="Hello", expected={"response": "Hi"})
        assert tc.is_freeform is False

    def test_is_freeform_false_with_regex(self) -> None:
        """Test is_freeform is False when regex_pattern is set."""
        tc = TestCase(input="Hello", regex_pattern=r"\d+")
        assert tc.is_freeform is False

    def test_is_freeform_false_with_reference(self) -> None:
        """Test is_freeform is False when reference is set."""
        tc = TestCase(input="Summarize this", reference="Expected summary")
        assert tc.is_freeform is False

    def test_is_freeform_false_with_validator(self) -> None:
        """Test is_freeform is False when validator is set."""
        tc = TestCase(input="Hello", validator="custom_check")
        assert tc.is_freeform is False


class TestLatencyMetrics:
    """Tests for LatencyMetrics model."""

    def test_valid_latency(self) -> None:
        """Test valid latency metrics."""
        latency = LatencyMetrics(total_seconds=1.5, time_to_first_token_seconds=0.3)
        assert latency.total_seconds == 1.5
        assert latency.time_to_first_token_seconds == 0.3

    def test_latency_without_ttft(self) -> None:
        """Test latency without TTFT."""
        latency = LatencyMetrics(total_seconds=1.5)
        assert latency.time_to_first_token_seconds is None

    def test_negative_latency_rejected(self) -> None:
        """Test that negative latency is rejected."""
        with pytest.raises(ValidationError):
            LatencyMetrics(total_seconds=-1.0)

    def test_ttft_greater_than_total_rejected(self) -> None:
        """Test that TTFT > total is rejected."""
        with pytest.raises(ValidationError):
            LatencyMetrics(total_seconds=1.0, time_to_first_token_seconds=2.0)


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_valid_usage(self) -> None:
        """Test valid token usage."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_negative_tokens_rejected(self) -> None:
        """Test that negative tokens are rejected."""
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=50)


class TestTestResult:
    """Tests for TestResult model."""

    @pytest.fixture
    def valid_test_case(self) -> TestCase:
        """Create a valid test case fixture."""
        return TestCase(input="Test input", expected={"key": "value"})

    @pytest.fixture
    def valid_latency(self) -> LatencyMetrics:
        """Create valid latency metrics fixture."""
        return LatencyMetrics(total_seconds=1.0)

    @pytest.fixture
    def valid_usage(self) -> TokenUsage:
        """Create valid token usage fixture."""
        return TokenUsage(prompt_tokens=100, completion_tokens=50)

    def test_passed_result(
        self,
        valid_test_case: TestCase,
        valid_latency: LatencyMetrics,
        valid_usage: TokenUsage,
    ) -> None:
        """Test a passing test result."""
        result = TestResult(
            test_case=valid_test_case,
            passed=True,
            status=ValidationStatus.PASSED,
            actual_output={"key": "value"},
            latency=valid_latency,
            token_usage=valid_usage,
            cost_usd=0.001,
        )
        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    def test_failed_result_with_error(
        self,
        valid_test_case: TestCase,
        valid_latency: LatencyMetrics,
        valid_usage: TokenUsage,
    ) -> None:
        """Test a failing test result with error message."""
        result = TestResult(
            test_case=valid_test_case,
            passed=False,
            status=ValidationStatus.FAILED_JSON_PARSE,
            latency=valid_latency,
            token_usage=valid_usage,
            cost_usd=0.001,
            error_message="Invalid JSON",
        )
        assert result.passed is False
        assert result.error_message == "Invalid JSON"

    def test_failed_json_parse_requires_error(
        self,
        valid_test_case: TestCase,
        valid_latency: LatencyMetrics,
        valid_usage: TokenUsage,
    ) -> None:
        """Test that failed JSON parse requires error message."""
        with pytest.raises(ValidationError):
            TestResult(
                test_case=valid_test_case,
                passed=False,
                status=ValidationStatus.FAILED_JSON_PARSE,
                latency=valid_latency,
                token_usage=valid_usage,
                cost_usd=0.001,
            )


class TestModelResult:
    """Tests for ModelResult model."""

    @pytest.fixture
    def sample_results(self) -> list[TestResult]:
        """Create sample test results."""
        tc = TestCase(input="Test", expected={})
        latency = LatencyMetrics(total_seconds=1.0)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        return [
            TestResult(
                test_case=tc,
                passed=True,
                status=ValidationStatus.PASSED,
                latency=latency,
                token_usage=usage,
                cost_usd=0.001,
            ),
            TestResult(
                test_case=tc,
                passed=True,
                status=ValidationStatus.PASSED,
                latency=LatencyMetrics(total_seconds=2.0),
                token_usage=usage,
                cost_usd=0.002,
            ),
            TestResult(
                test_case=tc,
                passed=False,
                status=ValidationStatus.FAILED_EQUALITY,
                latency=LatencyMetrics(total_seconds=1.5),
                token_usage=usage,
                cost_usd=0.001,
            ),
        ]

    def test_model_result_properties(self, sample_results: list[TestResult]) -> None:
        """Test ModelResult computed properties."""
        result = ModelResult(model_name="gpt-4", test_results=sample_results)

        assert result.total_tests == 3
        assert result.passed_tests == 2
        assert result.pass_rate == pytest.approx(66.67, rel=0.01)
        assert result.total_cost_usd == pytest.approx(0.004)

    def test_empty_model_result(self) -> None:
        """Test ModelResult with no tests."""
        result = ModelResult(model_name="gpt-4")
        assert result.total_tests == 0
        assert result.pass_rate == 0.0
        assert result.p95_latency_seconds == 0.0

    def test_empty_model_name_rejected(self) -> None:
        """Test that empty model name is rejected."""
        with pytest.raises(ValidationError):
            ModelResult(model_name="")


class TestBenchConfig:
    """Tests for BenchConfig model."""

    def test_valid_config(self) -> None:
        """Test valid benchmark configuration."""
        config = BenchConfig(
            name="Test Benchmark",
            system_prompt="You are a helpful assistant.",
            models=["gpt-4", "claude-3"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[TestCase(input="Hello", expected={"response": "Hi"})],
        )
        assert config.name == "Test Benchmark"
        assert len(config.models) == 2
        assert len(config.test_cases) == 1

    def test_config_with_schema_path(self) -> None:
        """Test config with schema path."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            schema_path=Path("schema.json"),
            models=["gpt-4"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[TestCase(input="Test", expected={})],
        )
        assert config.schema_path == Path("schema.json")

    def test_config_with_custom_run_config(self) -> None:
        """Test config with custom run configuration."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            models=["gpt-4"],
            config=RunConfig(concurrency=10, temperature=0.5),
            test_cases=[TestCase(input="Test", expected={})],
        )
        assert config.config.concurrency == 10
        assert config.config.temperature == 0.5

    def test_empty_models_rejected(self) -> None:
        """Test that empty models list is rejected."""
        with pytest.raises(ValidationError):
            BenchConfig(
                name="Test",
                system_prompt="Test prompt",
                models=[],
                config=RunConfig(concurrency=5, temperature=0.1),
                test_cases=[TestCase(input="Test", expected={})],
            )

    def test_empty_test_cases_rejected(self) -> None:
        """Test that empty test cases list is rejected."""
        with pytest.raises(ValidationError):
            BenchConfig(
                name="Test",
                system_prompt="Test prompt",
                models=["gpt-4"],
                config=RunConfig(concurrency=5, temperature=0.1),
                test_cases=[],
            )

    def test_empty_name_rejected(self) -> None:
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError):
            BenchConfig(
                name="",
                system_prompt="Test prompt",
                models=["gpt-4"],
                config=RunConfig(concurrency=5, temperature=0.1),
                test_cases=[TestCase(input="Test", expected={})],
            )

    def test_is_freeform_true_all_freeform_tests(self) -> None:
        """Test is_freeform is True when all test cases are freeform."""
        config = BenchConfig(
            name="Freeform Test",
            system_prompt="Test prompt",
            models=["gpt-4"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[
                TestCase(input="What is 2+2?"),
                TestCase(input="Explain AI"),
            ],
        )
        assert config.is_freeform is True

    def test_is_freeform_false_with_mixed_tests(self) -> None:
        """Test is_freeform is False when some tests have validation."""
        config = BenchConfig(
            name="Mixed Test",
            system_prompt="Test prompt",
            models=["gpt-4"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[
                TestCase(input="What is 2+2?"),  # freeform
                TestCase(input="Hello", expected={"response": "Hi"}),  # has expected
            ],
        )
        assert config.is_freeform is False

    def test_is_freeform_false_all_validated(self) -> None:
        """Test is_freeform is False when all tests have validation."""
        config = BenchConfig(
            name="Validated Test",
            system_prompt="Test prompt",
            models=["gpt-4"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[
                TestCase(input="Hello", expected={"response": "Hi"}),
                TestCase(input="Count", regex_pattern=r"\d+"),
            ],
        )
        assert config.is_freeform is False


class TestBenchmarkRun:
    """Tests for BenchmarkRun model."""

    def test_benchmark_run_total_cost(self) -> None:
        """Test BenchmarkRun total cost calculation."""
        config = BenchConfig(
            name="Test",
            system_prompt="Test",
            models=["gpt-4"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[TestCase(input="Test", expected={})],
        )

        tc = TestCase(input="Test", expected={})
        latency = LatencyMetrics(total_seconds=1.0)
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        model_result = ModelResult(
            model_name="gpt-4",
            test_results=[
                TestResult(
                    test_case=tc,
                    passed=True,
                    status=ValidationStatus.PASSED,
                    latency=latency,
                    token_usage=usage,
                    cost_usd=0.01,
                ),
            ],
        )

        run = BenchmarkRun(config=config, model_results=[model_result])
        assert run.total_cost_usd == pytest.approx(0.01)

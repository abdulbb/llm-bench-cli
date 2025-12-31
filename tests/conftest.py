"""Pytest configuration and shared fixtures for LLM-Bench tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_bench.llm import LLMResponse
from llm_bench.models import (
    BenchConfig,
    LatencyMetrics,
    ModelResult,
    RunConfig,
    TestCase,
    TestResult,
    TokenUsage,
    ValidationStatus,
)

# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a basic test case for testing."""
    return TestCase(
        input="What is the capital of France?",
        expected={"country": "France", "capital": "Paris"},
    )


@pytest.fixture
def sample_test_case_with_regex() -> TestCase:
    """Create a test case with regex validation."""
    return TestCase(
        input="Write a Python function to add two numbers",
        regex_pattern=r"def\s+\w+\s*\(",
    )


@pytest.fixture
def sample_test_case_with_validator() -> TestCase:
    """Create a test case with custom validator."""
    return TestCase(
        input="Generate valid JSON",
        validator="is_valid_json",
    )


@pytest.fixture
def sample_run_config() -> RunConfig:
    """Create a basic run configuration."""
    return RunConfig(
        concurrency=5,
        temperature=0.1,
        judge_model="gpt-3.5-turbo",
    )


@pytest.fixture
def sample_bench_config(
    sample_test_case: TestCase, sample_run_config: RunConfig
) -> BenchConfig:
    """Create a complete benchmark configuration."""
    return BenchConfig(
        name="Test Benchmark",
        system_prompt="You are a helpful assistant. Respond in JSON format.",
        models=["openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
        config=sample_run_config,
        test_cases=[sample_test_case],
    )


@pytest.fixture
def minimal_bench_config() -> BenchConfig:
    """Create a minimal benchmark configuration for quick tests."""
    return BenchConfig(
        name="Minimal Test",
        system_prompt="Test prompt",
        models=["test-model"],
        test_cases=[TestCase(input="test", expected={"key": "value"})],
    )


# =============================================================================
# LLM Response Fixtures
# =============================================================================


@pytest.fixture
def sample_latency() -> LatencyMetrics:
    """Create sample latency metrics."""
    return LatencyMetrics(
        total_seconds=1.5,
        time_to_first_token_seconds=0.2,
    )


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    """Create sample token usage."""
    return TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
    )


@pytest.fixture
def sample_llm_response(
    sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
) -> LLMResponse:
    """Create a successful LLM response."""
    return LLMResponse(
        content='{"country": "France", "capital": "Paris"}',
        latency=sample_latency,
        token_usage=sample_token_usage,
        cost_usd=0.001,
        model="test-model",
    )


@pytest.fixture
def sample_llm_response_invalid_json(
    sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
) -> LLMResponse:
    """Create an LLM response with invalid JSON."""
    return LLMResponse(
        content="This is not valid JSON",
        latency=sample_latency,
        token_usage=sample_token_usage,
        cost_usd=0.001,
        model="test-model",
    )


@pytest.fixture
def sample_llm_response_markdown(
    sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
) -> LLMResponse:
    """Create an LLM response with JSON wrapped in markdown."""
    return LLMResponse(
        content='```json\n{"country": "France", "capital": "Paris"}\n```',
        latency=sample_latency,
        token_usage=sample_token_usage,
        cost_usd=0.001,
        model="test-model",
    )


# =============================================================================
# Test Result Fixtures
# =============================================================================


@pytest.fixture
def sample_test_result_passed(
    sample_test_case: TestCase,
    sample_latency: LatencyMetrics,
    sample_token_usage: TokenUsage,
) -> TestResult:
    """Create a passing test result."""
    return TestResult(
        test_case=sample_test_case,
        passed=True,
        status=ValidationStatus.PASSED,
        actual_output={"country": "France", "capital": "Paris"},
        raw_output='{"country": "France", "capital": "Paris"}',
        latency=sample_latency,
        token_usage=sample_token_usage,
        cost_usd=0.001,
    )


@pytest.fixture
def sample_test_result_failed(
    sample_test_case: TestCase,
    sample_latency: LatencyMetrics,
    sample_token_usage: TokenUsage,
) -> TestResult:
    """Create a failing test result."""
    return TestResult(
        test_case=sample_test_case,
        passed=False,
        status=ValidationStatus.FAILED_EQUALITY,
        actual_output={"country": "France", "capital": "Lyon"},
        raw_output='{"country": "France", "capital": "Lyon"}',
        latency=sample_latency,
        token_usage=sample_token_usage,
        cost_usd=0.001,
        error_message="Expected 'Paris', got 'Lyon'",
    )


@pytest.fixture
def sample_model_result(sample_test_result_passed: TestResult) -> ModelResult:
    """Create a model result with one passed test."""
    return ModelResult(
        model_name="test-model",
        test_results=[sample_test_result_passed],
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_cache() -> MagicMock:
    """Create a mock cache that returns None (cache miss)."""
    cache = MagicMock()
    cache.get.return_value = None
    return cache


@pytest.fixture
def mock_cache_hit(sample_llm_response: LLMResponse) -> MagicMock:
    """Create a mock cache that returns a cached response."""
    cache = MagicMock()
    cache.get.return_value = sample_llm_response
    return cache


# =============================================================================
# Temporary File Fixtures
# =============================================================================


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """Create a sample YAML config file."""
    config_file = tmp_path / "bench.config.yaml"
    config_file.write_text("""
name: "Test Benchmark"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4o-mini"
test_cases:
  - input: "Hello"
    expected:
      response: "Hi"
""")
    return config_file


@pytest.fixture
def sample_schema_json(tmp_path: Path) -> Path:
    """Create a sample JSON schema file."""
    schema_file = tmp_path / "schema.json"
    schema_file.write_text("""{
    "type": "object",
    "properties": {
        "country": {"type": "string"},
        "capital": {"type": "string"}
    },
    "required": ["country", "capital"]
}""")
    return schema_file


@pytest.fixture
def sample_validators_py(tmp_path: Path) -> Path:
    """Create a sample validators Python file."""
    validators_file = tmp_path / "validators.py"
    validators_file.write_text("""
import json
import re

def is_valid_json(output: str) -> bool:
    \"\"\"Check if output is valid JSON.\"\"\"
    try:
        json.loads(output)
        return True
    except json.JSONDecodeError:
        return False

def contains_keyword(output: str) -> tuple[bool, str]:
    \"\"\"Check if output contains expected keyword.\"\"\"
    if "Paris" in output:
        return True, ""
    return False, "Output does not contain 'Paris'"

def matches_pattern(output: str) -> bool:
    \"\"\"Check if output matches a pattern.\"\"\"
    return bool(re.search(r'def\\s+\\w+', output))
""")
    return validators_file


@pytest.fixture
def sample_test_cases_csv(tmp_path: Path) -> Path:
    """Create a sample CSV test cases file."""
    csv_file = tmp_path / "test_cases.csv"
    csv_file.write_text("""input,expected
"What is 2+2?","{""result"": 4}"
"What is 3+3?","{""result"": 6}"
""")
    return csv_file


@pytest.fixture
def sample_test_cases_jsonl(tmp_path: Path) -> Path:
    """Create a sample JSONL test cases file."""
    jsonl_file = tmp_path / "test_cases.jsonl"
    jsonl_file.write_text("""{"input": "What is 2+2?", "expected": {"result": 4}}
{"input": "What is 3+3?", "expected": {"result": 6}}
""")
    return jsonl_file

"""Tests for export functionality."""

import json
from pathlib import Path

import pytest

from llm_bench.export import export_to_csv, export_to_html, export_to_json
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


@pytest.fixture
def mock_results() -> BenchmarkRun:
    """Create mock benchmark results for testing."""
    test_case = TestCase(input="What is 2+2?", expected={"answer": 4})

    test_result_pass = TestResult(
        test_case=test_case,
        passed=True,
        status=ValidationStatus.PASSED,
        actual_output={"answer": 4},
        latency=LatencyMetrics(total_seconds=1.5, time_to_first_token_seconds=0.5),
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0001,
    )

    test_result_fail = TestResult(
        test_case=test_case,
        passed=False,
        status=ValidationStatus.FAILED_EQUALITY,
        actual_output={"answer": 5},
        latency=LatencyMetrics(total_seconds=2.0, time_to_first_token_seconds=0.6),
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0001,
        error_message="Expected 4, got 5",
    )

    model_result = ModelResult(
        model_name="test-model",
        test_results=[test_result_pass, test_result_fail],
    )

    config = BenchConfig(
        name="Test Benchmark",
        system_prompt="Be helpful",
        models=["test-model"],
        config=RunConfig(concurrency=5, temperature=0.1),
        test_cases=[test_case],
    )

    return BenchmarkRun(config=config, model_results=[model_result])


def test_export_to_json(mock_results: BenchmarkRun, tmp_path: Path) -> None:
    """Test JSON export."""
    output_path = tmp_path / "results.json"
    export_to_json(mock_results, output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
        assert data["config"]["name"] == "Test Benchmark"
        assert len(data["model_results"]) == 1
        assert data["model_results"][0]["model_name"] == "test-model"


def test_export_to_csv(mock_results: BenchmarkRun, tmp_path: Path) -> None:
    """Test CSV export."""
    output_path = tmp_path / "results.csv"
    export_to_csv(mock_results, output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        content = f.read()
        assert "model_name,test_index,passed,status" in content
        assert "test-model,1,True,passed" in content
        assert "test-model,2,False,failed_equality" in content


def test_export_to_html(mock_results: BenchmarkRun, tmp_path: Path) -> None:
    """Test HTML export."""
    output_path = tmp_path / "report.html"
    export_to_html(mock_results, output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        content = f.read()
        assert "<title>LLM-Bench Report - Test Benchmark</title>" in content
        assert "test-model" in content
        assert "50.0%" in content  # Pass rate
        assert "https://cdn.jsdelivr.net/npm/chart.js" in content
        assert "new Chart" in content
        assert "Expected 4, got 5" in content  # Failure detail


@pytest.fixture
def mock_freeform_results() -> BenchmarkRun:
    """Create mock freeform benchmark results (no validation criteria)."""
    # Freeform test cases have only input, no expected/regex/validator
    test_case1 = TestCase(input="What is 2+2?")
    test_case2 = TestCase(input="Explain quantum computing.")

    test_result1 = TestResult(
        test_case=test_case1,
        passed=True,  # Always passes in freeform mode
        status=ValidationStatus.PASSED,
        actual_output=None,
        raw_output="The answer is 4.",
        latency=LatencyMetrics(total_seconds=1.5, time_to_first_token_seconds=0.5),
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        cost_usd=0.0001,
    )

    test_result2 = TestResult(
        test_case=test_case2,
        passed=True,
        status=ValidationStatus.PASSED,
        actual_output=None,
        raw_output="Quantum computing uses qubits...",
        latency=LatencyMetrics(total_seconds=2.0, time_to_first_token_seconds=0.6),
        token_usage=TokenUsage(prompt_tokens=15, completion_tokens=20),
        cost_usd=0.0002,
    )

    model_result = ModelResult(
        model_name="test-model",
        test_results=[test_result1, test_result2],
    )

    config = BenchConfig(
        name="Freeform Test",
        system_prompt="Be helpful",
        models=["test-model"],
        config=RunConfig(concurrency=5, temperature=0.1),
        test_cases=[test_case1, test_case2],
    )

    return BenchmarkRun(config=config, model_results=[model_result])


def test_export_freeform_to_html(
    mock_freeform_results: BenchmarkRun, tmp_path: Path
) -> None:
    """Test HTML export for freeform benchmark."""
    output_path = tmp_path / "freeform_report.html"
    export_to_html(mock_freeform_results, output_path)

    assert output_path.exists()
    with open(output_path, encoding="utf-8") as f:
        content = f.read()
        # Check freeform-specific elements
        assert "Freeform Mode" in content
        assert "Freeform Test" in content
        assert "test-model" in content
        # Should NOT have pass rate column (freeform summary)
        assert "Pass Rate" not in content
        # Should have prompts displayed
        assert "What is 2+2?" in content
        assert "Explain quantum computing." in content
        # Should have raw outputs
        assert "The answer is 4." in content
        assert "Quantum computing uses qubits..." in content
        # Should still have charts
        assert "https://cdn.jsdelivr.net/npm/chart.js" in content
        assert "speedPriceChart" in content

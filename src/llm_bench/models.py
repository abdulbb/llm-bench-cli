"""Core Pydantic models for LLM-Bench."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ValidationStatus(str, Enum):
    """Status of validation for a test result.

    Represents the outcome of validating an LLM response through
    the multi-stage validation pipeline.
    """

    PASSED = "passed"
    FAILED_JSON_PARSE = "failed_json_parse"
    FAILED_SCHEMA = "failed_schema"
    FAILED_EQUALITY = "failed_equality"
    FAILED_REGEX = "failed_regex"
    FAILED_CUSTOM = "failed_custom"
    FAILED_FUZZY = "failed_fuzzy"


class RunConfig(BaseModel):
    """Configuration for benchmark execution.

    Controls how benchmarks are run, including parallelism,
    generation parameters, and cost limits.
    """

    concurrency: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of parallel requests (1-100)",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature (0.0-2.0). If not set, defaults to 0.1",
    )
    judge_model: str | None = Field(
        default="gpt-3.5-turbo",
        description="Model to use for LLM-as-judge fuzzy matching",
    )
    max_cost: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum cost in USD before stopping execution",
    )

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        """Ensure concurrency is within reasonable bounds."""
        if v < 1:
            raise ValueError("Concurrency must be at least 1")
        if v > 100:
            raise ValueError("Concurrency cannot exceed 100")
        return v


class TestCase(BaseModel):
    """A single test case with input and expected output.

    Defines a test scenario including the prompt to send to the LLM
    and the validation criteria for the response.
    """

    input: str = Field(
        ...,
        min_length=1,
        description="The prompt/question to send to the LLM",
    )
    expected: dict[str, Any] | None = Field(
        default=None,
        description="Expected JSON output for equality comparison",
    )
    reference: str | None = Field(
        default=None,
        description="Reference text for ROUGE-L scoring",
    )
    regex_pattern: str | None = Field(
        default=None,
        description="Regex pattern the output must match",
    )
    validator: str | None = Field(
        default=None,
        description="Name of custom validator function to use",
    )

    @field_validator("input")
    @classmethod
    def validate_input_not_empty(cls, v: str) -> str:
        """Ensure input is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Input cannot be empty or whitespace only")
        return v

    @property
    def is_freeform(self) -> bool:
        """Check if this is a freeform test case (no validation criteria).

        A freeform test case has only an input prompt with no expected output,
        regex pattern, reference text, or custom validator. These tests are used
        for manual inspection of model outputs rather than automated validation.

        Returns:
            True if no validation criteria are defined.
        """
        return (
            self.expected is None
            and self.regex_pattern is None
            and self.reference is None
            and self.validator is None
        )


class LatencyMetrics(BaseModel):
    """Latency measurements for a single request.

    Tracks timing information including total request duration
    and time to first token for streaming responses.
    """

    total_seconds: float = Field(
        ...,
        ge=0.0,
        description="Total request duration in seconds",
    )
    time_to_first_token_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Time to first token in seconds (streaming only)",
    )

    @model_validator(mode="after")
    def validate_ttft_less_than_total(self) -> "LatencyMetrics":
        """Ensure TTFT is less than or equal to total latency."""
        if (
            self.time_to_first_token_seconds is not None
            and self.time_to_first_token_seconds > self.total_seconds
        ):
            raise ValueError("TTFT cannot exceed total latency")
        return self


class TokenUsage(BaseModel):
    """Token usage for a single request.

    Tracks the number of tokens consumed by both the prompt
    and the completion for cost calculation.
    """

    prompt_tokens: int = Field(
        ...,
        ge=0,
        description="Number of tokens in the prompt",
    )
    completion_tokens: int = Field(
        ...,
        ge=0,
        description="Number of tokens in the completion",
    )

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class TestResult(BaseModel):
    """Result of running a single test case against a model.

    Contains all information about a single test execution including
    the validation outcome, timing, token usage, and cost.
    """

    test_case: TestCase = Field(description="The test case that was run")
    passed: bool = Field(description="Whether the test passed validation")
    status: ValidationStatus = Field(description="Detailed validation status")
    actual_output: dict[str, Any] | None = Field(
        default=None,
        description="Parsed JSON output from the LLM",
    )
    raw_output: str | None = Field(
        default=None,
        description="Raw string output from the LLM",
    )
    latency: LatencyMetrics = Field(description="Timing measurements")
    token_usage: TokenUsage = Field(description="Token consumption")
    cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Cost of this request in USD",
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Additional metrics (e.g., ROUGE-L score)",
    )
    error_message: str | None = Field(
        default=None,
        description="Error details if validation failed",
    )
    used_fuzzy_match: bool = Field(
        default=False,
        description="Whether LLM judge was used for validation",
    )
    is_cached: bool = Field(
        default=False,
        description="Whether response was served from cache",
    )

    @model_validator(mode="after")
    def validate_error_on_failure(self) -> "TestResult":
        """Ensure error message is present when test fails (except fuzzy)."""
        if (
            not self.passed
            and self.status != ValidationStatus.FAILED_FUZZY
            and self.error_message is None
            and self.status
            in (ValidationStatus.FAILED_JSON_PARSE, ValidationStatus.FAILED_SCHEMA)
        ):
            raise ValueError(f"Error message required for status {self.status.value}")
        return self


class ModelResult(BaseModel):
    """Aggregated results for a single model across all test cases.

    Provides computed properties for pass rate, latency percentiles,
    cost totals, and throughput metrics.
    """

    model_name: str = Field(
        ...,
        min_length=1,
        description="Model identifier (e.g., 'openai/gpt-4o')",
    )
    test_results: list[TestResult] = Field(
        default_factory=list,
        description="Individual test results for this model",
    )

    @property
    def total_tests(self) -> int:
        """Total number of tests run."""
        return len(self.test_results)

    @property
    def passed_tests(self) -> int:
        """Number of tests that passed."""
        return sum(1 for r in self.test_results if r.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate as a percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all tests."""
        return sum(r.cost_usd for r in self.test_results)

    @property
    def p95_latency_seconds(self) -> float:
        """95th percentile latency in seconds."""
        if not self.test_results:
            return 0.0
        latencies = sorted(r.latency.total_seconds for r in self.test_results)
        idx = int(len(latencies) * 0.95)
        idx = min(idx, len(latencies) - 1)
        return latencies[idx]

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all tests."""
        return sum(r.token_usage.total_tokens for r in self.test_results)

    @property
    def total_time_seconds(self) -> float:
        """Total time across all tests."""
        return sum(r.latency.total_seconds for r in self.test_results)

    @property
    def throughput_tokens_per_second(self) -> float:
        """Average throughput in tokens per second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_tokens / self.total_time_seconds


class BenchConfig(BaseModel):
    """Main benchmark configuration.

    Defines all parameters for a benchmark run including models to test,
    test cases, validation settings, and execution configuration.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Name of the benchmark for identification",
    )
    system_prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt sent to all models",
    )
    schema_path: Path | None = Field(
        default=None,
        description="Path to JSON schema for output validation",
    )
    validators_file: Path | None = Field(
        default=None,
        description="Path to Python file with custom validators",
    )
    test_cases_file: Path | None = Field(
        default=None,
        description="Path to CSV/JSONL file with test cases",
    )
    models: list[str] = Field(
        ...,
        min_length=1,
        description="List of model identifiers to benchmark",
    )
    config: RunConfig = Field(
        ...,
        description="Execution configuration",
    )
    test_cases: list[TestCase] = Field(
        ...,
        min_length=1,
        description="Test cases to run against each model",
    )

    @field_validator("models")
    @classmethod
    def validate_models_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure models list is not empty and has valid entries."""
        if not v:
            raise ValueError("At least one model must be specified")
        for model in v:
            if not model.strip():
                raise ValueError("Model names cannot be empty")
        return v

    @field_validator("test_cases")
    @classmethod
    def validate_test_cases_not_empty(cls, v: list[TestCase]) -> list[TestCase]:
        """Ensure test cases list is not empty."""
        if not v:
            raise ValueError("At least one test case must be specified")
        return v

    @field_validator("schema_path")
    @classmethod
    def validate_schema_path(cls, v: Path | None) -> Path | None:
        """Validate schema path if provided."""
        if v is not None and not str(v).strip():
            raise ValueError("Schema path cannot be empty string")
        return v

    @field_validator("test_cases_file")
    @classmethod
    def validate_test_cases_file(cls, v: Path | None) -> Path | None:
        """Validate test cases file path if provided."""
        if v is not None and not str(v).strip():
            raise ValueError("Test cases file path cannot be empty string")
        return v

    @field_validator("validators_file")
    @classmethod
    def validate_validators_file(cls, v: Path | None) -> Path | None:
        """Validate validators file path if provided."""
        if v is not None and not str(v).strip():
            raise ValueError("Validators file path cannot be empty string")
        return v

    @property
    def is_freeform(self) -> bool:
        """Check if all test cases are freeform (no validation criteria).

        When all test cases are freeform, the benchmark is in "inspection mode"
        where outputs are displayed for manual review rather than automated validation.

        Returns:
            True if all test cases are freeform.
        """
        return all(tc.is_freeform for tc in self.test_cases)


class GitInfo(BaseModel):
    """Git repository information for tagging benchmark runs.

    Captures version control context to help track which code
    version produced specific benchmark results.
    """

    commit_hash: str | None = Field(
        default=None,
        description="Full commit SHA hash",
    )
    commit_short: str | None = Field(
        default=None,
        description="Short (7-char) commit hash",
    )
    branch: str | None = Field(
        default=None,
        description="Current branch name",
    )
    is_dirty: bool = Field(
        default=False,
        description="Whether working directory has uncommitted changes",
    )
    tag: str | None = Field(
        default=None,
        description="Git tag if current commit is tagged",
    )

    def summary(self) -> str:
        """Get a short summary string for display."""
        if not self.commit_short:
            return "not a git repo"
        parts = [self.commit_short]
        if self.branch:
            parts.insert(0, self.branch)
        if self.tag:
            parts.append(f"({self.tag})")
        if self.is_dirty:
            parts.append("*")
        return " ".join(parts)


class BenchmarkRun(BaseModel):
    """Complete results of a benchmark run.

    Contains all configuration, results, and metadata for a single
    benchmark execution. Can be serialized to JSON for storage and comparison.
    """

    config: BenchConfig = Field(description="Configuration used for this run")
    model_results: list[ModelResult] = Field(
        default_factory=list,
        description="Results for each model tested",
    )
    git_info: GitInfo | None = Field(
        default=None,
        description="Git repository state when benchmark ran",
    )
    run_timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of when benchmark started",
    )

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all models."""
        return sum(m.total_cost_usd for m in self.model_results)

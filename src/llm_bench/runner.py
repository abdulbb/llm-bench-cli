"""Async execution engine for LLM-Bench."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from llm_bench.cache import ResponseCache, get_cache
from llm_bench.config import load_json_schema
from llm_bench.git_utils import get_git_info
from llm_bench.llm import LLMError, LLMResponse, call_llm
from llm_bench.metrics import calculate_rouge_l
from llm_bench.models import (
    BenchConfig,
    BenchmarkRun,
    GitInfo,
    LatencyMetrics,
    ModelResult,
    TestCase,
    TestResult,
    TokenUsage,
    ValidationStatus,
)
from llm_bench.validation import load_validators_module, validate_response


@dataclass
class TaskSpec:
    """Specification for a single benchmark task."""

    model: str
    test_case: TestCase
    index: int


def generate_task_specs(config: BenchConfig) -> list[TaskSpec]:
    """Generate all (model, test_case) combinations from config.

    Args:
        config: Benchmark configuration.

    Returns:
        List of TaskSpec objects for each combination.
    """
    specs: list[TaskSpec] = []
    index = 0

    for model in config.models:
        for test_case in config.test_cases:
            specs.append(
                TaskSpec(
                    model=model,
                    test_case=test_case,
                    index=index,
                )
            )
            index += 1

    return specs


async def run_benchmark(
    config: BenchConfig,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    fail_fast: bool = False,
) -> BenchmarkRun:
    """Run the full benchmark pipeline.

    Args:
        config: Benchmark configuration.
        use_cache: Whether to use cached responses.
        on_progress: Optional callback for progress updates (called with completed, total).
        fail_fast: Stop execution on first test failure.

    Returns:
        BenchmarkRun with all results.
    """
    # Generate all task specifications
    task_specs = generate_task_specs(config)
    total_tasks = len(task_specs)

    # Load schema if provided
    schema: dict[str, Any] | None = None
    if config.schema_path and config.schema_path.exists():
        schema = load_json_schema(config.schema_path)

    # Load validators if provided
    validators: dict[str, Callable[..., bool]] = {}
    if config.validators_file:
        validators = load_validators_module(config.validators_file)

    # Get cache instance if caching is enabled
    cache: ResponseCache | None = get_cache() if use_cache else None

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(config.config.concurrency)

    # Track completed tasks for progress
    completed = 0
    results_lock = asyncio.Lock()
    results: list[tuple[str, TestResult]] = []

    # Cost tracking
    current_cost = 0.0
    cost_lock = asyncio.Lock()

    # Fail-fast flag
    should_stop = False
    stop_lock = asyncio.Lock()

    async def run_single_task(spec: TaskSpec) -> None:
        """Run a single benchmark task with semaphore control."""
        nonlocal completed, current_cost, should_stop

        # Check if we should stop (fail-fast mode)
        if fail_fast:
            async with stop_lock:
                if should_stop:
                    async with results_lock:
                        completed += 1
                        if on_progress:
                            on_progress(completed, total_tasks)
                    return

        # Check cost limit
        if config.config.max_cost is not None:
            async with cost_lock:
                if current_cost >= config.config.max_cost:
                    # Skip task but mark as completed for progress bar
                    async with results_lock:
                        completed += 1
                        if on_progress:
                            on_progress(completed, total_tasks)
                    return

        # Resolve validator function
        validator_func: Callable[..., bool] | None = None
        if spec.test_case.validator:
            if spec.test_case.validator in validators:
                validator_func = validators[spec.test_case.validator]
            else:
                # Validator not found - we should probably fail this test
                # But for now, let's just log or handle in execute_single_test?
                # Ideally execute_single_test should handle "validator not found" error if we passed the name.
                # But we pass the function.
                # Let's handle it by passing None and letting validation fail?
                # No, better to fail fast.
                pass

                # Actually, if we can't find the validator, we can't run the validation.
                # Let's create a dummy failing validator?
                def fail_validator(_: str) -> bool:
                    return False

                validator_func = fail_validator

        async with semaphore:
            # Get custom api_base for this model if configured
            model_api_base = config.get_model_api_base(spec.model)

            result = await execute_single_test(
                model=spec.model,
                test_case=spec.test_case,
                system_prompt=config.system_prompt,
                temperature=config.config.temperature,
                judge_model=config.config.judge_model,
                schema=schema,
                cache=cache,
                custom_validator=validator_func,
                api_base=model_api_base,
            )

            # Update cost if not cached
            if not result.is_cached and result.cost_usd > 0:
                async with cost_lock:
                    current_cost += result.cost_usd

            async with results_lock:
                results.append((spec.model, result))
                completed += 1

                # Check for fail-fast
                if fail_fast and not result.passed:
                    async with stop_lock:
                        should_stop = True

                if on_progress:
                    on_progress(completed, total_tasks)

    # Create and run all tasks concurrently
    tasks = [run_single_task(spec) for spec in task_specs]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results by model
    model_results = aggregate_results(results, config.models)

    # Capture git info and timestamp
    git_info_raw = get_git_info()
    git_info = GitInfo(
        commit_hash=git_info_raw.commit_hash,
        commit_short=git_info_raw.commit_short,
        branch=git_info_raw.branch,
        is_dirty=git_info_raw.is_dirty,
        tag=git_info_raw.tag,
    )
    run_timestamp = datetime.now(timezone.utc).isoformat()

    return BenchmarkRun(
        config=config,
        model_results=model_results,
        git_info=git_info,
        run_timestamp=run_timestamp,
    )


async def execute_single_test(
    model: str,
    test_case: TestCase,
    system_prompt: str,
    temperature: float | None,
    judge_model: str | None = None,
    schema: dict[str, Any] | None = None,
    cache: ResponseCache | None = None,
    custom_validator: Callable[[str], Any] | None = None,
    api_base: str | None = None,
) -> TestResult:
    """Execute a single test case against a model.

    Args:
        model: Model identifier.
        test_case: Test case to run.
        system_prompt: System prompt for the LLM.
        temperature: Generation temperature.
        judge_model: Optional LLM model to use as a judge.
        schema: Optional JSON schema for validation.
        cache: Optional response cache for avoiding redundant API calls.
        custom_validator: Optional custom validation function.
        api_base: Optional custom API base URL for local models.

    Returns:
        TestResult with validation results and metrics.
    """
    # Apply default temperature if not set
    effective_temperature = temperature if temperature is not None else 0.1

    try:
        response: LLMResponse | None = None
        is_cached = False

        # Check cache first if available
        if cache is not None:
            response = cache.get(
                model=model,
                system_prompt=system_prompt,
                user_input=test_case.input,
                temperature=effective_temperature,
            )
            if response is not None:
                is_cached = True

        # Cache miss - call the LLM
        if response is None:
            response = await call_llm(
                model=model,
                system_prompt=system_prompt,
                user_input=test_case.input,
                temperature=effective_temperature,
                stream=True,
                api_base=api_base,
            )

            # Store in cache if available
            if cache is not None:
                cache.set(
                    model=model,
                    system_prompt=system_prompt,
                    user_input=test_case.input,
                    temperature=effective_temperature,
                    response=response,
                )

        # Validate the response
        validation = await validate_response(
            raw_output=response.content,
            expected=test_case.expected,
            schema=schema,
            judge_model=judge_model,
            cache=cache,
            regex_pattern=test_case.regex_pattern,
            custom_validator=custom_validator,
        )

        # Calculate text metrics if reference is provided
        metrics = {}
        if test_case.reference:
            rouge_score = calculate_rouge_l(response.content, test_case.reference)
            metrics["rouge_l"] = rouge_score

        return TestResult(
            test_case=test_case,
            passed=validation.passed,
            status=validation.status,
            actual_output=validation.parsed_output,
            raw_output=response.content,
            latency=response.latency,
            token_usage=response.token_usage,
            cost_usd=response.cost_usd,
            metrics=metrics,
            error_message=validation.error_message,
            used_fuzzy_match=validation.used_fuzzy_match,
            is_cached=is_cached,
        )

    except LLMError as e:
        # Handle LLM errors gracefully
        return TestResult(
            test_case=test_case,
            passed=False,
            status=ValidationStatus.FAILED_JSON_PARSE,
            actual_output=None,
            raw_output=None,
            latency=LatencyMetrics(total_seconds=0.0),
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
            cost_usd=0.0,
            error_message=str(e),
            used_fuzzy_match=False,
        )


def aggregate_results(
    results: list[tuple[str, TestResult]],
    models: list[str],
) -> list[ModelResult]:
    """Aggregate test results by model.

    Args:
        results: List of (model_name, test_result) tuples.
        models: List of model names to ensure ordering.

    Returns:
        List of ModelResult objects, one per model.
    """
    # Group results by model
    model_results_map: dict[str, list[TestResult]] = {model: [] for model in models}

    for model_name, result in results:
        if model_name in model_results_map:
            model_results_map[model_name].append(result)

    # Create ModelResult objects preserving model order
    return [
        ModelResult(
            model_name=model,
            test_results=model_results_map[model],
        )
        for model in models
    ]


async def run_with_progress(
    config: BenchConfig,
    use_cache: bool = True,
) -> tuple[BenchmarkRun, list[tuple[int, int]]]:
    """Run benchmark and collect progress updates.

    Utility function for testing progress callbacks.

    Args:
        config: Benchmark configuration.
        use_cache: Whether to use cached responses.

    Returns:
        Tuple of (BenchmarkRun, list of progress updates).
    """
    progress_updates: list[tuple[int, int]] = []

    def on_progress(completed: int, total: int) -> None:
        progress_updates.append((completed, total))

    result = await run_benchmark(config, use_cache, on_progress)
    return result, progress_updates

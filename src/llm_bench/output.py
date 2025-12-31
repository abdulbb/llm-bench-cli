"""Rich terminal output for LLM-Bench."""

import json
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from llm_bench.models import BenchmarkRun, ModelResult, TestResult, ValidationStatus


class OutputLevel(str, Enum):
    """Output verbosity levels."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


class OutputManager:
    """Centralized output management with verbosity control.

    Manages all terminal output with support for quiet, normal, and verbose modes,
    as well as color control for CI/CD environments.
    """

    def __init__(
        self,
        verbose: bool = False,
        quiet: bool = False,
        no_color: bool = False,
    ) -> None:
        """Initialize the output manager.

        Args:
            verbose: Enable verbose/debug output.
            quiet: Suppress all output except errors.
            no_color: Disable ANSI color codes.
        """
        self.verbose = verbose
        self.quiet = quiet
        self.no_color = no_color

        # Determine output level
        if quiet:
            self.level = OutputLevel.QUIET
        elif verbose:
            self.level = OutputLevel.VERBOSE
        else:
            self.level = OutputLevel.NORMAL

        # Create console with appropriate settings
        self.console = Console(
            force_terminal=None if not no_color else False,
            no_color=no_color,
        )

    def print(self, message: str, style: str | None = None) -> None:
        """Print a message (respects quiet mode).

        Args:
            message: Message to print.
            style: Optional Rich style string.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(message, style=style)

    def debug(self, message: str) -> None:
        """Print debug message (only in verbose mode).

        Args:
            message: Debug message to print.
        """
        if self.level == OutputLevel.VERBOSE:
            self.console.print(f"[dim][DEBUG][/dim] {message}")

    def info(self, message: str) -> None:
        """Print info message (respects quiet mode).

        Args:
            message: Info message to print.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(message)

    def success(self, message: str) -> None:
        """Print success message (respects quiet mode).

        Args:
            message: Success message to print.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(f"[green]{message}[/green]")

    def warning(self, message: str) -> None:
        """Print warning message (shown in normal and verbose modes).

        Args:
            message: Warning message to print.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(f"[yellow]Warning:[/yellow] {message}")

    def error(self, message: str) -> None:
        """Print error message (always shown, even in quiet mode).

        Args:
            message: Error message to print.
        """
        self.console.print(f"[red]Error:[/red] {message}")

    def table(self, table: Table) -> None:
        """Print a Rich table (respects quiet mode).

        Args:
            table: Rich Table to print.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(table)

    def panel(self, panel: Panel) -> None:
        """Print a Rich panel (respects quiet mode).

        Args:
            panel: Rich Panel to print.
        """
        if self.level != OutputLevel.QUIET:
            self.console.print(panel)

    def newline(self) -> None:
        """Print an empty line (respects quiet mode)."""
        if self.level != OutputLevel.QUIET:
            self.console.print()


# Global output manager (can be overridden)
_output_manager: OutputManager | None = None


def get_output_manager() -> OutputManager:
    """Get or create the global output manager."""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager


def set_output_manager(manager: OutputManager) -> None:
    """Set the global output manager."""
    global _output_manager
    _output_manager = manager


def create_progress_bar(
    no_color: bool = False,
    description: str = "Running tests...",
) -> Progress:
    """Create a Rich progress bar for benchmark execution.

    Args:
        no_color: Whether to disable colors.
        description: Text description to show.

    Returns:
        Configured Progress instance.
    """
    console = Console(
        force_terminal=None if not no_color else False,
        no_color=no_color,
    )
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA:[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def create_model_progress_bar(
    no_color: bool = False,
) -> Progress:
    """Create a Rich progress bar with model-specific tracking.

    Args:
        no_color: Whether to disable colors.

    Returns:
        Configured Progress instance with model info columns.
    """
    console = Console(
        force_terminal=None if not no_color else False,
        no_color=no_color,
    )
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.fields[model]}[/bold cyan]"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[dim]ETA:[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def format_pass_rate(rate: float) -> str:
    """Format pass rate with color coding.

    Args:
        rate: Pass rate as a percentage (0-100).

    Returns:
        Formatted string with Rich markup.
    """
    if rate >= 90:
        return f"[bold green]{rate:.1f}%[/bold green]"
    elif rate >= 70:
        return f"[yellow]{rate:.1f}%[/yellow]"
    else:
        return f"[bold red]{rate:.1f}%[/bold red]"


def format_latency(seconds: float) -> str:
    """Format latency value.

    Args:
        seconds: Latency in seconds.

    Returns:
        Formatted string (e.g., "1.23s" or "456ms").
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def format_throughput(tokens_per_second: float) -> str:
    """Format throughput value.

    Args:
        tokens_per_second: Throughput in tokens/second.

    Returns:
        Formatted string (e.g., "123.4 tok/s").
    """
    return f"{tokens_per_second:.1f} tok/s"


def format_cost(cost_usd: float) -> str:
    """Format cost value.

    Args:
        cost_usd: Cost in USD.

    Returns:
        Formatted string (e.g., "$0.0012" or "$1.23").
    """
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    return f"${cost_usd:.2f}"


def create_summary_table(results: BenchmarkRun) -> Table:
    """Create a Rich table summarizing benchmark results.

    Args:
        results: Complete benchmark run results.

    Returns:
        Formatted Rich Table.
    """
    table = Table(
        title=f"[bold]Benchmark Results: {results.config.name}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )

    # Add columns
    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Pass Rate", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Est. Cost", justify="right")

    # Sort by pass rate descending
    sorted_results = sort_results_by_pass_rate(results.model_results)

    # Add rows
    for model_result in sorted_results:
        table.add_row(
            model_result.model_name,
            format_pass_rate(model_result.pass_rate),
            format_latency(model_result.p95_latency_seconds),
            format_throughput(model_result.throughput_tokens_per_second),
            format_cost(model_result.total_cost_usd),
        )

    return table


def sort_results_by_pass_rate(results: list[ModelResult]) -> list[ModelResult]:
    """Sort model results by pass rate in descending order.

    Args:
        results: List of model results.

    Returns:
        Sorted list (highest pass rate first).
    """
    return sorted(results, key=lambda r: r.pass_rate, reverse=True)


def print_summary(results: BenchmarkRun, console: Console | None = None) -> None:
    """Print the benchmark summary table to the console.

    Args:
        results: Complete benchmark run results.
        console: Optional Rich console (uses default if not provided).
    """
    if console is None:
        console = Console()

    table = create_summary_table(results)
    console.print()
    console.print(table)
    console.print()

    # Print cost breakdown
    print_cost_breakdown(results, console)


def print_freeform_results(
    results: BenchmarkRun, console: Console | None = None
) -> None:
    """Print freeform benchmark results showing raw model outputs.

    Used when test cases have no validation criteria (freeform/inspection mode).
    Displays each model's raw output for each prompt for manual review.

    Args:
        results: Complete benchmark run results.
        console: Optional Rich console (uses default if not provided).
    """
    if console is None:
        console = Console()

    console.print()
    console.print(
        Panel(
            "[bold]Freeform Mode[/bold] - Showing raw model outputs for manual inspection",
            border_style="cyan",
        )
    )
    console.print()

    # Get unique test case inputs in order
    test_inputs = [tc.input for tc in results.config.test_cases]

    for test_idx, test_input in enumerate(test_inputs):
        # Truncate long prompts for display
        display_input = test_input
        if len(display_input) > 100:
            display_input = display_input[:100] + "..."

        console.print(f"[bold cyan]Prompt {test_idx + 1}:[/bold cyan] {display_input}")
        console.print()

        # Create a table for this prompt's outputs across all models
        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Model", style="bold cyan", no_wrap=True, width=30)
        table.add_column("Output", style="white", overflow="fold")
        table.add_column("Latency", justify="right", width=10)
        table.add_column("Cost", justify="right", width=10)

        for model_result in results.model_results:
            # Find the test result for this prompt
            test_result = None
            for tr in model_result.test_results:
                if tr.test_case.input == test_input:
                    test_result = tr
                    break

            if test_result is None:
                continue

            # Get the raw output, truncate if very long for table display
            if test_result.raw_output:
                raw_output = test_result.raw_output
                if len(raw_output) > 500:
                    raw_output = raw_output[:500] + "\n[dim]... (truncated)[/dim]"
            elif test_result.error_message:
                # Show error message if there's no output
                error_msg = test_result.error_message
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                raw_output = f"[red]Error:[/red] [dim]{error_msg}[/dim]"
            else:
                raw_output = "[dim]<no output>[/dim]"

            table.add_row(
                model_result.model_name,
                raw_output,
                format_latency(test_result.latency.total_seconds),
                format_cost(test_result.cost_usd),
            )

        console.print(table)
        console.print()

    # Print summary metrics (without pass rate since there's no validation)
    _print_freeform_summary(results, console)


def _print_freeform_summary(
    results: BenchmarkRun, console: Console | None = None
) -> None:
    """Print summary metrics for freeform benchmark (no pass rate).

    Args:
        results: Complete benchmark run results.
        console: Optional Rich console (uses default if not provided).
    """
    if console is None:
        console = Console()

    table = Table(
        title=f"[bold]Summary: {results.config.name}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Model", style="bold", no_wrap=True)
    table.add_column("Tests", justify="right")
    table.add_column("P95 Latency", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Est. Cost", justify="right")

    # Sort by latency (fastest first) since there's no pass rate
    sorted_results = sorted(results.model_results, key=lambda r: r.p95_latency_seconds)

    for model_result in sorted_results:
        table.add_row(
            model_result.model_name,
            str(model_result.total_tests),
            format_latency(model_result.p95_latency_seconds),
            format_throughput(model_result.throughput_tokens_per_second),
            format_cost(model_result.total_cost_usd),
        )

    console.print(table)
    console.print()

    # Print total cost
    total_cost = results.total_cost_usd
    console.print(f"[bold]Total Cost:[/bold] {format_cost(total_cost)}")
    console.print()


def print_cost_breakdown(results: BenchmarkRun, console: Console | None = None) -> None:
    """Print a detailed cost breakdown by model.

    Args:
        results: Complete benchmark run results.
        console: Optional Rich console (uses default if not provided).
    """
    if console is None:
        console = Console()

    total_cost = results.total_cost_usd

    # Create cost breakdown table
    table = Table(
        title="Cost Breakdown",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Model", style="bold")
    table.add_column("Tests", justify="right")
    table.add_column("Prompt Tokens", justify="right")
    table.add_column("Completion Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("% of Total", justify="right")

    # Sort models by cost (highest first)
    sorted_models = sorted(
        results.model_results, key=lambda m: m.total_cost_usd, reverse=True
    )

    for model_result in sorted_models:
        # Calculate token totals
        total_prompt_tokens = sum(
            r.token_usage.prompt_tokens for r in model_result.test_results
        )
        total_completion_tokens = sum(
            r.token_usage.completion_tokens for r in model_result.test_results
        )

        # Calculate percentage
        cost_pct = (
            (model_result.total_cost_usd / total_cost * 100) if total_cost > 0 else 0
        )

        # Create a visual bar for percentage
        bar_width = 10
        filled = int(cost_pct / 100 * bar_width)
        bar = (
            "[green]"
            + "=" * filled
            + "[/green]"
            + "[dim]-[/dim]" * (bar_width - filled)
        )

        table.add_row(
            model_result.model_name,
            str(model_result.total_tests),
            f"{total_prompt_tokens:,}",
            f"{total_completion_tokens:,}",
            format_cost(model_result.total_cost_usd),
            f"{cost_pct:.1f}% {bar}",
        )

    # Add total row
    total_prompt = sum(
        sum(r.token_usage.prompt_tokens for r in m.test_results)
        for m in results.model_results
    )
    total_completion = sum(
        sum(r.token_usage.completion_tokens for r in m.test_results)
        for m in results.model_results
    )
    total_tests = sum(m.total_tests for m in results.model_results)

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_tests}[/bold]",
        f"[bold]{total_prompt:,}[/bold]",
        f"[bold]{total_completion:,}[/bold]",
        f"[bold]{format_cost(total_cost)}[/bold]",
        "[bold]100%[/bold]",
    )

    console.print(table)
    console.print()


def calculate_metrics(model_result: ModelResult) -> dict[str, float | str]:
    """Calculate display metrics for a model result.

    Args:
        model_result: Model result to calculate metrics for.

    Returns:
        Dictionary with formatted metrics.
    """
    return {
        "pass_rate": model_result.pass_rate,
        "pass_rate_formatted": format_pass_rate(model_result.pass_rate),
        "p95_latency": model_result.p95_latency_seconds,
        "p95_latency_formatted": format_latency(model_result.p95_latency_seconds),
        "throughput": model_result.throughput_tokens_per_second,
        "throughput_formatted": format_throughput(
            model_result.throughput_tokens_per_second
        ),
        "cost": model_result.total_cost_usd,
        "cost_formatted": format_cost(model_result.total_cost_usd),
    }


def format_error_type(status: ValidationStatus) -> str:
    """Format the error type for display.

    Args:
        status: Validation status indicating the error type.

    Returns:
        Formatted error type string with Rich markup.
    """
    error_labels = {
        ValidationStatus.FAILED_JSON_PARSE: "[red]JSON Parse Error[/red]",
        ValidationStatus.FAILED_SCHEMA: "[yellow]Schema Validation Error[/yellow]",
        ValidationStatus.FAILED_EQUALITY: "[red]Value Mismatch[/red]",
        ValidationStatus.FAILED_FUZZY: "[yellow]Fuzzy Match Failed[/yellow]",
        ValidationStatus.PASSED: "[green]Passed[/green]",
    }
    return error_labels.get(status, f"[red]{status.value}[/red]")


def format_json_for_display(data: dict[str, Any] | None) -> str:
    """Format JSON data for display with indentation.

    Args:
        data: Dictionary to format, or None.

    Returns:
        Formatted JSON string.
    """
    if data is None:
        return "[dim]<no output>[/dim]"
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)


def create_diff_table(
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
) -> Table:
    """Create a side-by-side diff table for expected vs actual output.

    Args:
        expected: Expected output dictionary.
        actual: Actual output dictionary (may be None).

    Returns:
        Rich Table with side-by-side comparison.
    """
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        expand=True,
    )

    table.add_column("Expected", style="green", ratio=1)
    table.add_column("Actual", style="red", ratio=1)

    expected_lines = format_json_for_display(expected).split("\n")
    actual_lines = format_json_for_display(actual).split("\n")

    # Pad to same length
    max_lines = max(len(expected_lines), len(actual_lines))
    expected_lines.extend([""] * (max_lines - len(expected_lines)))
    actual_lines.extend([""] * (max_lines - len(actual_lines)))

    for exp_line, act_line in zip(expected_lines, actual_lines, strict=True):
        # Highlight differences
        if exp_line != act_line:
            exp_text = Text(exp_line, style="green")
            act_text = Text(act_line, style="red bold")
        else:
            exp_text = Text(exp_line, style="dim")
            act_text = Text(act_line, style="dim")

        table.add_row(exp_text, act_text)

    return table


def format_failure_detail(
    test_result: TestResult,
    index: int,
) -> Panel:
    """Format a single test failure for display.

    Args:
        test_result: The failed test result.
        index: Test case index for display.

    Returns:
        Rich Panel containing the failure details.
    """
    console = Console(force_terminal=True, no_color=False)

    # Build content
    content_parts = []

    # Test input (truncated if long)
    input_text = test_result.test_case.input
    if len(input_text) > 100:
        input_text = input_text[:100] + "..."
    content_parts.append(f"[bold]Input:[/bold] {input_text}")

    # Error type
    content_parts.append(f"[bold]Error:[/bold] {format_error_type(test_result.status)}")

    # Error message if present
    if test_result.error_message:
        error_msg = test_result.error_message
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        content_parts.append(f"[bold]Details:[/bold] [dim]{error_msg}[/dim]")

    # Fuzzy match indicator
    if test_result.used_fuzzy_match:
        content_parts.append("[yellow]Fuzzy matching was attempted[/yellow]")

    # Create diff table for value mismatches
    if test_result.status in (
        ValidationStatus.FAILED_EQUALITY,
        ValidationStatus.FAILED_FUZZY,
    ):
        content_parts.append("")
        expected = test_result.test_case.expected or {}
        diff_table = create_diff_table(
            expected,
            test_result.actual_output,
        )
        # Render the diff table to string
        with console.capture() as capture:
            console.print(diff_table)
        content_parts.append(capture.get())

    # For JSON parse errors, show raw output
    elif test_result.status == ValidationStatus.FAILED_JSON_PARSE:
        if test_result.raw_output:
            raw = test_result.raw_output
            if len(raw) > 300:
                raw = raw[:300] + "..."
            content_parts.append(f"\n[bold]Raw Output:[/bold]\n[dim]{raw}[/dim]")

    content = "\n".join(content_parts)

    return Panel(
        content,
        title=f"[bold red]Test #{index + 1} Failed[/bold red]",
        border_style="red",
        expand=False,
    )


def get_failed_tests(model_result: ModelResult) -> list[tuple[int, TestResult]]:
    """Get all failed tests from a model result.

    Args:
        model_result: Model result to check.

    Returns:
        List of (index, test_result) tuples for failed tests.
    """
    return [
        (i, result)
        for i, result in enumerate(model_result.test_results)
        if not result.passed
    ]


def print_failures(
    results: BenchmarkRun,
    console: Console | None = None,
    max_failures_per_model: int = 5,
) -> None:
    """Print detailed failure information grouped by model.

    Args:
        results: Complete benchmark run results.
        console: Optional Rich console (uses default if not provided).
        max_failures_per_model: Maximum failures to show per model.
    """
    if console is None:
        console = Console()

    # Check if there are any failures
    total_failures = sum(len(get_failed_tests(mr)) for mr in results.model_results)

    if total_failures == 0:
        console.print("\n[green]All tests passed![/green]")
        return

    console.print(f"\n[bold red]Failed Tests ({total_failures} total)[/bold red]")

    for model_result in results.model_results:
        failed_tests = get_failed_tests(model_result)

        if not failed_tests:
            continue

        # Model header
        console.print(
            f"\n[bold cyan]Model: {model_result.model_name}[/bold cyan] "
            f"({len(failed_tests)} failures)"
        )

        # Show failures (limited)
        for displayed, (index, test_result) in enumerate(failed_tests):
            if displayed >= max_failures_per_model:
                remaining = len(failed_tests) - displayed
                console.print(
                    f"\n[dim]... and {remaining} more failures not shown[/dim]"
                )
                break

            panel = format_failure_detail(test_result, index)
            console.print(panel)


def count_failures_by_type(
    results: BenchmarkRun,
) -> dict[ValidationStatus, int]:
    """Count failures by error type across all models.

    Args:
        results: Complete benchmark run results.

    Returns:
        Dictionary mapping ValidationStatus to count.
    """
    counts: dict[ValidationStatus, int] = {}

    for model_result in results.model_results:
        for test_result in model_result.test_results:
            if not test_result.passed:
                counts[test_result.status] = counts.get(test_result.status, 0) + 1

    return counts

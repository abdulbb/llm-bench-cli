"""Result comparison functionality for LLM-Bench."""

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from llm_bench.models import BenchmarkRun


@dataclass
class ModelComparison:
    """Comparison of a single model between two runs."""

    model_name: str
    pass_rate_a: float
    pass_rate_b: float
    pass_rate_diff: float
    p95_latency_a: float
    p95_latency_b: float
    latency_diff: float
    latency_diff_pct: float
    cost_a: float
    cost_b: float
    cost_diff: float
    cost_diff_pct: float
    throughput_a: float
    throughput_b: float
    throughput_diff: float
    throughput_diff_pct: float


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""

    run_a_name: str
    run_b_name: str
    run_a_git: str | None
    run_b_git: str | None
    run_a_timestamp: str | None
    run_b_timestamp: str | None
    model_comparisons: list[ModelComparison]
    models_only_in_a: list[str]
    models_only_in_b: list[str]


def load_benchmark_run(path: Path) -> BenchmarkRun:
    """Load a benchmark run from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        BenchmarkRun object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not valid JSON or doesn't match the schema.
    """
    import json

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return BenchmarkRun.model_validate(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from None
    except Exception as e:
        raise ValueError(f"Failed to parse benchmark run from {path}: {e}") from None


def compare_runs(run_a: BenchmarkRun, run_b: BenchmarkRun) -> ComparisonResult:
    """Compare two benchmark runs.

    Args:
        run_a: First benchmark run (baseline).
        run_b: Second benchmark run (to compare).

    Returns:
        ComparisonResult with detailed comparisons.
    """
    # Build model result maps
    models_a = {mr.model_name: mr for mr in run_a.model_results}
    models_b = {mr.model_name: mr for mr in run_b.model_results}

    # Find common and unique models
    common_models = set(models_a.keys()) & set(models_b.keys())
    only_in_a = sorted(set(models_a.keys()) - common_models)
    only_in_b = sorted(set(models_b.keys()) - common_models)

    # Compare common models
    comparisons: list[ModelComparison] = []
    for model_name in sorted(common_models):
        mr_a = models_a[model_name]
        mr_b = models_b[model_name]

        # Calculate differences
        pass_rate_diff = mr_b.pass_rate - mr_a.pass_rate
        latency_diff = mr_b.p95_latency_seconds - mr_a.p95_latency_seconds
        latency_diff_pct = (
            (latency_diff / mr_a.p95_latency_seconds * 100)
            if mr_a.p95_latency_seconds > 0
            else 0
        )
        cost_diff = mr_b.total_cost_usd - mr_a.total_cost_usd
        cost_diff_pct = (
            (cost_diff / mr_a.total_cost_usd * 100) if mr_a.total_cost_usd > 0 else 0
        )
        throughput_diff = (
            mr_b.throughput_tokens_per_second - mr_a.throughput_tokens_per_second
        )
        throughput_diff_pct = (
            (throughput_diff / mr_a.throughput_tokens_per_second * 100)
            if mr_a.throughput_tokens_per_second > 0
            else 0
        )

        comparisons.append(
            ModelComparison(
                model_name=model_name,
                pass_rate_a=mr_a.pass_rate,
                pass_rate_b=mr_b.pass_rate,
                pass_rate_diff=pass_rate_diff,
                p95_latency_a=mr_a.p95_latency_seconds,
                p95_latency_b=mr_b.p95_latency_seconds,
                latency_diff=latency_diff,
                latency_diff_pct=latency_diff_pct,
                cost_a=mr_a.total_cost_usd,
                cost_b=mr_b.total_cost_usd,
                cost_diff=cost_diff,
                cost_diff_pct=cost_diff_pct,
                throughput_a=mr_a.throughput_tokens_per_second,
                throughput_b=mr_b.throughput_tokens_per_second,
                throughput_diff=throughput_diff,
                throughput_diff_pct=throughput_diff_pct,
            )
        )

    # Get git info
    git_a = run_a.git_info.summary() if run_a.git_info else None
    git_b = run_b.git_info.summary() if run_b.git_info else None

    return ComparisonResult(
        run_a_name=run_a.config.name,
        run_b_name=run_b.config.name,
        run_a_git=git_a,
        run_b_git=git_b,
        run_a_timestamp=run_a.run_timestamp,
        run_b_timestamp=run_b.run_timestamp,
        model_comparisons=comparisons,
        models_only_in_a=only_in_a,
        models_only_in_b=only_in_b,
    )


def format_diff(
    value: float, is_percentage: bool = False, higher_is_better: bool = True
) -> str:
    """Format a difference value with color indicators.

    Args:
        value: The difference value.
        is_percentage: Whether the value is a percentage.
        higher_is_better: Whether positive values are good (True) or bad (False).

    Returns:
        Formatted string with color markup.
    """
    if abs(value) < 0.001:
        return "[dim]~0[/dim]"

    sign = "+" if value > 0 else ""
    suffix = "%" if is_percentage else ""

    if value > 0:
        color = "[green]" if higher_is_better else "[red]"
    else:
        color = "[red]" if higher_is_better else "[green]"

    return f"{color}{sign}{value:.2f}{suffix}[/{color.strip('[').strip(']')}]"


def print_comparison(result: ComparisonResult, console: Console | None = None) -> None:
    """Print comparison results to the console.

    Args:
        result: ComparisonResult to display.
        console: Optional Rich console (creates one if not provided).
    """
    if console is None:
        console = Console()

    console.print("\n[bold]Benchmark Comparison[/bold]\n")

    # Print run info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Label", style="dim")
    info_table.add_column("Run A (Baseline)")
    info_table.add_column("Run B (Compare)")

    info_table.add_row("Name", result.run_a_name, result.run_b_name)
    if result.run_a_timestamp or result.run_b_timestamp:
        info_table.add_row(
            "Timestamp",
            result.run_a_timestamp or "N/A",
            result.run_b_timestamp or "N/A",
        )
    if result.run_a_git or result.run_b_git:
        info_table.add_row(
            "Git",
            result.run_a_git or "N/A",
            result.run_b_git or "N/A",
        )

    console.print(info_table)
    console.print()

    # Print comparison table
    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Pass Rate A", justify="right")
    table.add_column("Pass Rate B", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("P95 Latency A", justify="right")
    table.add_column("P95 Latency B", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("Cost A", justify="right")
    table.add_column("Cost B", justify="right")
    table.add_column("Diff", justify="right")

    for comp in result.model_comparisons:
        table.add_row(
            comp.model_name,
            f"{comp.pass_rate_a:.1f}%",
            f"{comp.pass_rate_b:.1f}%",
            format_diff(comp.pass_rate_diff, is_percentage=True, higher_is_better=True),
            f"{comp.p95_latency_a:.3f}s",
            f"{comp.p95_latency_b:.3f}s",
            format_diff(
                comp.latency_diff_pct, is_percentage=True, higher_is_better=False
            ),
            f"${comp.cost_a:.4f}",
            f"${comp.cost_b:.4f}",
            format_diff(comp.cost_diff_pct, is_percentage=True, higher_is_better=False),
        )

    console.print(table)

    # Print models only in one run
    if result.models_only_in_a:
        console.print(
            f"\n[yellow]Models only in Run A:[/yellow] {', '.join(result.models_only_in_a)}"
        )
    if result.models_only_in_b:
        console.print(
            f"\n[yellow]Models only in Run B:[/yellow] {', '.join(result.models_only_in_b)}"
        )

    console.print()

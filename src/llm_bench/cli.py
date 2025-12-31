"""CLI entry point for LLM-Bench."""

import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from llm_bench.cache import get_cache
from llm_bench.config import ConfigError, load_config
from llm_bench.export import export_to_csv, export_to_html, export_to_json
from llm_bench.llm import check_missing_api_keys
from llm_bench.models import BenchConfig, RunConfig
from llm_bench.output import (
    OutputManager,
    create_progress_bar,
    print_failures,
    print_freeform_results,
    print_summary,
    set_output_manager,
)
from llm_bench.runner import run_benchmark

# Version from pyproject.toml
__version__ = "0.1.0"


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"llm-bench version {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="llm-bench",
    help="CLI tool for evaluating and comparing LLMs",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Cache subcommand group
cache_app = typer.Typer(help="Cache management commands")
app.add_typer(cache_app, name="cache")

console = Console()


# Provider to env var mapping (None means no API key required - local models)
PROVIDER_ENV_VARS: dict[str, list[str] | None] = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "azure": ["AZURE_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    # Local model providers (no API key required)
    "ollama": None,
    "ollama_chat": None,
    "lm_studio": None,
    "hosted_vllm": None,
}

# Popular models by provider for the models command
POPULAR_MODELS: dict[str, list[str]] = {
    "openai": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo",
    ],
    "anthropic": [
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-opus-20240229",
    ],
    "google": [
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-pro",
    ],
    "openrouter": [
        "openrouter/google/gemma-2-9b-it:free",
        "openrouter/meta-llama/llama-3-8b-instruct:free",
        "openrouter/mistralai/mistral-7b-instruct:free",
    ],
    "mistral": [
        "mistral/mistral-large-latest",
        "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest",
    ],
    "groq": [
        "groq/llama-3.1-70b-versatile",
        "groq/llama-3.1-8b-instant",
        "groq/mixtral-8x7b-32768",
    ],
    # Local model providers
    "ollama": [
        "ollama/llama3.1",
        "ollama/llama3.1:70b",
        "ollama/mistral",
        "ollama/codellama",
        "ollama/phi3",
    ],
}


@app.callback()
def main(
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            help="Path to .env file for loading environment variables.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    _version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """LLM-Bench: CLI tool for evaluating and comparing LLMs."""
    # Load environment variables from .env files
    # Priority: --env-file > .env.local > .env
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        # Auto-discover .env files
        load_dotenv(".env.local", override=True)
        load_dotenv(".env", override=False)


class ExportFormat(str, Enum):
    """Supported export formats."""

    HTML = "html"
    CSV = "csv"
    JSON = "json"


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the benchmark configuration YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("bench.config.yaml"),
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "-m",
            help="Model to benchmark. Can be specified multiple times to override config.",
        ),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option(
            "--concurrency",
            "-j",
            help="Number of parallel requests (1-100).",
            min=1,
            max=100,
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--temperature",
            "-t",
            help="Generation temperature (0.0-2.0).",
            min=0.0,
            max=2.0,
        ),
    ] = None,
    max_cost: Annotated[
        float | None,
        typer.Option(
            "--max-cost",
            help="Maximum cost in USD. Execution stops if exceeded.",
            min=0.0,
        ),
    ] = None,
    validators_file: Annotated[
        Path | None,
        typer.Option(
            "--validators-file",
            help="Path to a Python file containing custom validator functions.",
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Bypass cache and force fresh API calls.",
        ),
    ] = False,
    export: Annotated[
        ExportFormat | None,
        typer.Option(
            "--export",
            "-e",
            help="Export results to file format.",
            case_sensitive=False,
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Custom filename for the export.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            resolve_path=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output with debug information.",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress all output except errors. Useful for CI/CD.",
        ),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable colored output. Useful for non-TTY environments.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate config and show plan without executing benchmarks.",
        ),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast",
            help="Stop benchmark on first test failure.",
        ),
    ] = False,
) -> None:
    """Run benchmark tests against LLM models.

    Loads configuration from a YAML file and executes benchmark tests
    against the specified models. Results are displayed in the terminal
    with optional export to HTML, CSV, or JSON.
    """
    # Set up output manager
    out = OutputManager(verbose=verbose, quiet=quiet, no_color=no_color)
    set_output_manager(out)

    # Load and validate config
    try:
        bench_config = load_config(config)
    except ConfigError as e:
        out.error(str(e))
        raise typer.Exit(code=1) from None

    # Apply CLI overrides
    bench_config = apply_cli_overrides(
        bench_config,
        models=model,
        concurrency=concurrency,
        temperature=temperature,
        max_cost=max_cost,
        validators_file=validators_file,
    )

    # Display configuration summary
    out.debug(f"Config file: {config}")
    out.print(f"\n[bold]Benchmark:[/bold] {bench_config.name}")
    out.print(f"[bold]Models:[/bold] {', '.join(bench_config.models)}")
    out.print(f"[bold]Test cases:[/bold] {len(bench_config.test_cases)}")
    out.print(f"[bold]Concurrency:[/bold] {bench_config.config.concurrency}")
    out.print(f"[bold]Temperature:[/bold] {bench_config.config.temperature}")
    if bench_config.config.max_cost:
        out.print(f"[bold]Max Cost:[/bold] ${bench_config.config.max_cost:.2f}")
    out.print(f"[bold]Cache:[/bold] {'disabled' if no_cache else 'enabled'}")

    # Check for missing API keys
    missing_keys = check_missing_api_keys(bench_config.models)
    if missing_keys:
        out.print("\n[yellow bold]Warning: Missing Environment Variables[/yellow bold]")
        for key in missing_keys:
            out.print(f"[yellow]  - {key}[/yellow]")
        out.print(
            "[yellow]Benchmarking may fail for these models if keys are not set.[/yellow]\n"
        )

    if export:
        out.print(f"[bold]Export:[/bold] {export.value}")

    out.newline()

    # Dry run mode - show plan and exit
    if dry_run:
        total_tasks = len(bench_config.models) * len(bench_config.test_cases)
        out.print("[bold cyan]Dry Run Mode[/bold cyan] - No API calls will be made\n")
        out.print(f"[bold]Total tasks:[/bold] {total_tasks}")
        out.print(f"  - {len(bench_config.models)} model(s)")
        out.print(f"  - {len(bench_config.test_cases)} test case(s) each")
        if bench_config.is_freeform:
            out.print(
                "\n[bold cyan]Freeform Mode:[/bold cyan] "
                "No validation criteria defined. Outputs will be shown for manual inspection."
            )
        out.newline()

        if missing_keys:
            out.warning("Some API keys are missing. Benchmark may fail.")
        else:
            out.success("All required API keys are configured.")

        out.newline()
        out.print("[dim]Run without --dry-run to execute the benchmark.[/dim]")
        return

    # Run the benchmark with progress bar
    total_tasks = len(bench_config.models) * len(bench_config.test_cases)

    with create_progress_bar(no_color=no_color) as progress:
        task_id = progress.add_task("Running", total=total_tasks)

        def on_progress(completed: int, _total: int) -> None:
            progress.update(task_id, completed=completed)

        # Run async benchmark
        out.debug("Starting async benchmark execution")
        results = asyncio.run(
            run_benchmark(
                config=bench_config,
                use_cache=not no_cache,
                on_progress=on_progress,
                fail_fast=fail_fast,
            )
        )

    # Display summary table
    if bench_config.is_freeform:
        # Freeform mode: show raw outputs for manual inspection
        print_freeform_results(results, out.console)
    else:
        # Standard mode: show pass rates and validation results
        print_summary(results, out.console)

        # Display failure details
        print_failures(results, out.console)

    # Handle export if requested
    if export:
        output_file = output or Path(f"report.{export.value}")
        if export == ExportFormat.HTML:
            export_to_html(results, output_file)
        elif export == ExportFormat.CSV:
            export_to_csv(results, output_file)
        elif export == ExportFormat.JSON:
            export_to_json(results, output_file)

        out.success(f"Results exported to {output_file}")


def apply_cli_overrides(
    config: BenchConfig,
    models: list[str] | None = None,
    concurrency: int | None = None,
    temperature: float | None = None,
    max_cost: float | None = None,
    validators_file: Path | None = None,
) -> BenchConfig:
    """Apply CLI flag overrides to the configuration.

    Args:
        config: The loaded benchmark configuration.
        models: Optional list of models to override config models.
        concurrency: Optional concurrency value to override.
        temperature: Optional temperature value to override.
        max_cost: Optional max cost value to override.
        validators_file: Optional validators file path to override.

    Returns:
        Updated configuration with CLI overrides applied.
    """
    updates: dict[str, object] = {}

    if models:
        updates["models"] = models

    if validators_file:
        updates["validators_file"] = validators_file

    if concurrency is not None or temperature is not None or max_cost is not None:
        new_run_config = RunConfig(
            concurrency=(
                concurrency if concurrency is not None else config.config.concurrency
            ),
            temperature=(
                temperature if temperature is not None else config.config.temperature
            ),
            max_cost=(max_cost if max_cost is not None else config.config.max_cost),
            judge_model=config.config.judge_model,
        )
        updates["config"] = new_run_config

    if updates:
        return BenchConfig(**{**config.model_dump(), **updates})

    return config


@app.command()
def validate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to the benchmark configuration YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("bench.config.yaml"),
) -> None:
    """Validate configuration file without running benchmarks.

    Checks YAML syntax, validates against schema, and verifies
    all referenced files exist and API keys are configured.
    """
    console.print(f"\n[bold]Validating:[/bold] {config}\n")

    # Load and validate config
    try:
        bench_config = load_config(config)
    except ConfigError as e:
        console.print("[red]Validation Failed[/red]\n")
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Check for referenced files
    issues: list[str] = []

    if bench_config.schema_path and not bench_config.schema_path.exists():
        issues.append(f"Schema file not found: {bench_config.schema_path}")

    if bench_config.validators_file and not bench_config.validators_file.exists():
        issues.append(f"Validators file not found: {bench_config.validators_file}")

    if bench_config.test_cases_file and not bench_config.test_cases_file.exists():
        issues.append(f"Test cases file not found: {bench_config.test_cases_file}")

    # Check API keys
    missing_keys = check_missing_api_keys(bench_config.models)

    # Display results
    console.print(f"[bold]Benchmark Name:[/bold] {bench_config.name}")
    console.print(f"[bold]Models:[/bold] {len(bench_config.models)}")
    console.print(f"[bold]Test Cases:[/bold] {len(bench_config.test_cases)}")
    console.print()

    if issues:
        console.print("[yellow bold]Warnings:[/yellow bold]")
        for issue in issues:
            console.print(f"[yellow]  - {issue}[/yellow]")
        console.print()

    if missing_keys:
        console.print("[yellow bold]Missing API Keys:[/yellow bold]")
        for key in missing_keys:
            console.print(f"[yellow]  - {key}[/yellow]")
        console.print()

    if not issues and not missing_keys:
        console.print("[green bold]Configuration is valid![/green bold]")
    elif not issues:
        console.print("[green]Configuration syntax is valid.[/green]")
        console.print("[yellow]Warning: Some API keys may be missing.[/yellow]")
    else:
        console.print("[yellow]Configuration has warnings. Review above.[/yellow]")


@app.command()
def models(
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Filter by provider (openai, anthropic, google, etc.)",
        ),
    ] = None,
) -> None:
    """List available models and their providers.

    Shows popular models grouped by provider, along with the
    required environment variables for each.
    """
    console.print("\n[bold]Available Models[/bold]\n")

    providers_to_show = [provider.lower()] if provider else list(POPULAR_MODELS.keys())

    for prov in providers_to_show:
        if prov not in POPULAR_MODELS:
            if provider:  # User specified this provider
                console.print(f"[yellow]Unknown provider: {prov}[/yellow]")
            continue

        # Check if API key is configured (local models don't need keys)
        env_vars = PROVIDER_ENV_VARS.get(prov)
        if env_vars is None:
            # Local provider - no API key needed
            key_status = "[cyan]local[/cyan]"
            env_var_display = "No API key required"
        else:
            key_configured = any(os.getenv(var) for var in env_vars)
            key_status = (
                "[green]configured[/green]"
                if key_configured
                else "[yellow]not set[/yellow]"
            )
            env_var_display = " or ".join(env_vars)

        # Provider header
        console.print(f"[bold cyan]{prov.upper()}[/bold cyan] ({key_status})")
        console.print(f"  [dim]Env var: {env_var_display}[/dim]")

        # Models
        for model in POPULAR_MODELS[prov]:
            console.print(f"    {model}")

        console.print()

    console.print("[dim]Use --provider to filter by specific provider.[/dim]")
    console.print("[dim]Model format: provider/model-name (e.g., openai/gpt-4o)[/dim]")


@cache_app.command("info")
def cache_info() -> None:
    """Show cache statistics and location."""
    cache = get_cache()
    stats = cache.get_stats()

    console.print("\n[bold]Cache Information[/bold]\n")
    console.print(f"[bold]Location:[/bold] {stats['path']}")
    console.print(f"[bold]Entries:[/bold] {stats['entry_count']}")

    # Format size
    size_bytes = stats["size_bytes"]
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

    console.print(f"[bold]Size:[/bold] {size_str}")
    console.print()


@cache_app.command("clear")
def cache_clear(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Skip confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Clear all cached responses."""
    cache = get_cache()
    stats = cache.get_stats()

    if stats["entry_count"] == 0:
        console.print("[yellow]Cache is already empty.[/yellow]")
        return

    if not force:
        console.print(f"\n[bold]Cache contains {stats['entry_count']} entries.[/bold]")
        confirm = typer.confirm("Are you sure you want to clear the cache?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Abort()

    cache.clear()
    console.print("[green]Cache cleared successfully.[/green]")


@app.command()
def init(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for the generated configuration file.",
        ),
    ] = Path("bench.config.yaml"),
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Generate a template without prompts.",
        ),
    ] = False,
) -> None:
    """Initialize a new benchmark configuration file.

    Runs an interactive wizard to create a benchmark configuration,
    or generates a template file with --non-interactive.
    """
    if output.exists():
        console.print(f"[yellow]Warning: {output} already exists.[/yellow]")
        if not typer.confirm("Overwrite?"):
            console.print("[dim]Aborted.[/dim]")
            raise typer.Abort()

    if non_interactive:
        # Generate template
        _generate_template_config(output)
        console.print(f"\n[green]Template configuration created: {output}[/green]")
        console.print("[dim]Edit the file to customize your benchmark.[/dim]")
        return

    # Interactive wizard
    from llm_bench.init import run_init_wizard

    run_init_wizard(output)


@app.command()
def compare(
    run_a: Annotated[
        Path,
        typer.Argument(
            help="Path to first benchmark result JSON file (baseline).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    run_b: Annotated[
        Path,
        typer.Argument(
            help="Path to second benchmark result JSON file (to compare).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Compare two benchmark result files.

    Displays a side-by-side comparison of pass rates, latency,
    and cost between two benchmark runs. Useful for regression testing
    or comparing different model versions.

    Export benchmark results to JSON first using:
        llm-bench run --export json --output results.json
    """
    from llm_bench.compare import (
        compare_runs,
        load_benchmark_run,
        print_comparison,
    )

    try:
        console.print(f"[dim]Loading {run_a}...[/dim]")
        benchmark_a = load_benchmark_run(run_a)

        console.print(f"[dim]Loading {run_b}...[/dim]")
        benchmark_b = load_benchmark_run(run_b)

        result = compare_runs(benchmark_a, benchmark_b)
        print_comparison(result, console)

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from None


def _generate_template_config(output_path: Path) -> None:
    """Generate a template configuration file."""
    import yaml

    template = {
        "name": "My Benchmark",
        "system_prompt": "You are a helpful assistant. Respond in JSON format.",
        "models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-haiku-20241022",
        ],
        "config": {
            "concurrency": 5,
            "temperature": 0.1,
        },
        "test_cases": [
            {
                "input": "What is the capital of France?",
                "expected": {
                    "country": "France",
                    "capital": "Paris",
                },
            },
            {
                "input": "What is 2 + 2?",
                "expected": {
                    "result": 4,
                },
            },
        ],
    }

    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    app()

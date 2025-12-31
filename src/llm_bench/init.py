"""Interactive configuration wizard for LLM-Bench."""

import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

# Provider configurations
PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "is_local": False,
        "models": [
            ("openai/gpt-4o", "Latest flagship model"),
            ("openai/gpt-4o-mini", "Cost-effective, fast"),
            ("openai/gpt-4-turbo", "Previous flagship"),
            ("openai/gpt-3.5-turbo", "Legacy, very cheap"),
        ],
    },
    "anthropic": {
        "name": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "is_local": False,
        "models": [
            ("anthropic/claude-3-5-sonnet-20241022", "Best balance of speed/quality"),
            ("anthropic/claude-3-5-haiku-20241022", "Fastest, most affordable"),
            ("anthropic/claude-3-opus-20240229", "Most capable"),
        ],
    },
    "google": {
        "name": "Google (Gemini)",
        "env_var": "GEMINI_API_KEY",
        "is_local": False,
        "models": [
            ("gemini/gemini-1.5-pro", "Best for complex tasks"),
            ("gemini/gemini-1.5-flash", "Fast and efficient"),
            ("gemini/gemini-pro", "Previous generation"),
        ],
    },
    "openrouter": {
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "is_local": False,
        "models": [
            ("openrouter/google/gemma-2-9b-it:free", "Free tier model"),
            ("openrouter/meta-llama/llama-3-8b-instruct:free", "Free Llama model"),
            ("openrouter/mistralai/mistral-7b-instruct:free", "Free Mistral model"),
        ],
    },
    "groq": {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "is_local": False,
        "models": [
            ("groq/llama-3.1-70b-versatile", "Large Llama model"),
            ("groq/llama-3.1-8b-instant", "Fast, small Llama"),
            ("groq/mixtral-8x7b-32768", "Mixtral MoE"),
        ],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "env_var": None,
        "is_local": True,
        "api_base": "http://localhost:11434",
        "models": [
            ("ollama/llama3.1", "Meta Llama 3.1"),
            ("ollama/llama3.1:70b", "Llama 3.1 70B"),
            ("ollama/mistral", "Mistral 7B"),
            ("ollama/codellama", "Code Llama"),
            ("ollama/phi3", "Microsoft Phi-3"),
        ],
    },
    "lm_studio": {
        "name": "LM Studio (Local)",
        "env_var": None,
        "is_local": True,
        "api_base": "http://localhost:1234/v1",
        "models": [
            ("openai/local-model", "Your loaded model (via OpenAI-compatible API)"),
        ],
    },
}


def run_init_wizard(output_path: Path) -> None:
    """Run the interactive configuration wizard.

    Args:
        output_path: Path where the configuration file will be saved.
    """
    console.print()
    console.print(
        Panel(
            "[bold cyan]LLM-Bench Configuration Wizard[/bold cyan]\n\n"
            "This wizard will help you create a benchmark configuration file.\n"
            "Press Ctrl+C at any time to cancel.",
            expand=False,
        )
    )
    console.print()

    config: dict[str, Any] = {}

    # Step 1: Benchmark name
    console.print("[bold]Step 1: Benchmark Name[/bold]")
    config["name"] = Prompt.ask(
        "Enter a name for your benchmark",
        default="My LLM Benchmark",
    )
    console.print()

    # Step 2: System prompt
    console.print("[bold]Step 2: System Prompt[/bold]")
    console.print("[dim]This prompt is sent to all models for every test.[/dim]")
    console.print(
        "[dim]Tip: Ask models to respond in JSON format for easy validation.[/dim]"
    )
    console.print()

    default_prompt = "You are a helpful assistant. Always respond with valid JSON."
    config["system_prompt"] = Prompt.ask(
        "Enter your system prompt",
        default=default_prompt,
    )
    console.print()

    # Step 3: Select providers
    console.print("[bold]Step 3: Select Providers[/bold]")
    console.print("[dim]Choose which LLM providers to benchmark.[/dim]")
    console.print()

    # Show available providers
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Provider")
    table.add_column("Status")

    provider_list = list(PROVIDERS.keys())
    for i, key in enumerate(provider_list, 1):
        prov = PROVIDERS[key]
        if prov.get("is_local"):
            status = "[cyan]local[/cyan]"
        else:
            env_var = str(prov["env_var"])
            is_configured = bool(os.getenv(env_var))
            status = (
                "[green]configured[/green]"
                if is_configured
                else "[yellow]not set[/yellow]"
            )
        table.add_row(str(i), str(prov["name"]), status)

    console.print(table)
    console.print()

    selected_providers: list[str] = []
    provider_input = Prompt.ask(
        "Enter provider numbers (comma-separated, e.g., 1,2)",
        default="1",
    )

    try:
        indices = [int(x.strip()) for x in provider_input.split(",")]
        for idx in indices:
            if 1 <= idx <= len(provider_list):
                selected_providers.append(provider_list[idx - 1])
    except ValueError:
        console.print("[yellow]Invalid input, defaulting to OpenAI[/yellow]")
        selected_providers = ["openai"]

    console.print(f"[dim]Selected: {', '.join(selected_providers)}[/dim]")
    console.print()

    # Step 4: Select models
    console.print("[bold]Step 4: Select Models[/bold]")
    console.print("[dim]Choose models from each selected provider.[/dim]")
    console.print()

    selected_models: list[str] = []
    model_configs: dict[str, dict[str, Any]] = {}

    for provider_key in selected_providers:
        prov = PROVIDERS[provider_key]
        console.print(f"[bold cyan]{prov['name']} Models:[/bold cyan]")

        # For local providers, show a note about the server
        if prov.get("is_local"):
            api_base = prov.get("api_base", "http://localhost:11434")
            console.print(f"[dim]Server: {api_base}[/dim]")

        model_table = Table(show_header=True, header_style="bold")
        model_table.add_column("#", style="dim", width=3)
        model_table.add_column("Model")
        model_table.add_column("Description")

        for i, (model_name, description) in enumerate(prov["models"], 1):
            model_table.add_row(str(i), model_name, description)

        console.print(model_table)

        model_input = Prompt.ask(
            "Select models (comma-separated, e.g., 1,2) or 'all'",
            default="1",
        )

        models_to_add: list[str] = []
        if model_input.lower() == "all":
            models_to_add = [m[0] for m in prov["models"]]
        else:
            try:
                indices = [int(x.strip()) for x in model_input.split(",")]
                for idx in indices:
                    if 1 <= idx <= len(prov["models"]):
                        models_to_add.append(prov["models"][idx - 1][0])
            except ValueError:
                # Default to first model
                models_to_add.append(prov["models"][0][0])

        # Add models and track model_configs for local providers
        for model_name in models_to_add:
            selected_models.append(model_name)
            # For local providers with custom api_base, add to model_configs
            if prov.get("is_local") and prov.get("api_base"):
                model_configs[model_name] = {"api_base": prov["api_base"]}

        console.print()

    config["models"] = selected_models
    if model_configs:
        config["model_configs"] = model_configs
    console.print(f"[dim]Selected {len(selected_models)} model(s)[/dim]")
    console.print()

    # Step 5: Test cases
    console.print("[bold]Step 5: Test Cases[/bold]")
    console.print("[dim]Add test cases with expected outputs for validation.[/dim]")
    console.print()

    test_cases: list[dict[str, Any]] = []

    # Add sample test cases
    if Confirm.ask("Add sample test cases to get started?", default=True):
        test_cases.extend(
            [
                {
                    "input": "What is the capital of France? Respond with JSON containing 'country' and 'capital' fields.",
                    "expected": {"country": "France", "capital": "Paris"},
                },
                {
                    "input": "Calculate 15 + 27. Respond with JSON containing a 'result' field.",
                    "expected": {"result": 42},
                },
            ]
        )
        console.print("[green]Added 2 sample test cases.[/green]")

    # Allow adding custom test cases
    while Confirm.ask("Add a custom test case?", default=False):
        console.print()
        test_input = Prompt.ask("Enter the input prompt")
        console.print("[dim]Expected output (leave empty to skip validation):[/dim]")

        expected_str = Prompt.ask(
            'Expected JSON (e.g., {"key": "value"})',
            default="",
        )

        test_case: dict[str, Any] = {"input": test_input}
        if expected_str:
            try:
                import json

                test_case["expected"] = json.loads(expected_str)
            except json.JSONDecodeError:
                console.print("[yellow]Invalid JSON, skipping expected value[/yellow]")

        test_cases.append(test_case)
        console.print("[green]Test case added.[/green]")
        console.print()

    config["test_cases"] = test_cases
    console.print()

    # Step 6: Configuration options
    console.print("[bold]Step 6: Configuration Options[/bold]")

    run_config: dict[str, Any] = {}

    concurrency_str = Prompt.ask(
        "Concurrency (parallel requests)",
        default="5",
    )
    try:
        run_config["concurrency"] = min(100, max(1, int(concurrency_str)))
    except ValueError:
        run_config["concurrency"] = 5

    temperature_str = Prompt.ask(
        "Temperature (0.0-2.0)",
        default="0.1",
    )
    try:
        run_config["temperature"] = min(2.0, max(0.0, float(temperature_str)))
    except ValueError:
        run_config["temperature"] = 0.1

    config["config"] = run_config
    console.print()

    # Summary
    console.print("[bold]Configuration Summary[/bold]")
    console.print(f"  Name: {config['name']}")
    console.print(f"  Models: {len(config['models'])}")
    console.print(f"  Test Cases: {len(config['test_cases'])}")
    console.print(f"  Concurrency: {run_config['concurrency']}")
    console.print(f"  Temperature: {run_config['temperature']}")
    console.print()

    # Write file
    if Confirm.ask(f"Save configuration to {output_path}?", default=True):
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        console.print(
            f"\n[green bold]Configuration saved to {output_path}[/green bold]"
        )
        console.print()
        console.print("[dim]Next steps:[/dim]")
        console.print(f"  1. Review and edit {output_path} as needed")
        console.print(f"  2. Run: [bold]llm-bench run -c {output_path}[/bold]")
        console.print(
            f"  3. Or validate first: [bold]llm-bench validate -c {output_path}[/bold]"
        )
    else:
        console.print("[dim]Configuration not saved.[/dim]")

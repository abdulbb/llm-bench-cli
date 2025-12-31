<div align="center">

# LLM-Bench

A developer-centric CLI tool to systematically evaluate and compare Large Language Models (LLMs). Define prompts, test cases, and expected outputs, run them concurrently against multiple providers, and get detailed performance metrics (Accuracy, Latency, Throughput, Cost) in your terminal.

![Demo](docs/demo.gif)

</div>

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Google Gemini, Groq, Mistral, Cohere, Together AI, Azure, and more (via `litellm`).
- **Rich Terminal Output**: Real-time progress bars with ETA.
- **Detailed Metrics**: Pass rates, P95 latency, TTFT, tokens/sec, and cost.
- **Flexible Validation**: Regex patterns, custom Python validators, JSON Schema, LLM judges.
- **ROUGE-L Scoring**: Compare outputs against reference texts.
- **Concurrent Execution**: Fast parallel testing with rate limiting.
- **Caching**: Saves money on repeated runs.
- **Export**: HTML, CSV, or JSON reports with interactive charts.
- **Comparison**: Compare benchmark results across runs.
- **External Data**: Load test cases from CSV or JSONL files.
- **Interactive Setup**: Config wizard to get started quickly.
- **CI/CD Friendly**: Quiet mode, no-color output, fail-fast, and dry-run support.

## Installation

```bash
pip install llm-bench-cli
```

Or install from source:

```bash
git clone https://github.com/abdulbb/llm-bench-cli.git
cd llm-bench-cli
pip install -e .
```

 ## Setting Up API Keys

 Set your API keys as environment variables before running benchmarks:

 ```bash
 export OPENAI_API_KEY="sk-..."
 export ANTHROPIC_API_KEY="sk-ant-..."
 export GEMINI_API_KEY="..."
 export GROQ_API_KEY="gsk_..."
 export OPENROUTER_API_KEY="sk-or-..."
 ```

 To make these persistent across terminal sessions, add them to your `~/.bashrc`, `~/.zshrc`, or use a tool like `direnv`.

 For more detailed setup instructions, see [**Provider Setup**](docs/providers.md).

 ## Documentation

*   [**CLI Reference**](docs/cli.md): All commands and options.
*   [**Configuration Guide**](docs/configuration.md): Learn how to write `bench.config.yaml`.
*   [**Real-World Examples**](docs/examples.md): **Start Here!** Code generation, summarization, and data extraction scenarios.
*   [**Evaluation & Validation**](docs/evaluation.md): How results are judged (Regex, Custom Validators, JSON Schema, LLM Judge).
*   [**Exporting Results**](docs/export.md): Details on HTML, CSV, and JSON output.
*   [**Provider Setup**](docs/providers.md): How to setup API keys for OpenAI, Gemini, Groq, etc.
*   [**Development**](docs/development.md): How to contribute to this project.

## Quick Start

**1. Create a new benchmark configuration:**
```bash
# Interactive wizard
llm-bench init

# Or generate a template
llm-bench init --non-interactive
```

**2. Validate your configuration:**
```bash
llm-bench validate --config bench.config.yaml
```

**3. Preview without making API calls:**
```bash
llm-bench run --config bench.config.yaml --dry-run
```

**4. Run the benchmark:**
```bash
llm-bench run --config bench.config.yaml
```

## Usage Examples

**Run a Standard Benchmark:**
```bash
llm-bench run --config code_gen.yaml
```

**Compare Models on the Fly:**
```bash
llm-bench run --model openai/gpt-4o --model anthropic/claude-3-5-sonnet-20241022
```

**Generate an Interactive Report:**
```bash
llm-bench run --config summarization.yaml --export html --output report.html
```

**Set Safety Limits:**
```bash
llm-bench run --config expensive_test.yaml --max-cost 1.0
```

**Stop on First Failure:**
```bash
llm-bench run --config ci_tests.yaml --fail-fast
```

**CI/CD Mode (quiet, no colors):**
```bash
llm-bench run --config bench.config.yaml --quiet --no-color
```

**List Available Models:**
```bash
llm-bench models
llm-bench models --provider openai
llm-bench models --provider groq
```

**Manage Cache:**
```bash
llm-bench cache info
llm-bench cache clear
```

**Compare Benchmark Results:**
```bash
llm-bench compare run1.json run2.json --export html --output comparison.html
```

**Enable Shell Completions:**
```bash
# Install completions for your shell
llm-bench --install-completion

# Show completion script without installing
llm-bench --show-completion
```

See [**CLI Reference**](docs/cli.md) for all commands and options, and [**Real-World Examples**](docs/examples.md) for full configuration files.

## Example Configuration

```yaml
name: "Code Generation Benchmark"
system_prompt: "You are a Python coding assistant."
models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"
  - "groq/llama-3.1-70b-versatile"

validators_file: "validators.py"

test_cases:
  - input: "Write a function to calculate factorial"
    regex_pattern: "def factorial"
    validator: "is_valid_python"
    
  - input: "What is 2+2?"
    expected: 4
```

## License

MIT

# CLI Reference

Complete reference for all `llm-bench` commands and options.

## Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Run benchmark tests against LLM models |
| `init` | Initialize a new benchmark configuration file |
| `validate` | Validate configuration file without running |
| `models` | List available models and providers |
| `compare` | Compare two benchmark result files |
| `cache info` | Show cache statistics |
| `cache clear` | Clear cached responses |

## Global Options

| Option | Description |
|--------|-------------|
| `--env-file` | Path to .env file for loading environment variables |
| `--install-completion` | Install shell completion for current shell |
| `--show-completion` | Show completion script for current shell |

### Environment Variable Loading

LLM-Bench automatically loads environment variables from `.env` files:

1. `.env.local` (loaded first, can override)
2. `.env` (loaded second)

You can also specify a custom file:
```bash
llm-bench --env-file my-config.env run --config bench.config.yaml
```

### Shell Completion

Install shell completion for your shell:
```bash
# Bash
llm-bench --install-completion bash

# Zsh
llm-bench --install-completion zsh

# Fish
llm-bench --install-completion fish
```

---

## `llm-bench run`

Run benchmark tests against LLM models.

```bash
llm-bench run [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to benchmark configuration YAML file | `bench.config.yaml` |
| `--model` | `-m` | Model to benchmark (can be repeated) | From config |
| `--concurrency` | `-j` | Number of parallel requests (1-100) | From config |
| `--temperature` | `-t` | Generation temperature (0.0-2.0) | From config |
| `--max-cost` | | Maximum cost in USD before stopping | None |
| `--validators-file` | | Path to custom validator Python file | None |
| `--no-cache` | | Bypass cache and force fresh API calls | False |
| `--export` | `-e` | Export format: `html`, `csv`, or `json` | None |
| `--output` | `-o` | Custom filename for export | `report.<format>` |
| `--verbose` | `-v` | Enable verbose/debug output | False |
| `--quiet` | `-q` | Suppress all output except errors | False |
| `--no-color` | | Disable colored output | False |
| `--dry-run` | | Validate and show plan without executing | False |
| `--fail-fast` | | Stop on first test failure | False |

### Examples

```bash
# Basic run
llm-bench run --config bench.config.yaml

# Override models
llm-bench run -c bench.config.yaml -m openai/gpt-4o -m anthropic/claude-3-5-sonnet-20241022

# Dry run (no API calls)
llm-bench run --config bench.config.yaml --dry-run

# CI/CD friendly
llm-bench run --config bench.config.yaml --quiet --no-color

# With export
llm-bench run --config bench.config.yaml --export html --output results.html

# Cost limit
llm-bench run --config bench.config.yaml --max-cost 5.0

# Verbose debugging
llm-bench run --config bench.config.yaml --verbose

# Stop on first failure
llm-bench run --config bench.config.yaml --fail-fast
```

---

## `llm-bench init`

Initialize a new benchmark configuration file interactively or from a template.

```bash
llm-bench init [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output` | `-o` | Output path for configuration file | `bench.config.yaml` |
| `--non-interactive` | | Generate template without prompts | False |

### Examples

```bash
# Interactive wizard
llm-bench init

# Generate template
llm-bench init --non-interactive

# Custom output path
llm-bench init --output my-benchmark.yaml
```

### Interactive Wizard Steps

1. **Benchmark Name**: Name for your benchmark
2. **System Prompt**: Prompt sent to all models
3. **Select Providers**: Choose from OpenAI, Anthropic, Google, OpenRouter, Groq
4. **Select Models**: Pick models from each provider
5. **Test Cases**: Add sample or custom test cases
6. **Configuration**: Set concurrency and temperature

---

## `llm-bench validate`

Validate a configuration file without running benchmarks. Checks:
- YAML syntax
- Required fields
- Referenced file existence
- API key availability

```bash
llm-bench validate [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration file | `bench.config.yaml` |

### Examples

```bash
# Validate default config
llm-bench validate

# Validate specific file
llm-bench validate --config my-benchmark.yaml
```

### Output

```
Validating: bench.config.yaml

Benchmark Name: My Benchmark
Models: 2
Test Cases: 5

Configuration is valid!
```

Or with warnings:

```
Missing API Keys:
  - OPENAI_API_KEY (for OpenAI models)

Configuration syntax is valid.
Warning: Some API keys may be missing.
```

---

## `llm-bench models`

List available models grouped by provider, with API key status.

```bash
llm-bench models [OPTIONS]
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--provider` | `-p` | Filter by provider name | All |

### Examples

```bash
# List all providers
llm-bench models

# Filter by provider
llm-bench models --provider openai
llm-bench models -p anthropic
llm-bench models -p groq
```

### Output

```
Available Models

OPENAI (configured)
  Env var: OPENAI_API_KEY
    openai/gpt-4o
    openai/gpt-4o-mini
    openai/gpt-4-turbo
    openai/gpt-3.5-turbo

ANTHROPIC (not set)
  Env var: ANTHROPIC_API_KEY
    anthropic/claude-3-5-sonnet-20241022
    anthropic/claude-3-5-haiku-20241022
    anthropic/claude-3-opus-20240229

GOOGLE (not set)
  Env var: GEMINI_API_KEY or GOOGLE_API_KEY
    gemini/gemini-1.5-pro
    gemini/gemini-1.5-flash
    gemini/gemini-pro

OPENROUTER (not set)
  Env var: OPENROUTER_API_KEY
    openrouter/google/gemma-2-9b-it:free
    openrouter/meta-llama/llama-3-8b-instruct:free
    openrouter/mistralai/mistral-7b-instruct:free

MISTRAL (not set)
  Env var: MISTRAL_API_KEY
    mistral/mistral-large-latest
    mistral/mistral-medium-latest
    mistral/mistral-small-latest

GROQ (not set)
  Env var: GROQ_API_KEY
    groq/llama-3.1-70b-versatile
    groq/llama-3.1-8b-instant
    groq/mixtral-8x7b-32768
```

---

## `llm-bench compare`

Compare two benchmark result files side-by-side. Useful for regression testing or comparing different model versions.

```bash
llm-bench compare RUN_A RUN_B
```

### Arguments

| Argument | Description |
|----------|-------------|
| `RUN_A` | Path to first benchmark result JSON file (baseline) |
| `RUN_B` | Path to second benchmark result JSON file (to compare) |

### Examples

```bash
# Compare two benchmark runs
llm-bench compare baseline.json new-run.json

# First export results as JSON, then compare
llm-bench run --config config.yaml --export json --output v1.json
# ... make changes ...
llm-bench run --config config.yaml --export json --output v2.json
llm-bench compare v1.json v2.json
```

### Output

The compare command displays:
- **Run metadata**: Names, timestamps, and git info for both runs
- **Model comparison table**: Pass rate, P95 latency, throughput, and cost differences
- **Difference indicators**: Green for improvements, red for regressions
- **Model coverage**: Lists models only present in one run

Example output:
```
Benchmark Comparison

Name       Run A (Baseline)    Run B (Compare)
Timestamp  2024-01-01T00:00    2024-01-02T00:00
Git        main abc123d        main def456d

Model Comparison
Model          Pass Rate A  Pass Rate B  Diff    P95 Latency...
gpt-4o         95.0%        98.0%        +3.00%  0.850s...
claude-3       92.0%        90.0%        -2.00%  1.200s...
```

---

## `llm-bench cache`

Cache management subcommands.

### `llm-bench cache info`

Show cache statistics including location, entry count, and size.

```bash
llm-bench cache info
```

#### Output

```
Cache Information

Location: /Users/you/.cache/llm-bench
Entries: 42
Size: 1.2 MB
```

### `llm-bench cache clear`

Clear all cached responses.

```bash
llm-bench cache clear [OPTIONS]
```

#### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--force` | `-f` | Skip confirmation prompt | False |

#### Examples

```bash
# With confirmation prompt
llm-bench cache clear

# Skip confirmation
llm-bench cache clear --force
```

---

## Global Behavior

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration or runtime error |
| 2 | Invalid arguments |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `GOOGLE_API_KEY` | Alternative for Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `GROQ_API_KEY` | Groq API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `COHERE_API_KEY` | Cohere API key |
| `TOGETHER_API_KEY` | Together AI API key |
| `AZURE_API_KEY` | Azure OpenAI API key |

See [Provider Setup](providers.md) for detailed instructions.

### Cache Location

Responses are cached in `~/.cache/llm-bench/` by default with secure permissions (owner read/write only).

---

## Metrics Explained

### Latency Metrics

| Metric | Description |
|--------|-------------|
| **Total Latency** | Complete request-response time in seconds |
| **TTFT (Time to First Token)** | Time until first token received (streaming only) |
| **P95 Latency** | 95th percentile latency across all tests for a model |

### Throughput

| Metric | Description |
|--------|-------------|
| **Tokens/sec** | Total tokens (prompt + completion) divided by total time |

### Cost

| Metric | Description |
|--------|-------------|
| **Total Cost** | Sum of all API call costs for a model |
| **Cost per test** | Estimated cost for individual test cases |

---

## Tips

### CI/CD Integration

```bash
# Quiet mode for clean logs
llm-bench run --config bench.config.yaml --quiet --no-color --export json

# Fail fast on errors
llm-bench run --config bench.config.yaml --fail-fast

# Set exit code on failures (default behavior)
llm-bench run --config bench.config.yaml || echo "Benchmark had failures"
```

### Cost Control

```bash
# Set a spending limit
llm-bench run --config bench.config.yaml --max-cost 10.0

# Preview without spending
llm-bench run --config bench.config.yaml --dry-run

# Use free models for testing
llm-bench run -m openrouter/google/gemma-2-9b-it:free
```

### Debugging

```bash
# Verbose output shows debug info
llm-bench run --config bench.config.yaml --verbose

# Validate config first
llm-bench validate --config bench.config.yaml

# Check cache status
llm-bench cache info
```

### Regression Testing

Use the compare command to detect regressions between benchmark runs:

```bash
# Save baseline results
llm-bench run --config bench.config.yaml --export json --output baseline.json

# Later, compare against new run
llm-bench run --config bench.config.yaml --export json --output current.json
llm-bench compare baseline.json current.json
```

### Git Integration

Benchmark results automatically include git metadata when run from a git repository:
- Commit hash (full and short)
- Branch name
- Dirty state indicator (uncommitted changes)
- Tag (if any)

This information appears in:
- Terminal output summary
- HTML export header
- CSV export columns
- JSON export data

---

## Features

### Automatic Retry Logic

LLM-Bench automatically retries failed API calls with exponential backoff:
- **Rate limit errors**: Retried up to 3 times with increasing delays
- **Connection errors**: Retried up to 3 times
- **Server errors (5xx)**: Retried up to 3 times
- **Authentication errors**: Not retried (fail immediately)

The retry delay starts at 1 second and doubles with each attempt, capped at 60 seconds. Random jitter (±10%) is added to prevent thundering herd issues.

### Rate Limiting

Built-in rate limiting prevents API abuse:
- **Default limit**: 100 API calls per 60 seconds
- **Sliding window**: Smoothly spreads requests over time
- **Automatic waiting**: Pauses when limit is reached

This protects against accidentally exhausting API quotas during large benchmark runs.

### Cost Breakdown

After each benchmark run, a detailed cost breakdown table is displayed:
- Cost per model (sorted by highest first)
- Prompt and completion token counts
- Percentage of total cost with visual bar

This helps identify which models are consuming the most budget.

### Helpful Error Messages

When errors occur, LLM-Bench provides:
- Clear error descriptions
- Likely cause identification
- Actionable suggestions for resolution

Example:
```
[AuthenticationError] Authentication failed - check your API key (model: openai/gpt-4)
Suggestion: Set OPENAI_API_KEY environment variable or add it to your .env file
```

### Streaming Support

By default, LLM-Bench uses streaming API calls which enables:
- **TTFT tracking**: Measure time to first token
- **Better progress visibility**: See responses as they arrive
- **Memory efficiency**: Don't buffer entire responses

The TTFT (Time to First Token) metric is particularly useful for measuring perceived latency in interactive applications.

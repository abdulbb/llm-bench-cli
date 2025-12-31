# Development Guide

This guide is for contributors who want to modify or extend `llm-bench`.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abdulbb/llm-bench-cli.git
    cd llm-bench-cli
    ```

2.  **Install in editable mode:**
    We recommend using a virtual environment.
    ```bash
    pip install -e ".[dev]"
    ```
    This installs the package in editable mode along with development dependencies like `pytest`, `pytest-cov`, `mypy`, and `ruff`.

## Project Structure

```
llm-bench-cli/
├── src/llm_bench/           # Source code
│   ├── __init__.py          # Package initialization, version
│   ├── cli.py               # Typer CLI app, commands, provider mappings
│   ├── models.py            # Pydantic models (BenchmarkConfig, TestCase, etc.)
│   ├── runner.py            # Core benchmark execution logic
│   ├── llm.py               # LiteLLM wrapper, retry logic, rate limiting
│   ├── validation.py        # 6-stage validation pipeline
│   ├── cache.py             # DiskCache wrapper with secure permissions
│   ├── config.py            # YAML loading, path security, interpolation
│   ├── data.py              # CSV/JSONL test case loading
│   ├── export.py            # HTML/CSV/JSON report generation
│   ├── compare.py           # Benchmark result comparison
│   ├── output.py            # Rich terminal output formatting
│   ├── metrics.py           # ROUGE-L calculation
│   ├── git_utils.py         # Git repository info extraction
│   ├── init.py              # Interactive setup wizard
│   └── py.typed             # PEP 561 marker for type hints
├── tests/                   # Test suite (pytest)
│   ├── conftest.py          # Shared fixtures
│   ├── test_cli.py          # CLI command tests
│   ├── test_runner.py       # Benchmark runner tests
│   ├── test_validation.py   # Validation pipeline tests
│   ├── test_config.py       # Configuration loading tests
│   ├── test_data_loading.py # CSV/JSONL loading tests
│   ├── test_export.py       # Export format tests
│   ├── test_compare.py      # Comparison tests
│   ├── test_cache.py        # Cache tests
│   ├── test_llm.py          # LLM client tests
│   ├── test_metrics.py      # ROUGE-L tests
│   ├── test_git_utils.py    # Git utils tests
│   ├── test_models.py       # Pydantic model tests
│   ├── test_output.py       # Terminal output tests
│   ├── test_retry.py        # Retry logic tests
│   ├── test_regex.py        # Regex validation tests
│   ├── test_custom_validators.py  # Custom validator tests
│   └── test_features.py     # Integration tests
├── docs/                    # Documentation
├── examples/                # Example configurations
├── pyproject.toml           # Project metadata, dependencies
└── README.md
```

## Code Standards

We enforce high code quality standards using several tools.

### Testing

Run the test suite with `pytest`:
```bash
pytest
```

Run with coverage reporting:
```bash
pytest --cov=src/llm_bench --cov-report=html
```

This generates a coverage report in `htmlcov/`. Ensure all tests pass before submitting a PR.

### Type Checking

We use `mypy` for static type checking.
```bash
mypy src
```
The project is configured with strict type checking enabled.

### Linting & Formatting

We use `ruff` for fast linting and formatting.
```bash
# Check for issues
ruff check src tests

# Auto-fix issues
ruff check --fix src tests

# Format code
ruff format src tests
```

## Architecture Overview

### Validation Pipeline

The validation system (`validation.py`) uses a 6-stage pipeline:

1. **Stage 0**: Regex pattern check (`regex_pattern`)
2. **Stage 0.5**: Custom validator function (`validator`)
3. **Stage 1**: JSON parsing (if `expected` is dict/list or `schema` provided)
4. **Stage 2**: JSON Schema validation (`schema` or `schema_path`)
5. **Stage 3**: Equality check (`expected`)
6. **Stage 4**: LLM judge evaluation (`judge_prompt`)

Each stage can short-circuit and return early with a specific status code.

### LLM Client

The LLM client (`llm.py`) wraps LiteLLM and provides:
- Automatic retry with exponential backoff (3 attempts)
- Rate limiting (100 calls per 60 seconds)
- Streaming support with TTFT tracking
- Token counting and cost calculation

### Configuration Loading

The config loader (`config.py`) handles:
- YAML parsing with environment variable interpolation (`${VAR}`)
- Path traversal protection for file references
- Automatic loading of external test cases from CSV/JSONL

## Security Considerations

When contributing, be aware of these security features:

### Path Traversal Protection

All file paths in configurations are validated to prevent directory traversal attacks:
```python
# In config.py - paths are resolved relative to config directory
# Attempts to use "../" to escape are blocked
```

### Custom Validator Sandboxing

Custom validators run in a restricted environment (`validation.py`):
- **Allowed imports**: `re`, `json`, `math`, `datetime`, `collections`, `itertools`, `functools`, `string`, `typing`, `ast`
- **Blocked operations**: File I/O, network access, subprocess execution, `eval()`, `exec()`, `__import__()`

When adding new allowed imports, consider the security implications carefully.

### Cache Security

The disk cache (`cache.py`) uses restrictive permissions:
- Directory permissions: `0o700` (owner read/write/execute only)
- This prevents other users from reading cached API responses

### Rate Limiting

The rate limiter protects against accidental API abuse:
- Default: 100 calls per 60 seconds
- Implemented using a token bucket algorithm

## Adding a New Feature

1.  Create a branch for your feature.
2.  Implement your changes in `src/llm_bench/`.
3.  Add tests in `tests/` to cover your new functionality.
4.  Run `pytest`, `mypy`, and `ruff` to ensure quality.
5.  Update documentation if needed.
6.  Submit a Pull Request.

## Adding a New Validation Stage

To add a new validation method:

1. Add a new `ValidationStatus` enum value in `models.py`
2. Implement the validation logic in `validation.py`
3. Update the `validate_response()` function to call your stage
4. Add tests in `tests/test_validation.py`
5. Document the new validation in `docs/evaluation.md`

## Adding a New Export Format

To add a new export format:

1. Add a handler function in `export.py` (e.g., `export_xml()`)
2. Update the `export_results()` dispatcher function
3. Add the format to the CLI `--export` option in `cli.py`
4. Add tests in `tests/test_export.py`
5. Document the format in `docs/export.md`

## Adding Provider Support

Provider support is handled by LiteLLM, but you may need to:

1. Add environment variable mappings in `cli.py` if the provider uses non-standard key names
2. Update `docs/providers.md` with setup instructions
3. Add the provider to the models list if it has special handling

## Running the Full CI Check

Before submitting a PR, run the full check locally:
```bash
# Run all checks
ruff check src tests
mypy src
pytest --cov=src/llm_bench

# Or use a single command if you have make
make check
```

# Configuration Guide

`llm-bench` uses a YAML configuration file (default: `bench.config.yaml`) to define benchmarks. This file controls which models are tested, how they are run, and what test cases are executed.

## Structure

The configuration file maps to the `BenchConfig` object.

```yaml
name: "My Benchmark"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4o"
config:
  concurrency: 5
test_cases:
  - input: "Hello"
    expected:
      reply: "Hi there"
```

## Fields

### `name` (Required)
*   **Type:** `str`
*   **Description:** A friendly name for this benchmark suite. Used in reports.

### `system_prompt` (Required)
*   **Type:** `str`
*   **Description:** The system instruction sent to the LLM. This sets the persona or rules for the model.

### `models` (Required)
*   **Type:** `list[str]`
*   **Description:** A list of model identifiers to test. These must follow `litellm` conventions (provider/model-name).
*   **Examples:**
    *   `openai/gpt-4o`
    *   `anthropic/claude-3-5-sonnet-20241022`
    *   `gemini/gemini-1.5-pro`
    *   `ollama/llama3.1` (local model)

### `model_configs` (Optional)
*   **Type:** `dict[str, ModelConfig]`
*   **Description:** Per-model configuration options. Useful for local models or custom endpoints that need a specific `api_base`.

Each model config can have:
*   **`api_base`** (`str` | `null`): Custom API base URL for the model.

**Example - Local LM Studio model:**
```yaml
models:
  - "openai/local-model"

model_configs:
  "openai/local-model":
    api_base: "http://localhost:1234/v1"
```

**Example - Multiple local endpoints:**
```yaml
models:
  - "ollama/llama3.1"
  - "openai/custom-model"

model_configs:
  "openai/custom-model":
    api_base: "http://192.168.1.100:8000/v1"
```

**Note:** Ollama models (`ollama/model-name`) work automatically without needing `model_configs`. LiteLLM detects them and uses the default Ollama endpoint (`http://localhost:11434`).

### `config` (Optional)
Run-time settings for the execution engine.

*   **`concurrency`** (`int`, Default: 5): Number of parallel requests to issue (1-100). Higher values speed up testing but may hit API rate limits.
*   **`temperature`** (`float`, Default: 0.1): Sampling temperature (0.0-2.0). Lower values are more deterministic; higher values are more creative.
*   **`max_cost`** (`float`, Optional): Maximum cost in USD allowed for the session. Execution will stop gracefully if this limit is exceeded.
*   **`judge_model`** (`str`, Default: "gpt-3.5-turbo"): The model used for fuzzy matching when strict equality fails.

### `schema_path` (Optional)
*   **Type:** `str` (Path)
*   **Alias:** Can also be specified as `schema` in YAML.
*   **Description:** Path to a JSON schema file. If provided, the model's output will be validated against this schema.
*   **Security:** Path must be within the config file's directory (no path traversal allowed).

### `validators_file` (Optional)
*   **Type:** `str` (Path)
*   **Description:** Path to a Python file containing custom validator functions.
*   **Security:** Validator code is statically analyzed before execution to block dangerous operations. See [Evaluation & Validation](evaluation.md#custom-validators) for details.

### `test_cases_file` (Optional)
*   **Type:** `str` (Path)
*   **Description:** Path to an external file containing test cases. Supports CSV and JSONL formats.
*   **Note:** Test cases from this file are combined with inline `test_cases`.

#### CSV Format
```csv
input,expected
"What is 2+2?","{""result"": 4}"
"Capital of France?","{""capital"": ""Paris""}"
```

Required columns:
- `input`: The user prompt (required)
- `expected`: JSON string with expected output (optional)

#### JSONL Format
```jsonl
{"input": "What is 2+2?", "expected": {"result": 4}}
{"input": "Capital of France?", "expected": {"capital": "Paris"}}
```

Each line is a JSON object with:
- `input`: The user prompt (required)
- `expected`: Expected output object (optional)

#### Data File Limits
For security and performance:
- Maximum file size: 100 MB
- Maximum test cases per file: 10,000
- Maximum input length: 1 MB per input

### `test_cases` (Required if test_cases_file is missing)
A list of scenarios to test.

*   **`input`** (`str`): The user prompt to send to the model.
*   **`expected`** (`dict` | `null`): Key-value pairs expected in the model's JSON response.
*   **`regex_pattern`** (`str` | `null`): A regex pattern that the raw output must match. Useful for validating text formats or ensuring specific content exists.
    *   Example: `regex_pattern: "^\\d{3}-\\d{2}$"` (Matches 123-45)
    *   Example: `regex_pattern: "def calculate_area\\("` (Ensures function is defined)
*   **`validator`** (`str` | `null`): Name of a custom validator function (defined in `validators_file`) to run on the raw output.
    *   Function signature: `def my_validator(output: str) -> bool | tuple[bool, str]`
    *   Return `True` for pass, `False` for fail, or `(False, "reason")` for fail with message.
*   **`reference`** (`str` | `null`): Reference text for computing NLP metrics (like ROUGE-L) against the model's output.

#### Test Case Validation Order

When multiple validation methods are specified, they run in this order:
1. **Regex Pattern** - If fails, test fails immediately
2. **Custom Validator** - If fails, test fails immediately  
3. **JSON Parse** - Extract and parse JSON from output
4. **Schema Validation** - If `schema_path` is set
5. **Strict Equality** - Compare with `expected`
6. **Fuzzy Match** - LLM judge if strict fails and `judge_model` is set

If only `regex_pattern` or `validator` is specified (no `expected` or `schema`), passing those checks is sufficient for the test to pass.

## CLI Overrides

You can override specific configuration values at runtime using CLI flags:

*   `--model`: Overrides the list of models.
*   `--concurrency`: Overrides the concurrency setting.
*   `--temperature`: Overrides the temperature setting.
*   `--max-cost`: Overrides the max cost setting.
*   `--validators-file`: Overrides the validators file path.

## Caching

To save money and time, `llm-bench` caches model responses by default.

*   **Location:** Responses are stored locally in `~/.cache/llm-bench` (Mac/Linux) or `%USERPROFILE%\.cache\llm-bench` (Windows).
*   **Permissions:** Cache directory is created with secure permissions (700 - owner read/write/execute only).
*   **Keying:** The cache key is a SHA-256 hash of:
    *   Model Name
    *   System Prompt
    *   User Input
    *   Temperature
*   **Invalidation:** Changing any of the above parameters will generate a new cache key, forcing a fresh API call.
*   **Bypassing:** Use the `--no-cache` flag to force fresh calls even if a cached response exists.
*   **Clearing:** Use `llm-bench cache clear` or delete the `.cache/llm-bench` directory.

## Cost Calculation

`llm-bench` estimates the cost of each API call to help you track expenses.

*   **Source:** Costs are calculated using the [LiteLLM](https://docs.litellm.ai/docs/) library, which maintains an up-to-date database of pricing for popular providers (OpenAI, Anthropic, Gemini, etc.).
*   **Method:**
    1.  The tool tracks `prompt_tokens` and `completion_tokens` for each request.
    2.  It queries `litellm` for the cost per token for the specific model.
    3.  Total Cost = (Prompt Tokens × Input Price) + (Completion Tokens × Output Price).
*   **Missing Data:**
    *   If a model's pricing is not known to `litellm`, the cost will report as **$0.00**.
    *   If an API provider does not return token usage stats, `llm-bench` will attempt to estimate the token count locally to provide a rough cost approximation.

## Rate Limiting

`llm-bench` includes built-in rate limiting to prevent API abuse:

*   **Default:** 100 API calls per 60 seconds (sliding window)
*   **Behavior:** If the limit is reached, requests will wait before proceeding
*   **Per-session:** Rate limiting applies across all concurrent requests in a benchmark run

This helps prevent accidental API quota exhaustion and ensures fair usage.

## Security Features

### Path Traversal Protection
All file paths in configuration (`schema_path`, `validators_file`, `test_cases_file`) are validated to ensure they remain within the configuration file's directory. Attempts to use `..` or absolute paths outside the allowed directory will be rejected.

### Validator Code Sandboxing
Custom validator files undergo static analysis before execution to block:
- Dangerous built-ins: `exec`, `eval`, `compile`, `open`, `__import__`
- Unsafe imports: Only safe modules allowed (`re`, `json`, `math`, `string`, `datetime`, `collections`, `itertools`, `functools`, `operator`, `typing`)
- Dangerous attribute access: `__code__`, `__globals__`, `__builtins__`, `__subclasses__`

See [Evaluation & Validation](evaluation.md#custom-validators) for the full security model.

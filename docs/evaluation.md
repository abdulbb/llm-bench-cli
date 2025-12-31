# Evaluation & Validation

`llm-bench` uses a multi-stage validation pipeline to determine if a model's response is correct. This process is designed to be robust, handling common LLM quirks like markdown formatting.

## The Validation Pipeline

When an LLM provides a response, it passes through the following stages in order. Each stage can pass, fail, or be skipped based on your configuration.

### Stage 0: Regex Pattern Check
If a `regex_pattern` is specified in your test case, the raw output is checked first.

*   **How it works:** Uses Python's `re.search()` with `MULTILINE` flag to find a match anywhere in the output.
*   **On failure:** Test fails immediately with status `failed_regex`.
*   **On success:** Continues to next stage, or passes if no other validation is configured.

```yaml
test_cases:
  - input: "Write a Python function to add two numbers"
    regex_pattern: "def add\\("  # Ensures function is defined
```

### Stage 0.5: Custom Validator
If a `validator` function is specified, it runs on the raw output.

*   **How it works:** Calls your custom function with the raw output string.
*   **Return values:**
    - `True` - Validation passed
    - `False` - Validation failed (generic message)
    - `(True, "message")` - Passed with optional info
    - `(False, "reason")` - Failed with specific reason
*   **On failure:** Test fails with status `failed_custom`.
*   **On success:** Continues to next stage, or passes if no JSON validation is configured.

```yaml
validators_file: "validators.py"
test_cases:
  - input: "Generate valid Python code"
    validator: "is_valid_python"
```

### Stage 1: JSON Parsing
The tool tries to parse the response as a JSON object.

*   **Markdown Extraction:** If the LLM wraps the JSON in markdown code blocks (e.g., ` ```json ... ``` `), `llm-bench` will automatically extract the content.
*   **Object Enforcement:** The root of the JSON must be an object `{}` (dictionary), not a list or primitive.
*   **On failure:** Test fails with status `failed_json_parse`.

### Stage 2: Schema Validation (Optional)
If a `schema_path` is provided in your `bench.config.yaml`, the parsed JSON is validated against that schema.

*   It checks for required fields, data types (string, integer, etc.), and structure.
*   This uses Pydantic internally for high-performance validation.
*   **On failure:** Test fails with status `failed_schema`.

### Stage 3: Strict Equality
The tool compares the parsed JSON against the `expected` dictionary in your test case.

*   **Tooling:** Uses `DeepDiff` for comparison.
*   **Flexibility:** It ignores the order of keys in dictionaries and items in lists.
*   **Reporting:** If it fails, it provides a detailed breakdown of what changed, what was added, and what was missing.
*   **On failure:** Continues to Stage 4 if `judge_model` is configured.

### Stage 4: Fuzzy Match (LLM Judge)
If strict equality fails and you have a `judge_model` configured, the tool attempts a "Fuzzy Match".

*   **The Judge:** A second LLM (the "Judge") is given both the `expected` output and the `actual` output.
*   **The Prompt:** The judge is asked: *"Are these two JSON objects semantically equivalent?"*
*   **Result:** If the judge responds "PASS", the test is marked as passed (with a note that it used fuzzy matching).
*   **On failure:** Test fails with status `failed_fuzzy`.
*   **Default Judge:** `gpt-3.5-turbo` by default.

---

## Validation Status Codes

| Status | Description |
|--------|-------------|
| `passed` | Test passed (strict equality or fuzzy match) |
| `failed_regex` | Regex pattern did not match |
| `failed_custom` | Custom validator returned False |
| `failed_json_parse` | Output was not valid JSON |
| `failed_schema` | JSON did not match the schema |
| `failed_equality` | JSON did not match expected (no judge) |
| `failed_fuzzy` | LLM judge also determined mismatch |

---

## Test Case Structure

A typical test case in `bench.config.yaml` looks like this:

```yaml
test_cases:
  - input: "What is the capital of France?"
    expected:
      country: "France"
      capital: "Paris"
```

### How `expected` works:
*   **Exact Matching:** By default, the actual output must exactly match the expected output (same keys, same values).
*   **Extra Fields:** If the model returns extra fields not in `expected`, Stage 3 (Strict Equality) will fail, but Stage 4 (LLM Judge) might pass if semantically equivalent.
*   **Key Names:** Ensure the keys in your `expected` block match exactly what you've asked the model to produce in your `system_prompt`.

### Combining Validation Methods

You can combine multiple validation methods in a single test case:

```yaml
test_cases:
  - input: "Write a function that calculates area"
    regex_pattern: "def.*area"  # Must define a function with 'area'
    validator: "is_valid_python"  # Must be valid Python syntax
    # No 'expected' - passes if regex and validator pass

  - input: "Extract the email from: Contact us at test@example.com"
    regex_pattern: "test@example\\.com"  # Must contain the email
    expected:
      email: "test@example.com"  # Must also return correct JSON
```

## Configuration Options

You can control the judge in the `config` section:

```yaml
config:
  judge_model: "openai/gpt-4o"  # Use a smarter model for judging
```

To disable the judge entirely (requiring strict equality), set `judge_model` to `null`:

```yaml
config:
  judge_model: null  # Strict equality only
```

---

## Custom Validators

Custom validators let you implement complex validation logic beyond regex and JSON matching.

### Creating a Validator File

Create a Python file with your validation functions:

```python
# validators.py
import ast
import re

def is_valid_python(output: str) -> tuple[bool, str]:
    """Check if output contains valid Python syntax."""
    # Extract code from markdown if present
    code = output
    if "```python" in output:
        code = output.split("```python")[1].split("```")[0]
    elif "```" in output:
        code = output.split("```")[1].split("```")[0]
    
    try:
        ast.parse(code)
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def contains_sql_keywords(output: str) -> bool:
    """Check if output contains SQL keywords."""
    keywords = ["SELECT", "FROM", "WHERE", "INSERT", "UPDATE"]
    return any(kw in output.upper() for kw in keywords)

def has_minimum_length(output: str) -> tuple[bool, str]:
    """Ensure output has at least 100 characters."""
    if len(output) >= 100:
        return True, f"Length: {len(output)} chars"
    return False, f"Too short: {len(output)} chars (minimum 100)"
```

### Using Validators

Reference validators in your config:

```yaml
validators_file: "validators.py"

test_cases:
  - input: "Write a Python function to sort a list"
    validator: "is_valid_python"
  
  - input: "Generate a SQL query"
    validator: "contains_sql_keywords"
```

### Validator Security

**Important:** Validator files are executed as Python code. For security, `llm-bench` performs static analysis before execution to block potentially dangerous operations.

#### Allowed Imports
Only these safe modules can be imported:
- `re` - Regular expressions
- `json` - JSON parsing
- `math` - Mathematical functions
- `string` - String constants
- `datetime` - Date/time handling
- `collections` - Data structures
- `itertools` - Iteration tools
- `functools` - Function tools
- `operator` - Operators as functions
- `typing` - Type hints

#### Blocked Operations
The following are blocked and will raise `ValidatorSecurityError`:
- `exec()`, `eval()`, `compile()` - Code execution
- `open()` - File access
- `__import__()` - Dynamic imports
- `input()`, `breakpoint()` - Interactive operations
- `globals()`, `locals()`, `vars()` - Scope inspection
- Access to `__code__`, `__globals__`, `__builtins__`, `__subclasses__`

#### Example: Blocked Code
```python
# This will be rejected:
import os  # Not in allowed list
import subprocess  # Not in allowed list

def bad_validator(output):
    eval(output)  # Blocked
    open("/etc/passwd")  # Blocked
    return True
```

#### Path Restrictions
Validator files must be located within the same directory as your config file (or a subdirectory). Absolute paths or paths containing `..` that escape outside the config directory will be rejected.

---

## NLP Metrics

When you provide a `reference` field in a test case, `llm-bench` calculates text similarity metrics:

### ROUGE-L Score
Measures the longest common subsequence between the model output and reference.

```yaml
test_cases:
  - input: "Summarize this article..."
    reference: "The study found significant improvements in patient outcomes."
```

The ROUGE-L F1 score (0.0 to 1.0) is stored in the results under `metrics.rouge_l`.

---

## Debugging Validation

Use `--verbose` mode to see detailed validation information:

```bash
llm-bench run --config bench.config.yaml --verbose
```

This shows:
- Which validation stages were run
- Exact diff details for failures
- Fuzzy match judge responses
- Regex pattern matches

---

## Freeform Mode (Manual Inspection)

Freeform mode allows you to compare raw model outputs without any automated validation. This is useful when you want to:

- Compare how different models respond to the same prompt
- Evaluate response style, tone, or creativity subjectively
- Explore model capabilities before defining validation criteria
- Conduct qualitative assessments

### How to Enable Freeform Mode

Simply create test cases with only an `input` field - no `expected`, `regex_pattern`, `reference`, or `validator`:

```yaml
name: "Model Comparison"
system_prompt: "You are a helpful assistant."

models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"
  - "gemini/gemini-1.5-pro"

test_cases:
  - input: "What is 2+2?"
  - input: "Explain quantum entanglement in one sentence."
  - input: "Write a haiku about programming."
```

### What Happens in Freeform Mode

When all test cases are freeform:

1. **No Validation**: Tests always pass - the focus is on displaying outputs
2. **Different Output Format**: Terminal shows raw outputs instead of pass/fail tables
3. **Different HTML Report**: The HTML export displays outputs grouped by prompt for easy comparison
4. **Sorted by Latency**: Summary tables are sorted by speed instead of pass rate

### Terminal Output

In freeform mode, the CLI displays outputs like this:

```
╭──────────────────────────────────────────────────────────────────╮
│ Freeform Mode - Showing raw model outputs for manual inspection  │
╰──────────────────────────────────────────────────────────────────╯

Prompt 1: What is 2+2?

┌────────────────────┬─────────────────────────┬──────────┬──────────┐
│ Model              │ Output                  │ Latency  │ Cost     │
├────────────────────┼─────────────────────────┼──────────┼──────────┤
│ openai/gpt-4o      │ The answer is 4.        │ 0.45s    │ $0.0002  │
│ anthropic/claude-3 │ 2+2 equals 4.           │ 0.52s    │ $0.0003  │
│ gemini/gemini-1.5  │ Four.                   │ 0.38s    │ $0.0001  │
└────────────────────┴─────────────────────────┴──────────┴──────────┘
```

### HTML Report in Freeform Mode

Export to HTML to get a shareable report:

```bash
llm-bench run -c freeform.yaml --export html -o comparison.html
```

The freeform HTML report includes:
- A "Freeform Mode" badge indicating no validation
- Summary table sorted by latency (fastest first)
- Speed vs. Price scatter chart
- Cost breakdown
- **Model Outputs by Prompt**: Each prompt displays all model responses side-by-side

### Mixed Mode

If some test cases have validation criteria and others don't, the benchmark runs in **standard mode** (with pass/fail). Only when *all* test cases are freeform does the special freeform output appear.

```yaml
test_cases:
  # This has validation - NOT freeform
  - input: "What is 2+2?"
    expected:
      answer: 4
  
  # This is freeform (input only)
  - input: "Tell me a joke."
```

In this mixed case, standard validation output is used.

### Use Cases

**Comparing Response Styles**
```yaml
test_cases:
  - input: "Explain recursion to a 5-year-old."
  - input: "Explain recursion to a software engineer."
```

**Evaluating Creative Tasks**
```yaml
test_cases:
  - input: "Write a limerick about APIs."
  - input: "Create a metaphor for machine learning."
```

**Exploring Model Capabilities**
```yaml
test_cases:
  - input: "What are your capabilities?"
  - input: "What topics should I avoid asking you about?"
```

### Example Configuration

See `examples/freeform.yaml` for a complete working example:

```yaml
name: "Freeform Model Comparison"
system_prompt: "You are a helpful assistant. Answer concisely."

models:
  - "openrouter/mistralai/devstral-2512:free"
  - "openrouter/openai/gpt-oss-120b:free"

test_cases:
  - input: "What is 2+2?"
  - input: "Explain quantum entanglement in one sentence."
  - input: "Write a haiku about programming."
```

Run with:
```bash
llm-bench run -c examples/freeform.yaml --export html
```

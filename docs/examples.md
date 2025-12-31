# Real-World Examples

This guide provides concrete examples of how to use `llm-bench` for common LLM evaluation tasks.

---

## Scenario 1: Python Code Generation

**Goal:** Evaluate if models can generate syntactically correct Python code that follows a specific structure.

### 1. The Validator Script (`validators.py`)
Create a script to check if the code is valid Python.

```python
import ast

def is_valid_python(output: str) -> tuple[bool, str]:
    """Check if the output contains valid Python syntax."""
    # Extract code from markdown blocks if present
    code = output
    if "```python" in output:
        code = output.split("```python")[1].split("```")[0]
    elif "```" in output:
        code = output.split("```")[1].split("```")[0]
        
    try:
        ast.parse(code)
        return True, "Valid Python"
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
```

### 2. The Configuration (`code_gen.yaml`)
Use `regex_pattern` to ensure it looks like a function and `validator` for syntax checking.

```yaml
name: "Python Code Gen"
system_prompt: "You are a Python coding assistant. Return only the code."
models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"

validators_file: "validators.py"

test_cases:
  - input: "Write a function 'calculate_area' that takes radius as input and returns circle area."
    # Ensure it defines the specific function name
    regex_pattern: "def calculate_area\\("
    # Check syntax
    validator: "is_valid_python"
```

### 3. Run Command
```bash
llm-bench run code_gen.yaml --export html --output code_report.html
```

### 4. Expected Output (Terminal)
```text
Benchmark: Python Code Gen
Models: openai/gpt-4o, anthropic/claude-3-5-sonnet-20241022
Test cases: 1
Concurrency: 5
...

Running ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02

Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Model                     ┃ Pass Rate ┃ P95 Latency ┃ Throughput   ┃ Total Cost ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ openai/gpt-4o             │ 100.0%    │ 1.2s        │ 45.0 tok/s   │ $0.0012    │
│ anthropic/claude-3-5...   │ 100.0%    │ 1.5s        │ 38.0 tok/s   │ $0.0015    │
└───────────────────────────┴───────────┴─────────────┴──────────────┴────────────┘
```

---

## Scenario 2: Summarization Quality (ROUGE Score)

**Goal:** Compare generated summaries against a "Gold Standard" reference summary to measure content overlap using ROUGE-L.

### 1. The Configuration (`summarization.yaml`)

```yaml
name: "News Summarization"
system_prompt: "Summarize the text in one sentence."
models:
  - "gemini/gemini-1.5-pro"
  - "openai/gpt-4o-mini"

test_cases:
  - input: |
      The James Webb Space Telescope (JWST) has captured a lush landscape of stellar birth.
      The new image shows the molecular cloud Rho Ophiuchi, which is the closest star-forming region to Earth.
      It shows jets of gas bursting from young stars and impacting the surrounding gas.
    # We do not expect exact JSON, just text.
    # We provide a reference for ROUGE calculation.
    reference: "JWST captured detailed images of star birth in the Rho Ophiuchi cloud, showing gas jets from young stars."
```

### 2. Run Command
```bash
llm-bench run summarization.yaml --export json --output summary_metrics.json
```

### 3. Expected Output (JSON Snippet)
The CLI will show "Passed", but the JSON/HTML report will contain the metric.

```json
{
  "metrics": {
    "rouge_l": 0.78
  }
}
```

---

## Scenario 3: Data Extraction (JSON Schema)

**Goal:** Ensure the LLM extracts specific fields from unstructured text into strict JSON.

### 1. Create a JSON Schema (`invoice_schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "invoice_number": { "type": "string" },
    "vendor": { "type": "string" },
    "total": { "type": "number" },
    "date": { "type": "string", "format": "date" }
  },
  "required": ["invoice_number", "vendor", "total", "date"]
}
```

### 2. The Configuration (`extraction.yaml`)

```yaml
name: "Invoice Extraction"
system_prompt: "Extract invoice details into JSON. Return only valid JSON."
models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"

test_cases:
  - input: "Invoice #999 from Acme Corp. Total: $500.00 due by 2023-12-31."
    # Validate against schema
    schema: "invoice_schema.json"
    # Also check exact values
    expected:
      invoice_number: "999"
      vendor: "Acme Corp"
      total: 500.00
      date: "2023-12-31"
```

### 3. Run Command
```bash
llm-bench run extraction.yaml
```

### 4. Expected Output (Failure Case)
If the model output is `{ "total": "$500" }` (string instead of number), you will see:

```text
Failures
Model: openai/gpt-4o
Test Case: Invoice #999...
Status: failed_equality
Error: Output does not match expected:
  root['total']: type changed from float to str
  root['total']: expected 500.0, got '$500'
```

---

## Scenario 4: Loading Test Cases from CSV

**Goal:** Run benchmarks with test cases loaded from external data files.

### 1. Create Test Data (`test_data.csv`)

```csv
input,expected,regex_pattern
"What is 2+2?",4,\b4\b
"What is the capital of France?",Paris,\bParis\b
"Convert 100 Celsius to Fahrenheit",212,\b212\b
```

### 2. The Configuration (`math_qa.yaml`)

```yaml
name: "Math QA Benchmark"
system_prompt: "Answer the question concisely."
models:
  - "openai/gpt-4o-mini"
  - "groq/llama-3.1-70b-versatile"

# Load test cases from CSV
test_cases_file: "test_data.csv"
```

### 3. Run Command
```bash
llm-bench run math_qa.yaml --export csv --output results.csv
```

---

## Scenario 5: Loading Test Cases from JSONL

**Goal:** Use JSONL format for complex test cases with nested data.

### 1. Create Test Data (`test_cases.jsonl`)

```jsonl
{"input": "Extract the email from: Contact john@example.com for info", "regex_pattern": "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"}
{"input": "What is the sentiment of: I love this product!", "expected": "positive", "regex_pattern": "positive|negative|neutral"}
{"input": "Summarize: The quick brown fox jumps over the lazy dog.", "reference": "A fox jumps over a dog."}
```

### 2. The Configuration (`nlp_tasks.yaml`)

```yaml
name: "NLP Tasks"
system_prompt: "Complete the task as instructed."
models:
  - "anthropic/claude-3-5-sonnet-20241022"

test_cases_file: "test_cases.jsonl"
```

---

## Scenario 6: Multi-Stage Validation

**Goal:** Combine multiple validation methods for thorough testing.

### 1. The Validator Script (`validators.py`)

```python
import json

def has_required_sections(output: str) -> tuple[bool, str]:
    """Check if output has all required sections."""
    required = ["Introduction", "Methods", "Results", "Conclusion"]
    missing = [s for s in required if s.lower() not in output.lower()]
    if missing:
        return False, f"Missing sections: {', '.join(missing)}"
    return True, "All sections present"

def word_count_check(output: str) -> tuple[bool, str]:
    """Ensure output is between 100-500 words."""
    words = len(output.split())
    if words < 100:
        return False, f"Too short: {words} words (minimum 100)"
    if words > 500:
        return False, f"Too long: {words} words (maximum 500)"
    return True, f"Word count OK: {words} words"
```

### 2. The Configuration (`research_summary.yaml`)

```yaml
name: "Research Summary Generation"
system_prompt: |
  Generate a research summary with these sections:
  - Introduction
  - Methods  
  - Results
  - Conclusion
  Keep it between 100-500 words.

models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"
  - "gemini/gemini-1.5-pro"

validators_file: "validators.py"

test_cases:
  - input: "Summarize a study on the effects of caffeine on productivity."
    # Stage 0: Must mention "caffeine" and "productivity"
    regex_pattern: "(?i)caffeine.*productivity|productivity.*caffeine"
    # Stage 0.5: Check structure and length
    validator: "has_required_sections"
```

### 3. Validation Order
When this runs, validation happens in this order:
1. **Regex Check**: Must match `caffeine` and `productivity`
2. **Custom Validator**: Must have all 4 required sections

If regex fails → `failed_regex`  
If validator fails → `failed_custom`  
If both pass → `passed`

---

## Scenario 7: Using an LLM Judge

**Goal:** Use another LLM to evaluate response quality when exact matching isn't possible.

### 1. The Configuration (`creative_writing.yaml`)

```yaml
name: "Creative Writing Evaluation"
system_prompt: "Write a short story based on the prompt."
models:
  - "openai/gpt-4o"
  - "anthropic/claude-3-5-sonnet-20241022"

# Use GPT-4o as the judge
judge_model: "openai/gpt-4o"

test_cases:
  - input: "Write a 3-paragraph story about a robot learning to paint."
    # Custom judge prompt
    judge_prompt: |
      Evaluate this creative writing response:
      
      1. Does it have exactly 3 paragraphs?
      2. Is it about a robot?
      3. Does the robot learn to paint?
      4. Is the writing engaging and creative?
      
      Respond with PASS if all criteria are met, FAIL otherwise.
      Include a brief explanation.
```

---

## Scenario 8: Comparing Benchmark Results

**Goal:** Compare results from two different benchmark runs.

### 1. Run Benchmarks and Export JSON

```bash
# Run with GPT-4o
llm-bench run config.yaml --model openai/gpt-4o --export json --output run1.json

# Run with Claude
llm-bench run config.yaml --model anthropic/claude-3-5-sonnet-20241022 --export json --output run2.json
```

### 2. Compare Results

```bash
llm-bench compare run1.json run2.json --export html --output comparison.html
```

### 3. Expected Output
The comparison shows:
- Pass rate differences
- Latency improvements/regressions
- Cost differences
- Per-test-case breakdowns

---

## Scenario 9: CI/CD Integration

**Goal:** Run benchmarks in a CI/CD pipeline with failure thresholds.

### 1. The Configuration (`ci_benchmark.yaml`)

```yaml
name: "CI Quality Gate"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4o-mini"  # Use cheaper model for CI

test_cases:
  - input: "What is 2+2?"
    expected: 4
  - input: "Is the sky blue?"
    regex_pattern: "(?i)yes|blue"
```

### 2. CI Script (GitHub Actions)

```yaml
- name: Run LLM Benchmark
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  run: |
    llm-bench run ci_benchmark.yaml \
      --quiet \
      --no-color \
      --max-cost 0.10 \
      --fail-fast \
      --export json \
      --output results.json
```

### 3. Key Flags for CI
- `--quiet`: Minimal output
- `--no-color`: No ANSI colors (for logs)
- `--max-cost 0.10`: Stop if cost exceeds $0.10
- `--fail-fast`: Stop on first failure
- Exit code: `0` if all pass, `1` if any fail

---

## Scenario 10: Using Free Models via OpenRouter

**Goal:** Run benchmarks using free models to test without API costs.

### 1. The Configuration (`free_models.yaml`)

```yaml
name: "Free Model Testing"
system_prompt: "Answer questions accurately."
models:
  # Free models on OpenRouter
  - "openrouter/google/gemma-2-9b-it:free"
  - "openrouter/meta-llama/llama-3.2-3b-instruct:free"
  - "openrouter/qwen/qwen-2-7b-instruct:free"

test_cases:
  - input: "What is the capital of Japan?"
    expected: "Tokyo"
    regex_pattern: "(?i)tokyo"
```

### 2. Setup
```bash
export OPENROUTER_API_KEY="sk-or-..."
llm-bench run free_models.yaml
```

---

## Tips and Best Practices

### 1. Start Simple
Begin with `regex_pattern` for basic checks, then add complexity:
```yaml
test_cases:
  - input: "Generate a UUID"
    regex_pattern: "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
```

### 2. Use Dry Run First
Always validate your config before spending API credits:
```bash
llm-bench run config.yaml --dry-run
```

### 3. Cache for Iteration
The cache saves responses, so repeated runs are free:
```bash
llm-bench run config.yaml          # First run: API calls
llm-bench run config.yaml          # Second run: cached (free)
llm-bench run config.yaml --no-cache  # Force fresh calls
```

### 4. Export for Analysis
Use JSON export for programmatic analysis:
```bash
llm-bench run config.yaml --export json --output results.json
python analyze.py results.json
```

### 5. Set Cost Limits
Protect against runaway costs:
```bash
llm-bench run expensive_config.yaml --max-cost 5.0
```

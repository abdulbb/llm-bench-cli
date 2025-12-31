# Exporting Results

`llm-bench` allows you to export your benchmark results in multiple formats for sharing, analysis, or visualization.

## CLI Usage

Use the `--export` flag to specify the format and `--output` (optional) for the filename.

```bash
llm-bench run --export html --output my-report.html
llm-bench run --export csv --output data.csv
llm-bench run --export json --output full-results.json
```

---

## Formats

### 1. HTML Report (`--export html`)
**Best for:** Visual inspection and sharing with the team.

*   **Interactive Tables:** Filter, sort, and search through detailed results using DataTables. Quickly find failures matching specific keywords (e.g., "SQL").
*   **Summary Table:** Compare models by Pass Rate, P95 Latency, Throughput, and Cost.
*   **Charts:**
    - **Latency vs. Accuracy:** Scatter plot showing latency and pass/fail for each test
    - **Speed vs. Price:** Compare throughput against cost across models
    - **Cost Breakdown:** Doughnut chart showing cost distribution by model
*   **Failure Analysis:** Detailed breakdown of failed tests with formatted JSON diffs.
*   **Git Integration:** Shows commit hash, branch, and dirty state in the header.
*   **Self-Contained:** The output is a single HTML file with embedded CSS and JS (Chart.js and DataTables via CDN).

### 2. CSV (`--export csv`)
**Best for:** Data analysis in Excel, Pandas, or Google Sheets.

The CSV is "flattened" so that each row represents a single test run for a single model.

**Columns:**
*   `model_name`: The model identifier (e.g., `openai/gpt-4o`).
*   `test_index`: The 1-based index of the test case in your config.
*   `passed`: `True` or `False`.
*   `status`: The detailed result status (e.g., `passed`, `failed_json_parse`, `failed_equality`, `failed_fuzzy`, `failed_regex`, `failed_custom`).
*   `latency_total`: Total request time in seconds.
*   `latency_ttft`: Time to First Token in seconds (if available from streaming).
*   `prompt_tokens`, `completion_tokens`, `total_tokens`: Usage stats.
*   `cost_usd`: Estimated cost for this specific call.
*   `input`: The user prompt used.
*   `expected`: The expected JSON (stringified).
*   `actual_output`: The actual JSON returned (stringified).
*   `error_message`: Details if the test failed.
*   `run_timestamp`: ISO timestamp of when the benchmark was run.
*   `git_commit`: Short git commit hash (if in a git repo).
*   `git_branch`: Git branch name (if in a git repo).

### 3. JSON (`--export json`)
**Best for:** Programmatic processing, archiving, or use with the `compare` command.

This dumps the full internal state of the benchmark run. It follows the `BenchmarkRun` Pydantic model structure.

**Structure:**
```json
{
  "config": {
    "name": "My Benchmark",
    "system_prompt": "...",
    "models": ["openai/gpt-4o"],
    "config": {
      "concurrency": 5,
      "temperature": 0.1,
      "judge_model": "gpt-3.5-turbo",
      "max_cost": null
    },
    "test_cases": [...]
  },
  "model_results": [
    {
      "model_name": "openai/gpt-4o",
      "test_results": [
        {
          "test_case": {...},
          "passed": true,
          "status": "passed",
          "actual_output": {...},
          "raw_output": "...",
          "latency": {
            "total_seconds": 1.234,
            "time_to_first_token_seconds": 0.156
          },
          "token_usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100
          },
          "cost_usd": 0.0015,
          "metrics": {
            "rouge_l": 0.85
          },
          "error_message": null,
          "used_fuzzy_match": false,
          "is_cached": false
        }
      ]
    }
  ],
  "git_info": {
    "commit_hash": "abc123def456...",
    "commit_short": "abc123d",
    "branch": "main",
    "is_dirty": false,
    "tag": "v1.0.0"
  },
  "run_timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

## Git Information

When running from a git repository, all export formats include git metadata:

| Field | Description |
|-------|-------------|
| `commit_hash` | Full 40-character commit SHA |
| `commit_short` | Short 7-character commit SHA |
| `branch` | Current branch name |
| `is_dirty` | `true` if there are uncommitted changes |
| `tag` | Git tag if the current commit is tagged |

This helps track which code version produced which results, enabling proper regression testing.

---

## Cost Breakdown

All formats include detailed cost information:

### Per-Model Metrics
- Total cost for all tests
- Prompt token count
- Completion token count
- Percentage of total cost

### Per-Test Metrics
- Individual test cost
- Token counts per test

In the terminal and HTML export, you'll see a cost breakdown table:

```
Cost Breakdown
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Model              ┃ Tests ┃ Prompt Tokens  ┃ Completion Tokens┃ Cost     ┃ % of Total    ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ openai/gpt-4o      │ 10    │ 500            │ 1,200            │ $0.0340  │ 68.0% ======  │
│ openai/gpt-4o-mini │ 10    │ 500            │ 1,150            │ $0.0160  │ 32.0% ===     │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ TOTAL              │ 20    │ 1,000          │ 2,350            │ $0.0500  │ 100%          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison Workflow

Export to JSON to enable the `compare` command:

```bash
# Run baseline
llm-bench run --config bench.config.yaml --export json --output baseline.json

# Make changes to prompts/config...

# Run again
llm-bench run --config bench.config.yaml --export json --output new.json

# Compare results
llm-bench compare baseline.json new.json
```

See [CLI Reference - compare](cli.md#llm-bench-compare) for details on the comparison output.

---

## Best Practices

### For Archival
Use JSON export to preserve the complete benchmark state:
```bash
llm-bench run --config bench.config.yaml --export json \
  --output "results/benchmark-$(date +%Y%m%d-%H%M%S).json"
```

### For Spreadsheet Analysis
Use CSV for easy import into analysis tools:
```bash
llm-bench run --config bench.config.yaml --export csv --output results.csv
# Then open in Excel, Google Sheets, or use with pandas
```

### For Team Sharing
Use HTML for self-contained reports:
```bash
llm-bench run --config bench.config.yaml --export html --output report.html
# Share the single HTML file - no server needed
```

### For CI/CD
Combine JSON export with quiet mode:
```bash
llm-bench run --config bench.config.yaml --quiet --no-color \
  --export json --output results.json
# Parse results.json in your CI pipeline
```

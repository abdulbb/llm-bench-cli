"""Export functionality for LLM-Bench results."""

import csv
import html
import json
from pathlib import Path
from typing import Any

from llm_bench.models import BenchmarkRun


def _safe_json_for_script(data: Any) -> str:
    """Serialize data to JSON safe for embedding in HTML script tags.

    Escapes sequences that could break out of script context:
    - </script> becomes <\\/script>
    - <!-- becomes <\\!--

    Args:
        data: Data to serialize.

    Returns:
        JSON string safe for embedding in script tags.
    """
    json_str = json.dumps(data)
    # Escape sequences that could break script context
    json_str = json_str.replace("</script>", r"<\/script>")
    json_str = json_str.replace("</Script>", r"<\/Script>")
    json_str = json_str.replace("</SCRIPT>", r"<\/SCRIPT>")
    json_str = json_str.replace("<!--", r"<\!--")
    return json_str


def export_to_json(results: BenchmarkRun, output_path: Path) -> None:
    """Export benchmark results to a JSON file.

    Args:
        results: The benchmark run results.
        output_path: Path to the output JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(results.model_dump_json(indent=2))


def export_to_csv(results: BenchmarkRun, output_path: Path) -> None:
    """Export benchmark results to a CSV file.

    Flattened format: one row per test case per model.
    Includes metadata columns for git info and run timestamp.

    Args:
        results: The benchmark run results.
        output_path: Path to the output CSV file.
    """
    fieldnames = [
        "model_name",
        "test_index",
        "passed",
        "status",
        "latency_total",
        "latency_ttft",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
        "input",
        "expected",
        "actual_output",
        "error_message",
        "run_timestamp",
        "git_commit",
        "git_branch",
    ]

    # Extract git info
    git_commit = results.git_info.commit_short if results.git_info else None
    git_branch = results.git_info.branch if results.git_info else None
    run_timestamp = results.run_timestamp

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_result in results.model_results:
            for i, test_result in enumerate(model_result.test_results):
                row = {
                    "model_name": model_result.model_name,
                    "test_index": i + 1,
                    "passed": test_result.passed,
                    "status": test_result.status.value,
                    "latency_total": test_result.latency.total_seconds,
                    "latency_ttft": test_result.latency.time_to_first_token_seconds,
                    "prompt_tokens": test_result.token_usage.prompt_tokens,
                    "completion_tokens": test_result.token_usage.completion_tokens,
                    "total_tokens": test_result.token_usage.total_tokens,
                    "cost_usd": test_result.cost_usd,
                    "input": test_result.test_case.input,
                    "expected": json.dumps(test_result.test_case.expected),
                    "actual_output": json.dumps(test_result.actual_output)
                    if test_result.actual_output
                    else None,
                    "error_message": test_result.error_message,
                    "run_timestamp": run_timestamp,
                    "git_commit": git_commit,
                    "git_branch": git_branch,
                }
                writer.writerow(row)


def export_to_html(results: BenchmarkRun, output_path: Path) -> None:
    """Export benchmark results to a self-contained HTML report.

    Automatically detects freeform mode (no validation criteria) and
    generates an appropriate report focused on raw outputs instead of
    pass/fail metrics.

    Args:
        results: The benchmark run results.
        output_path: Path to the output HTML file.
    """
    # Check if this is a freeform benchmark
    is_freeform = results.config.is_freeform

    if is_freeform:
        _export_freeform_html(results, output_path)
    else:
        _export_standard_html(results, output_path)


def _export_freeform_html(results: BenchmarkRun, output_path: Path) -> None:
    """Export freeform benchmark results to HTML.

    Shows raw model outputs for manual inspection without pass/fail metrics.

    Args:
        results: The benchmark run results.
        output_path: Path to the output HTML file.
    """
    import datetime

    num_models = len(results.model_results)

    def generate_distinct_colors(n: int) -> list[str]:
        """Generate n visually distinct colors using HSV color space."""
        import colorsys

        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7
            value = 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1)")
        return colors

    colors = generate_distinct_colors(num_models)

    # Speed vs Price chart data
    speed_vs_price_datasets = []
    for idx, model_result in enumerate(results.model_results):
        speed_vs_price_datasets.append(
            {
                "label": model_result.model_name,
                "data": [
                    {
                        "x": model_result.total_cost_usd,
                        "y": model_result.throughput_tokens_per_second,
                    }
                ],
                "backgroundColor": colors[idx],
                "pointRadius": 8,
            }
        )

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM-Bench Report - {benchmark_name} (Freeform)</title>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- DataTables -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>

    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1600px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; vertical-align: top; }}
        th {{ background-color: #f1f3f5; font-weight: 600; }}
        .freeform-badge {{ background: #17a2b8; color: white; padding: 4px 12px; border-radius: 4px; font-size: 14px; margin-left: 10px; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; }}
        .code-block {{ background: #fdf6e3; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; font-size: 13px; border: 1px solid #eee; max-height: 300px; overflow-y: auto; display: block; }}
        .prompt-header {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #2196f3; }}
        .prompt-text {{ font-family: monospace; white-space: pre-wrap; margin: 0; }}
        .model-output {{ margin-bottom: 20px; padding: 15px; border: 1px solid #e9ecef; border-radius: 8px; }}
        .model-name {{ font-weight: bold; color: #3498db; margin-bottom: 10px; }}
        .output-meta {{ font-size: 12px; color: #6c757d; margin-top: 10px; }}
    </style>
</head>
<body>
    <h1>LLM-Bench: {benchmark_name} <span class="freeform-badge">Freeform Mode</span></h1>
    <p>Run Date: {run_date}{git_info_html}</p>
    <p><em>Freeform mode: Displaying raw model outputs for manual inspection. No pass/fail validation.</em></p>

    <div class="card">
        <h2>Summary</h2>
        <table id="summaryTable" class="display" style="width:100%">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Tests</th>
                    <th>P95 Latency</th>
                    <th>Throughput</th>
                    <th>Total Cost</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Output Speed vs. Price</h2>
        <div class="chart-container">
            <canvas id="speedPriceChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Cost Breakdown</h2>
        <div class="chart-container" style="height: 300px;">
            <canvas id="costPieChart"></canvas>
        </div>
        <table id="costTable" class="display" style="width:100%; margin-top: 20px;">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Tests</th>
                    <th>Prompt Tokens</th>
                    <th>Completion Tokens</th>
                    <th>Cost</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
                {cost_breakdown_rows}
            </tbody>
            <tfoot>
                <tr style="font-weight: bold; background-color: #f1f3f5;">
                    <td>TOTAL</td>
                    <td>{total_tests}</td>
                    <td>{total_prompt_tokens}</td>
                    <td>{total_completion_tokens}</td>
                    <td>{total_cost}</td>
                    <td>100%</td>
                </tr>
            </tfoot>
        </table>
    </div>

    <div class="card">
        <h2>Model Outputs by Prompt</h2>
        {prompt_outputs}
    </div>

    <script>
        $(document).ready(function() {{
            $('#summaryTable').DataTable({{
                pageLength: 10,
                order: [[2, 'asc']], // Sort by P95 Latency ascending (fastest first)
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]]
            }});
        }});

        const ctxSpeedPrice = document.getElementById('speedPriceChart').getContext('2d');
        new Chart(ctxSpeedPrice, {{
            type: 'scatter',
            data: {{
                datasets: {speed_price_datasets_json}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Total Cost (USD)' }},
                        beginAtZero: true
                    }},
                    y: {{
                        title: {{ display: true, text: 'Throughput (tokens/sec)' }},
                        beginAtZero: true
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': $' + context.parsed.x.toFixed(4) + ', ' + context.parsed.y.toFixed(1) + ' tok/s';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Cost Pie Chart
        const ctxCost = document.getElementById('costPieChart').getContext('2d');
        new Chart(ctxCost, {{
            type: 'doughnut',
            data: {{
                labels: {cost_labels_json},
                datasets: [{{
                    data: {cost_values_json},
                    backgroundColor: {cost_colors_json}
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const value = context.parsed;
                                const pct = ((value / total) * 100).toFixed(1);
                                return context.label + ': $' + value.toFixed(4) + ' (' + pct + '%)';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // DataTables for cost breakdown
        $('#costTable').DataTable({{
            pageLength: 10,
            order: [[4, 'desc']], // Sort by cost descending
            lengthMenu: [[10, 25, -1], [10, 25, "All"]],
            paging: false,
            searching: false,
            info: false
        }});
    </script>
</body>
</html>
"""

    # Use stored timestamp if available, otherwise current time
    if results.run_timestamp:
        run_date = results.run_timestamp
    else:
        run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format git info if available
    git_info_html = ""
    if results.git_info and results.git_info.commit_short:
        git_info_html = (
            f" | Git: <code>{html.escape(results.git_info.summary())}</code>"
        )

    # Summary rows (without pass rate)
    summary_rows = ""
    for mr in sorted(results.model_results, key=lambda x: x.p95_latency_seconds):
        summary_rows += f"""
                <tr>
                    <td><strong>{html.escape(mr.model_name)}</strong></td>
                    <td>{mr.total_tests}</td>
                    <td>{mr.p95_latency_seconds:.3f}s</td>
                    <td>{mr.throughput_tokens_per_second:.1f} tok/s</td>
                    <td>${mr.total_cost_usd:.4f}</td>
                </tr>"""

    # Generate prompt outputs section
    prompt_outputs = ""
    test_inputs = [tc.input for tc in results.config.test_cases]

    for test_idx, test_input in enumerate(test_inputs):
        # Escape and potentially truncate prompt for display
        display_input = html.escape(test_input)

        prompt_outputs += f"""
        <div class="prompt-header">
            <strong>Prompt {test_idx + 1}:</strong>
            <pre class="prompt-text">{display_input}</pre>
        </div>
        """

        for model_result in results.model_results:
            # Find the test result for this prompt
            test_result = None
            for tr in model_result.test_results:
                if tr.test_case.input == test_input:
                    test_result = tr
                    break

            if test_result is None:
                continue

            raw_output = html.escape(test_result.raw_output or "<no output>")

            prompt_outputs += f"""
        <div class="model-output">
            <div class="model-name">{html.escape(model_result.model_name)}</div>
            <div class="code-block">{raw_output}</div>
            <div class="output-meta">
                Latency: {test_result.latency.total_seconds:.3f}s |
                Tokens: {test_result.token_usage.total_tokens} |
                Cost: ${test_result.cost_usd:.5f}
            </div>
        </div>
            """

    # Generate cost breakdown data
    total_cost = results.total_cost_usd
    cost_breakdown_rows = ""
    cost_labels = []
    cost_values = []

    sorted_models = sorted(
        results.model_results, key=lambda m: m.total_cost_usd, reverse=True
    )

    for mr in sorted_models:
        prompt_tokens = sum(r.token_usage.prompt_tokens for r in mr.test_results)
        completion_tokens = sum(
            r.token_usage.completion_tokens for r in mr.test_results
        )
        cost_pct = (mr.total_cost_usd / total_cost * 100) if total_cost > 0 else 0

        cost_labels.append(mr.model_name)
        cost_values.append(mr.total_cost_usd)

        cost_breakdown_rows += f"""
                <tr>
                    <td><strong>{html.escape(mr.model_name)}</strong></td>
                    <td>{mr.total_tests}</td>
                    <td>{prompt_tokens:,}</td>
                    <td>{completion_tokens:,}</td>
                    <td>${mr.total_cost_usd:.4f}</td>
                    <td>{cost_pct:.1f}%</td>
                </tr>"""

    # Calculate totals
    total_tests = sum(m.total_tests for m in results.model_results)
    total_prompt_tokens = sum(
        sum(r.token_usage.prompt_tokens for r in m.test_results)
        for m in results.model_results
    )
    total_completion_tokens = sum(
        sum(r.token_usage.completion_tokens for r in m.test_results)
        for m in results.model_results
    )

    content = html_template.format(
        benchmark_name=html.escape(results.config.name),
        run_date=run_date,
        git_info_html=git_info_html,
        summary_rows=summary_rows,
        speed_price_datasets_json=_safe_json_for_script(speed_vs_price_datasets),
        prompt_outputs=prompt_outputs,
        cost_breakdown_rows=cost_breakdown_rows,
        cost_labels_json=_safe_json_for_script(cost_labels),
        cost_values_json=_safe_json_for_script(cost_values),
        cost_colors_json=_safe_json_for_script(colors),
        total_tests=total_tests,
        total_prompt_tokens=f"{total_prompt_tokens:,}",
        total_completion_tokens=f"{total_completion_tokens:,}",
        total_cost=f"${total_cost:.4f}",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def _export_standard_html(results: BenchmarkRun, output_path: Path) -> None:
    """Export standard benchmark results to HTML with pass/fail metrics.

    Args:
        results: The benchmark run results.
        output_path: Path to the output HTML file.
    """
    # Prepare data for Chart.js
    num_models = len(results.model_results)

    def generate_distinct_colors(n: int) -> list[str]:
        """Generate n visually distinct colors using HSV color space."""
        import colorsys

        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7
            value = 0.8
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1)")
        return colors

    colors = generate_distinct_colors(num_models)

    datasets = []
    for idx, model_result in enumerate(results.model_results):
        data_points = []
        for test_result in model_result.test_results:
            data_points.append(
                {
                    "x": test_result.latency.total_seconds,
                    "y": 100 if test_result.passed else 0,
                }
            )

        datasets.append(
            {
                "label": model_result.model_name,
                "data": data_points,
                "backgroundColor": colors[idx],
            }
        )

    speed_vs_price_datasets = []
    for idx, model_result in enumerate(results.model_results):
        speed_vs_price_datasets.append(
            {
                "label": model_result.model_name,
                "data": [
                    {
                        "x": model_result.total_cost_usd,
                        "y": model_result.throughput_tokens_per_second,
                    }
                ],
                "backgroundColor": colors[idx],
                "pointRadius": 8,
            }
        )

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM-Bench Report - {benchmark_name}</title>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- DataTables -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>

    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1600px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; vertical-align: top; }}
        th {{ background-color: #f1f3f5; font-weight: 600; }}
        .status-passed {{ color: #2ecc71; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; }}
        .code-block {{ background: #fdf6e3; padding: 5px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; font-size: 12px; border: 1px solid #eee; max-height: 150px; overflow-y: auto; display: block; }}
        .model-header {{ border-left: 5px solid #3498db; padding-left: 15px; margin-top: 40px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-label {{ font-size: 14px; color: #6c757d; text-transform: uppercase; }}
        /* DataTables overrides */
        .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter {{ margin-bottom: 15px; }}
        table.dataTable tbody tr td {{ padding: 8px 10px; }}
    </style>
</head>
<body>
    <h1>LLM-Bench: {benchmark_name}</h1>
    <p>Run Date: {run_date}{git_info_html}</p>

    <div class="card">
        <h2>Summary</h2>
        <table id="summaryTable" class="display" style="width:100%">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Pass Rate</th>
                    <th>P95 Latency</th>
                    <th>Throughput</th>
                    <th>Total Cost</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Latency vs. Accuracy</h2>
        <div class="chart-container">
            <canvas id="scatterChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Output Speed vs. Price</h2>
        <div class="chart-container">
            <canvas id="speedPriceChart"></canvas>
        </div>
    </div>

    <div class="card">
        <h2>Cost Breakdown</h2>
        <div class="chart-container" style="height: 300px;">
            <canvas id="costPieChart"></canvas>
        </div>
        <table id="costTable" class="display" style="width:100%; margin-top: 20px;">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Tests</th>
                    <th>Prompt Tokens</th>
                    <th>Completion Tokens</th>
                    <th>Cost</th>
                    <th>% of Total</th>
                </tr>
            </thead>
            <tbody>
                {cost_breakdown_rows}
            </tbody>
            <tfoot>
                <tr style="font-weight: bold; background-color: #f1f3f5;">
                    <td>TOTAL</td>
                    <td>{total_tests}</td>
                    <td>{total_prompt_tokens}</td>
                    <td>{total_completion_tokens}</td>
                    <td>{total_cost}</td>
                    <td>100%</td>
                </tr>
            </tfoot>
        </table>
    </div>

    <div class="card">
        <h2>Detailed Results</h2>
        <p>Use the search box to filter by input, error message, or model.</p>
        <table id="resultsTable" class="display" style="width:100%">
            <thead>
                <tr>
                    <th style="width: 10%;">Model</th>
                    <th style="width: 8%;">Status</th>
                    <th style="width: 25%;">Input</th>
                    <th style="width: 25%;">Actual Output</th>
                    <th style="width: 20%;">Error / Expected</th>
                    <th style="width: 6%;">Latency</th>
                    <th style="width: 6%;">Cost</th>
                </tr>
            </thead>
            <tbody>
                {detailed_rows}
            </tbody>
        </table>
    </div>

    <script>
        $(document).ready(function() {{
            $('#summaryTable').DataTable({{
                pageLength: 10,
                order: [[1, 'desc']], // Sort by Pass Rate descending
                lengthMenu: [[10, 25, 50, -1], [10, 25, 50, "All"]]
            }});
            $('#resultsTable').DataTable({{
                pageLength: 25,
                order: [[1, 'asc']], // Sort by Status
                lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                scrollX: true
            }});
        }});

        const ctx = document.getElementById('scatterChart').getContext('2d');
        new Chart(ctx, {{
            type: 'scatter',
            data: {{
                datasets: {datasets_json}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Latency (seconds)' }},
                        beginAtZero: true
                    }},
                    y: {{
                        title: {{ display: true, text: 'Passed (100=Yes, 0=No)' }},
                        min: -10,
                        max: 110,
                        ticks: {{
                            stepSize: 100,
                            callback: function(value) {{
                                if (value === 100) return 'Pass';
                                if (value === 0) return 'Fail';
                                return '';
                            }}
                        }}
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': ' + context.parsed.x.toFixed(3) + 's (' + (context.parsed.y === 100 ? 'Pass' : 'Fail') + ')';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        const ctxSpeedPrice = document.getElementById('speedPriceChart').getContext('2d');
        new Chart(ctxSpeedPrice, {{
            type: 'scatter',
            data: {{
                datasets: {speed_price_datasets_json}
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        title: {{ display: true, text: 'Total Cost (USD)' }},
                        beginAtZero: true
                    }},
                    y: {{
                        title: {{ display: true, text: 'Throughput (tokens/sec)' }},
                        beginAtZero: true
                    }}
                }},
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                return context.dataset.label + ': $' + context.parsed.x.toFixed(4) + ', ' + context.parsed.y.toFixed(1) + ' tok/s';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Cost Pie Chart
        const ctxCost = document.getElementById('costPieChart').getContext('2d');
        new Chart(ctxCost, {{
            type: 'doughnut',
            data: {{
                labels: {cost_labels_json},
                datasets: [{{
                    data: {cost_values_json},
                    backgroundColor: {cost_colors_json}
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'right'
                    }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const value = context.parsed;
                                const pct = ((value / total) * 100).toFixed(1);
                                return context.label + ': $' + value.toFixed(4) + ' (' + pct + '%)';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // DataTables for cost breakdown
        $('#costTable').DataTable({{
            pageLength: 10,
            order: [[4, 'desc']], // Sort by cost descending
            lengthMenu: [[10, 25, -1], [10, 25, "All"]],
            paging: false,
            searching: false,
            info: false
        }});
    </script>
</body>
</html>
"""

    import datetime

    # Use stored timestamp if available, otherwise current time
    if results.run_timestamp:
        run_date = results.run_timestamp
    else:
        run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format git info if available
    git_info_html = ""
    if results.git_info and results.git_info.commit_short:
        git_info_html = (
            f" | Git: <code>{html.escape(results.git_info.summary())}</code>"
        )

    summary_rows = ""
    for mr in sorted(results.model_results, key=lambda x: x.pass_rate, reverse=True):
        pass_rate_style = (
            "status-passed"
            if mr.pass_rate >= 90
            else ("status-failed" if mr.pass_rate < 70 else "")
        )
        summary_rows += f"""
                <tr>
                    <td><strong>{mr.model_name}</strong></td>
                    <td class="{pass_rate_style}">{mr.pass_rate:.1f}%</td>
                    <td>{mr.p95_latency_seconds:.3f}s</td>
                    <td>{mr.throughput_tokens_per_second:.1f} tok/s</td>
                    <td>${mr.total_cost_usd:.4f}</td>
                </tr>"""

    detailed_rows = ""
    for mr in results.model_results:
        for res in mr.test_results:
            status_class = "status-passed" if res.passed else "status-failed"

            # Format JSONs for display and escape HTML to prevent XSS
            input_text = html.escape(res.test_case.input)
            if len(input_text) > 500:
                input_text = input_text[:500] + "..."

            # Escape JSON output to prevent XSS
            actual_json = (
                html.escape(json.dumps(res.actual_output, indent=2))
                if res.actual_output
                else html.escape(res.raw_output or "")
            )
            expected_json = (
                html.escape(json.dumps(res.test_case.expected, indent=2))
                if res.test_case.expected
                else "N/A"
            )

            # Escape error messages to prevent XSS
            error_content = ""
            if res.error_message:
                escaped_error = html.escape(res.error_message)
                error_content += f"<strong style='color:red'>Error:</strong><div class='code-block'>{escaped_error}</div>"
            if res.test_case.expected:
                error_content += f"<strong>Expected:</strong><div class='code-block'>{expected_json}</div>"

            detailed_rows += f"""
                <tr>
                    <td>{mr.model_name}</td>
                    <td class="{status_class}">{res.status.value}</td>
                    <td><div class="code-block">{input_text}</div></td>
                    <td><div class="code-block">{actual_json}</div></td>
                    <td>{error_content}</td>
                    <td>{res.latency.total_seconds:.3f}s</td>
                    <td>${res.cost_usd:.5f}</td>
                </tr>
            """

    # Generate cost breakdown data
    total_cost = results.total_cost_usd
    cost_breakdown_rows = ""
    cost_labels = []
    cost_values = []

    # Sort by cost descending
    sorted_models = sorted(
        results.model_results, key=lambda m: m.total_cost_usd, reverse=True
    )

    for mr in sorted_models:
        prompt_tokens = sum(r.token_usage.prompt_tokens for r in mr.test_results)
        completion_tokens = sum(
            r.token_usage.completion_tokens for r in mr.test_results
        )
        cost_pct = (mr.total_cost_usd / total_cost * 100) if total_cost > 0 else 0

        cost_labels.append(mr.model_name)
        cost_values.append(mr.total_cost_usd)

        cost_breakdown_rows += f"""
                <tr>
                    <td><strong>{html.escape(mr.model_name)}</strong></td>
                    <td>{mr.total_tests}</td>
                    <td>{prompt_tokens:,}</td>
                    <td>{completion_tokens:,}</td>
                    <td>${mr.total_cost_usd:.4f}</td>
                    <td>{cost_pct:.1f}%</td>
                </tr>"""

    # Calculate totals
    total_tests = sum(m.total_tests for m in results.model_results)
    total_prompt_tokens = sum(
        sum(r.token_usage.prompt_tokens for r in m.test_results)
        for m in results.model_results
    )
    total_completion_tokens = sum(
        sum(r.token_usage.completion_tokens for r in m.test_results)
        for m in results.model_results
    )

    # Use safe JSON serialization for all data embedded in script tags
    content = html_template.format(
        benchmark_name=html.escape(results.config.name),
        run_date=run_date,
        git_info_html=git_info_html,
        summary_rows=summary_rows,
        datasets_json=_safe_json_for_script(datasets),
        speed_price_datasets_json=_safe_json_for_script(speed_vs_price_datasets),
        detailed_rows=detailed_rows,
        cost_breakdown_rows=cost_breakdown_rows,
        cost_labels_json=_safe_json_for_script(cost_labels),
        cost_values_json=_safe_json_for_script(cost_values),
        cost_colors_json=_safe_json_for_script(colors),
        total_tests=total_tests,
        total_prompt_tokens=f"{total_prompt_tokens:,}",
        total_completion_tokens=f"{total_completion_tokens:,}",
        total_cost=f"${total_cost:.4f}",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

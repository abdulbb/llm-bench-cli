"""Data loading utilities for LLM-Bench."""

import csv
import json
from pathlib import Path
from typing import Any

from llm_bench.models import TestCase

# Security limits for data loading
MAX_DATA_FILE_SIZE = 100 * 1024 * 1024  # 100MB max for data files
MAX_INPUT_LENGTH = 1_000_000  # 1MB per input
MAX_TEST_CASES = 10_000  # Maximum test cases per file


class DataError(Exception):
    """Error raised when loading data fails."""

    pass


def load_test_cases(file_path: Path) -> list[TestCase]:
    """Load test cases from a file (CSV or JSONL).

    Args:
        file_path: Path to the data file.

    Returns:
        List of TestCase objects.

    Raises:
        DataError: If loading fails or format is invalid.
    """
    if not file_path.exists():
        raise DataError(f"Data file not found: {file_path}")

    # Security: Check file size to prevent DoS
    try:
        file_size = file_path.stat().st_size
        if file_size > MAX_DATA_FILE_SIZE:
            raise DataError(
                f"Data file too large: {file_size} bytes (max: {MAX_DATA_FILE_SIZE} bytes)"
            )
    except OSError as e:
        raise DataError(f"Failed to read data file: {e}") from e

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(file_path)
    elif suffix == ".jsonl":
        return _load_jsonl(file_path)
    else:
        raise DataError(f"Unsupported file format: {suffix}. Use .csv or .jsonl")


def _load_csv(file_path: Path) -> list[TestCase]:
    """Load test cases from a CSV file.

    Expected columns: 'input' (required), 'expected' (optional).
    'expected' should be a JSON string if present.

    Security: Enforces limits on number of test cases and input length.
    """
    test_cases: list[TestCase] = []
    try:
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "input" not in reader.fieldnames:
                raise DataError("CSV must have an 'input' column")

            for i, row in enumerate(reader, start=1):
                # Security: Limit number of test cases
                if len(test_cases) >= MAX_TEST_CASES:
                    raise DataError(
                        f"Too many test cases: exceeded limit of {MAX_TEST_CASES}"
                    )

                input_text = row["input"]

                # Security: Limit input length
                if len(input_text) > MAX_INPUT_LENGTH:
                    raise DataError(
                        f"Input at row {i} too long: {len(input_text)} chars "
                        f"(max: {MAX_INPUT_LENGTH})"
                    )

                expected_data: dict[str, Any] | None = None

                if "expected" in row and row["expected"]:
                    try:
                        expected_data = json.loads(row["expected"])
                    except json.JSONDecodeError as e:
                        raise DataError(
                            f"Invalid JSON in 'expected' column at row {i}: {e}"
                        ) from None

                test_cases.append(TestCase(input=input_text, expected=expected_data))

    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Failed to read CSV file: {e}") from e

    return test_cases


def _load_jsonl(file_path: Path) -> list[TestCase]:
    """Load test cases from a JSONL file.

    Each line should be a JSON object with 'input' and optional 'expected'.

    Security: Enforces limits on number of test cases and input length.
    """
    test_cases: list[TestCase] = []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Security: Limit number of test cases
                if len(test_cases) >= MAX_TEST_CASES:
                    raise DataError(
                        f"Too many test cases: exceeded limit of {MAX_TEST_CASES}"
                    )

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DataError(f"Invalid JSON at line {i}: {e}") from None

                if "input" not in data:
                    raise DataError(f"Missing 'input' field at line {i}")

                input_text = data["input"]

                # Security: Limit input length
                if isinstance(input_text, str) and len(input_text) > MAX_INPUT_LENGTH:
                    raise DataError(
                        f"Input at line {i} too long: {len(input_text)} chars "
                        f"(max: {MAX_INPUT_LENGTH})"
                    )

                test_cases.append(
                    TestCase(
                        input=input_text,
                        expected=data.get("expected"),
                    )
                )

    except DataError:
        raise
    except Exception as e:
        raise DataError(f"Failed to read JSONL file: {e}") from e

    return test_cases

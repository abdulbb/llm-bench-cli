"""Tests for external data loading."""

import csv
import json
from pathlib import Path

import pytest

from llm_bench.config import ConfigError, load_config


class TestDataLoading:
    """Tests for loading test cases from files."""

    def test_load_csv(self, tmp_path: Path) -> None:
        """Test loading test cases from CSV."""
        data_file = tmp_path / "tests.csv"
        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input", "expected"])
            writer.writerow(["input1", '{"key": "val1"}'])
            writer.writerow(["input2", ""])  # Empty expected

        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            f"""
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "{data_file.name}"
            """
        )

        config = load_config(config_file)
        assert len(config.test_cases) == 2
        assert config.test_cases[0].input == "input1"
        assert config.test_cases[0].expected == {"key": "val1"}
        assert config.test_cases[1].input == "input2"
        assert config.test_cases[1].expected is None

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Test loading test cases from JSONL."""
        data_file = tmp_path / "tests.jsonl"
        with open(data_file, "w") as f:
            f.write(json.dumps({"input": "input1", "expected": {"k": "v"}}) + "\n")
            f.write(json.dumps({"input": "input2"}) + "\n")

        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            f"""
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "{data_file.name}"
            """
        )

        config = load_config(config_file)
        assert len(config.test_cases) == 2
        assert config.test_cases[0].input == "input1"
        assert config.test_cases[0].expected == {"k": "v"}
        assert config.test_cases[1].input == "input2"
        assert config.test_cases[1].expected is None

    def test_mix_file_and_inline(self, tmp_path: Path) -> None:
        """Test mixing file and inline test cases."""
        data_file = tmp_path / "tests.csv"
        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input"])
            writer.writerow(["input1"])

        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            f"""
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "{data_file.name}"
            test_cases:
              - input: "inline1"
            """
        )

        config = load_config(config_file)
        assert len(config.test_cases) == 2
        assert config.test_cases[0].input == "input1"
        assert config.test_cases[1].input == "inline1"

    def test_missing_file_error(self, tmp_path: Path) -> None:
        """Test error when data file is missing."""
        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            """
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "missing.csv"
            """
        )

        with pytest.raises(ConfigError) as exc:
            load_config(config_file)
        assert "Failed to load test cases" in str(exc.value)

    def test_invalid_csv_format(self, tmp_path: Path) -> None:
        """Test error when CSV format is invalid (missing input column)."""
        data_file = tmp_path / "tests.csv"
        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wrong_column"])
            writer.writerow(["val"])

        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            f"""
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "{data_file.name}"
            """
        )

        with pytest.raises(ConfigError) as exc:
            load_config(config_file)
        assert "CSV must have an 'input' column" in str(exc.value)

    def test_empty_test_cases_error(self, tmp_path: Path) -> None:
        """Test error when both file and inline are missing/empty."""
        # Create empty file
        data_file = tmp_path / "tests.csv"
        with open(data_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input"])

        config_file = tmp_path / "bench.yaml"
        config_file.write_text(
            f"""
            name: "Test"
            system_prompt: "Sys"
            concurrency: 5
            temperature: 0.1
            models: ["gpt-4"]
            test_cases_file: "{data_file.name}"
            """
        )

        with pytest.raises(ConfigError) as exc:
            load_config(config_file)
        assert "Configuration validation failed" in str(exc.value)

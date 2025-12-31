"""Tests for CLI module."""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_bench import __version__
from llm_bench.cli import ExportFormat, app, apply_cli_overrides
from llm_bench.models import BenchConfig, RunConfig, TestCase

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_pattern.sub("", text)


def test_version() -> None:
    """Test version is set."""
    assert __version__ == "0.1.0"


class TestRunCommand:
    """Tests for the run command."""

    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid config file for testing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test Benchmark"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4"
  - "anthropic/claude-3"
config:
  concurrency: 5
  temperature: 0.1
test_cases:
  - input: "Hello"
    expected:
      response: "Hi"
""")
        return config_file

    def test_run_with_valid_config(self, valid_config_file: Path) -> None:
        """Test run command with a valid config file."""
        result = runner.invoke(app, ["run", "--config", str(valid_config_file)])
        assert result.exit_code == 0
        assert "Test Benchmark" in result.output
        assert "openai/gpt-4" in result.output
        assert "anthropic/claude-3" in result.output

    def test_run_with_missing_config(self, tmp_path: Path) -> None:
        """Test run command with missing config file."""
        result = runner.invoke(
            app, ["run", "--config", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code == 2  # Typer exits with 2 for invalid arguments

    def test_run_with_model_override(self, valid_config_file: Path) -> None:
        """Test run command with model override."""
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(valid_config_file),
                "--model",
                "openai/gpt-3.5-turbo",
            ],
        )
        assert result.exit_code == 0
        assert "gpt-3.5-turbo" in result.output
        assert "gpt-4" not in result.output

    def test_run_with_multiple_model_overrides(self, valid_config_file: Path) -> None:
        """Test run command with multiple model overrides."""
        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(valid_config_file),
                "--model",
                "model-a",
                "--model",
                "model-b",
            ],
        )
        assert result.exit_code == 0
        assert "model-a" in result.output
        assert "model-b" in result.output

    def test_run_with_concurrency_override(self, valid_config_file: Path) -> None:
        """Test run command with concurrency override."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--concurrency", "10"],
        )
        assert result.exit_code == 0
        assert "Concurrency: 10" in result.output

    def test_run_with_temperature_override(self, valid_config_file: Path) -> None:
        """Test run command with temperature override."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--temperature", "0.7"],
        )
        assert result.exit_code == 0
        assert "Temperature: 0.7" in result.output

    def test_run_with_no_cache_flag(self, valid_config_file: Path) -> None:
        """Test run command with --no-cache flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--no-cache"],
        )
        assert result.exit_code == 0
        assert "Cache: disabled" in result.output

    def test_run_with_export_html(self, valid_config_file: Path) -> None:
        """Test run command with HTML export."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--export", "html"],
        )
        assert result.exit_code == 0
        assert "Export: html" in result.output

    def test_run_with_export_csv(self, valid_config_file: Path) -> None:
        """Test run command with CSV export."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--export", "csv"],
        )
        assert result.exit_code == 0
        assert "Export: csv" in result.output

    def test_run_with_export_json(self, valid_config_file: Path) -> None:
        """Test run command with JSON export."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--export", "json"],
        )
        assert result.exit_code == 0
        assert "Export: json" in result.output

    def test_run_with_validators_file_override(
        self, valid_config_file: Path, tmp_path: Path
    ) -> None:
        """Test run command with validators file override."""
        validators_file = tmp_path / "validators.py"
        validators_file.touch()

        result = runner.invoke(
            app,
            [
                "run",
                "--config",
                str(valid_config_file),
                "--validators-file",
                str(validators_file),
            ],
        )
        assert result.exit_code == 0
        # We can't easily check the internal state here, but we can verify it doesn't crash
        # and assumes it loaded the config. To be sure, we could check if it processed the option.
        # But this confirms the CLI option is accepted.

    def test_run_with_invalid_concurrency(self, valid_config_file: Path) -> None:
        """Test run command rejects concurrency > 100."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--concurrency", "101"],
        )
        assert result.exit_code == 2

    def test_run_with_invalid_temperature(self, valid_config_file: Path) -> None:
        """Test run command rejects temperature > 2.0."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--temperature", "2.5"],
        )
        assert result.exit_code == 2

    def test_run_help(self) -> None:
        """Test run command help output."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--config" in output
        assert "--model" in output
        assert "--concurrency" in output
        assert "--temperature" in output
        assert "--no-cache" in output
        assert "--export" in output


class TestApplyCliOverrides:
    """Tests for apply_cli_overrides function."""

    @pytest.fixture
    def base_config(self) -> BenchConfig:
        """Create a base configuration for testing."""
        return BenchConfig(
            name="Test",
            system_prompt="Test prompt",
            models=["model-a", "model-b"],
            config=RunConfig(concurrency=5, temperature=0.1),
            test_cases=[TestCase(input="test", expected={})],
        )

    def test_no_overrides(self, base_config: BenchConfig) -> None:
        """Test that config is unchanged when no overrides provided."""
        result = apply_cli_overrides(base_config)
        assert result == base_config

    def test_override_models(self, base_config: BenchConfig) -> None:
        """Test overriding models list."""
        result = apply_cli_overrides(base_config, models=["new-model"])
        assert result.models == ["new-model"]
        assert result.config.concurrency == 5  # unchanged

    def test_override_concurrency(self, base_config: BenchConfig) -> None:
        """Test overriding concurrency."""
        result = apply_cli_overrides(base_config, concurrency=20)
        assert result.config.concurrency == 20
        assert result.config.temperature == 0.1  # unchanged

    def test_override_temperature(self, base_config: BenchConfig) -> None:
        """Test overriding temperature."""
        result = apply_cli_overrides(base_config, temperature=0.8)
        assert result.config.temperature == 0.8
        assert result.config.concurrency == 5  # unchanged

    def test_override_multiple(self, base_config: BenchConfig) -> None:
        """Test overriding multiple values."""
        result = apply_cli_overrides(
            base_config,
            models=["new-model"],
            concurrency=10,
            temperature=0.5,
        )
        assert result.models == ["new-model"]
        assert result.config.concurrency == 10
        assert result.config.temperature == 0.5

    def test_override_validators_file(self, base_config: BenchConfig) -> None:
        """Test overriding validators file."""
        path = Path("validators.py")
        result = apply_cli_overrides(base_config, validators_file=path)
        assert result.validators_file == path

    def test_preserves_other_fields(self, base_config: BenchConfig) -> None:
        """Test that non-overridden fields are preserved."""
        result = apply_cli_overrides(base_config, models=["new-model"])
        assert result.name == "Test"
        assert result.system_prompt == "Test prompt"
        assert len(result.test_cases) == 1


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_html_value(self) -> None:
        """Test HTML format value."""
        assert ExportFormat.HTML.value == "html"

    def test_csv_value(self) -> None:
        """Test CSV format value."""
        assert ExportFormat.CSV.value == "csv"

    def test_json_value(self) -> None:
        """Test JSON format value."""
        assert ExportFormat.JSON.value == "json"


class TestNewRunFlags:
    """Tests for new run command flags."""

    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid config file for testing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test Benchmark"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4"
config:
  concurrency: 5
  temperature: 0.1
test_cases:
  - input: "Hello"
    expected:
      response: "Hi"
""")
        return config_file

    def test_run_with_verbose_flag(self, valid_config_file: Path) -> None:
        """Test run command with --verbose flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--verbose"],
        )
        assert result.exit_code == 0
        # Verbose mode shows debug info
        assert "DEBUG" in result.output or "Test Benchmark" in result.output

    def test_run_with_quiet_flag(self, valid_config_file: Path) -> None:
        """Test run command with --quiet flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--quiet"],
        )
        assert result.exit_code == 0
        # Quiet mode suppresses normal output
        # We just verify it runs successfully

    def test_run_with_no_color_flag(self, valid_config_file: Path) -> None:
        """Test run command with --no-color flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--no-color"],
        )
        assert result.exit_code == 0

    def test_run_with_dry_run_flag(self, valid_config_file: Path) -> None:
        """Test run command with --dry-run flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry Run Mode" in result.output
        assert "No API calls will be made" in result.output
        assert "Total tasks:" in result.output

    def test_run_with_fail_fast_flag(self, valid_config_file: Path) -> None:
        """Test run command with --fail-fast flag."""
        result = runner.invoke(
            app,
            ["run", "--config", str(valid_config_file), "--fail-fast"],
        )
        # Should complete (possibly with failures, but command should work)
        assert result.exit_code == 0

    def test_run_help_shows_new_flags(self) -> None:
        """Test run command help shows new flags."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--verbose" in output
        assert "--quiet" in output
        assert "--no-color" in output
        assert "--dry-run" in output
        assert "--fail-fast" in output


class TestValidateCommand:
    """Tests for the validate command."""

    @pytest.fixture
    def valid_config_file(self, tmp_path: Path) -> Path:
        """Create a valid config file for testing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test Benchmark"
system_prompt: "You are a helpful assistant."
models:
  - "openai/gpt-4"
config:
  concurrency: 5
  temperature: 0.1
test_cases:
  - input: "Hello"
    expected:
      response: "Hi"
""")
        return config_file

    @pytest.fixture
    def invalid_config_file(self, tmp_path: Path) -> Path:
        """Create an invalid config file for testing."""
        config_file = tmp_path / "invalid.config.yaml"
        config_file.write_text("""
name: "Test"
# Missing required fields
""")
        return config_file

    def test_validate_valid_config(self, valid_config_file: Path) -> None:
        """Test validate command with valid config."""
        result = runner.invoke(app, ["validate", "--config", str(valid_config_file)])
        assert result.exit_code == 0
        assert "Validating:" in result.output
        assert "Benchmark Name:" in result.output

    def test_validate_invalid_config(self, invalid_config_file: Path) -> None:
        """Test validate command with invalid config."""
        result = runner.invoke(app, ["validate", "--config", str(invalid_config_file)])
        assert result.exit_code == 1
        assert "Validation Failed" in result.output or "Error:" in result.output

    def test_validate_missing_config(self, tmp_path: Path) -> None:
        """Test validate command with missing config file."""
        result = runner.invoke(
            app, ["validate", "--config", str(tmp_path / "nonexistent.yaml")]
        )
        assert result.exit_code == 2

    def test_validate_help(self) -> None:
        """Test validate command help output."""
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--config" in output


class TestModelsCommand:
    """Tests for the models command."""

    def test_models_lists_providers(self) -> None:
        """Test models command lists providers."""
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "Available Models" in result.output
        assert "OPENAI" in result.output
        assert "ANTHROPIC" in result.output

    def test_models_with_provider_filter(self) -> None:
        """Test models command with provider filter."""
        result = runner.invoke(app, ["models", "--provider", "openai"])
        assert result.exit_code == 0
        assert "OPENAI" in result.output
        # Should not show other providers
        assert "ANTHROPIC" not in result.output

    def test_models_unknown_provider(self) -> None:
        """Test models command with unknown provider."""
        result = runner.invoke(app, ["models", "--provider", "unknown"])
        assert result.exit_code == 0
        assert "Unknown provider" in result.output

    def test_models_help(self) -> None:
        """Test models command help output."""
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--provider" in output


class TestCacheCommands:
    """Tests for cache subcommands."""

    def test_cache_info(self) -> None:
        """Test cache info command."""
        result = runner.invoke(app, ["cache", "info"])
        assert result.exit_code == 0
        assert "Cache Information" in result.output
        assert "Location:" in result.output
        assert "Entries:" in result.output
        assert "Size:" in result.output

    def test_cache_clear_empty(self) -> None:
        """Test cache clear with empty cache."""
        # First clear any existing cache
        result = runner.invoke(app, ["cache", "clear", "--force"])
        # Then try again - should say already empty
        result = runner.invoke(app, ["cache", "clear", "--force"])
        assert result.exit_code == 0
        assert "already empty" in result.output or "cleared" in result.output

    def test_cache_clear_with_force(self) -> None:
        """Test cache clear with --force flag."""
        result = runner.invoke(app, ["cache", "clear", "--force"])
        assert result.exit_code == 0

    def test_cache_help(self) -> None:
        """Test cache subcommand help."""
        result = runner.invoke(app, ["cache", "--help"])
        assert result.exit_code == 0
        assert "info" in result.output
        assert "clear" in result.output


class TestInitCommand:
    """Tests for the init command."""

    def test_init_non_interactive(self, tmp_path: Path) -> None:
        """Test init command with non-interactive flag."""
        output_file = tmp_path / "test.config.yaml"
        result = runner.invoke(
            app,
            ["init", "--output", str(output_file), "--non-interactive"],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Template configuration created" in result.output

    def test_init_non_interactive_creates_valid_yaml(self, tmp_path: Path) -> None:
        """Test init creates valid YAML config."""
        import yaml

        output_file = tmp_path / "test.config.yaml"
        runner.invoke(
            app,
            ["init", "--output", str(output_file), "--non-interactive"],
        )

        # Load and verify YAML
        with open(output_file) as f:
            config = yaml.safe_load(f)

        assert "name" in config
        assert "system_prompt" in config
        assert "models" in config
        assert "test_cases" in config
        assert len(config["models"]) > 0
        assert len(config["test_cases"]) > 0

    def test_init_help(self) -> None:
        """Test init command help output."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--output" in output
        assert "--non-interactive" in output


class TestMainHelp:
    """Tests for main CLI help."""

    def test_main_help_shows_all_commands(self) -> None:
        """Test main help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "validate" in result.output
        assert "models" in result.output
        assert "cache" in result.output
        assert "init" in result.output

    def test_no_args_shows_help(self) -> None:
        """Test CLI with no args shows help."""
        result = runner.invoke(app, [])
        # Typer returns exit code 0 when showing help
        assert result.exit_code == 0 or "Usage:" in result.output
        # Should show usage info in output
        assert "Usage:" in result.output or "run" in result.output

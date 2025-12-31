"""Tests for configuration loading."""

from pathlib import Path

import pytest

from llm_bench.config import ConfigError, load_config, load_json_schema


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid configuration file."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test Benchmark"
system_prompt: "You are a helpful assistant."
concurrency: 5
temperature: 0.1
models:
  - "openai/gpt-4"
  - "anthropic/claude-3"
test_cases:
  - input: "Hello"
    expected:
      response: "Hi"
""")
        config = load_config(config_file)
        assert config.name == "Test Benchmark"
        assert config.system_prompt == "You are a helpful assistant."
        assert config.models == ["openai/gpt-4", "anthropic/claude-3"]
        assert len(config.test_cases) == 1
        assert config.test_cases[0].input == "Hello"

    def test_load_config_with_schema(self, tmp_path: Path) -> None:
        """Test loading config with schema path."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object"}')

        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test prompt"
schema: "schema.json"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - input: "Test"
    expected: {}
""")
        config = load_config(config_file)
        assert config.schema_path == schema_file.resolve()

    def test_load_config_with_custom_settings(self, tmp_path: Path) -> None:
        """Test loading config with custom concurrency and temperature."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test prompt"
models:
  - "gpt-4"
config:
  concurrency: 10
  temperature: 0.5
test_cases:
  - input: "Test"
    expected: {}
""")
        config = load_config(config_file)
        assert config.config.concurrency == 10
        assert config.config.temperature == 0.5

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        """Test error when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "not found" in str(exc_info.value)

    def test_load_config_path_is_directory(self, tmp_path: Path) -> None:
        """Test error when config path is a directory."""
        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)
        assert "not a file" in str(exc_info.value)

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error when YAML syntax is invalid."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("name: [unclosed bracket")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "Invalid YAML" in str(exc_info.value)

    def test_load_config_not_a_mapping(self, tmp_path: Path) -> None:
        """Test error when YAML is not a mapping."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "mapping" in str(exc_info.value).lower()

    def test_load_config_missing_name(self, tmp_path: Path) -> None:
        """Test error when name field is missing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - input: "Test"
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "name" in str(exc_info.value).lower()

    def test_load_config_missing_models(self, tmp_path: Path) -> None:
        """Test error when models field is missing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
test_cases:
  - input: "Test"
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "models" in str(exc_info.value).lower()

    def test_load_config_empty_models(self, tmp_path: Path) -> None:
        """Test error when models list is empty."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models: []
test_cases:
  - input: "Test"
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "validation failed" in str(exc_info.value).lower()

    def test_load_config_missing_test_cases(self, tmp_path: Path) -> None:
        """Test error when test_cases field is missing."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "test_cases" in str(exc_info.value).lower()

    def test_load_config_empty_test_cases(self, tmp_path: Path) -> None:
        """Test error when test_cases list is empty."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases: []
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "validation failed" in str(exc_info.value).lower()

    def test_load_config_invalid_test_case(self, tmp_path: Path) -> None:
        """Test error when test case is invalid."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - input: ""
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "test case" in str(exc_info.value).lower()

    def test_load_config_test_case_not_mapping(self, tmp_path: Path) -> None:
        """Test error when test case is not a mapping."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - "just a string"
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "mapping" in str(exc_info.value).lower()

    def test_load_config_invalid_concurrency(self, tmp_path: Path) -> None:
        """Test error when concurrency is invalid."""
        config_file = tmp_path / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
models:
  - "gpt-4"
config:
  concurrency: 0
  temperature: 0.1
test_cases:
  - input: "Test"
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "validation failed" in str(exc_info.value).lower()

    def test_load_config_relative_schema_path(self, tmp_path: Path) -> None:
        """Test that relative schema path within config dir is resolved correctly."""
        config_file = tmp_path / "bench.config.yaml"
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object"}')
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
schema: "schema.json"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - input: "Test"
    expected: {}
""")
        config = load_config(config_file)
        expected_path = schema_file.resolve()
        assert config.schema_path == expected_path

    def test_load_config_parent_path_blocked(self, tmp_path: Path) -> None:
        """Test that parent directory path traversal is blocked for security."""
        subdir = tmp_path / "configs"
        subdir.mkdir()
        config_file = subdir / "bench.config.yaml"
        config_file.write_text("""
name: "Test"
system_prompt: "Test"
schema: "../schema.json"
concurrency: 5
temperature: 0.1
models:
  - "gpt-4"
test_cases:
  - input: "Test"
    expected: {}
""")
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "Security" in str(exc_info.value)
        assert "outside allowed directory" in str(exc_info.value)


class TestLoadJsonSchema:
    """Tests for load_json_schema function."""

    def test_load_valid_schema(self, tmp_path: Path) -> None:
        """Test loading a valid JSON schema."""
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(
            '{"type": "object", "properties": {"name": {"type": "string"}}}'
        )
        schema = load_json_schema(schema_file)
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_load_schema_not_found(self, tmp_path: Path) -> None:
        """Test error when schema file doesn't exist."""
        schema_file = tmp_path / "nonexistent.json"
        with pytest.raises(ConfigError) as exc_info:
            load_json_schema(schema_file)
        assert "not found" in str(exc_info.value)

    def test_load_schema_invalid_json(self, tmp_path: Path) -> None:
        """Test error when JSON is invalid."""
        schema_file = tmp_path / "invalid.json"
        schema_file.write_text("{invalid json}")
        with pytest.raises(ConfigError) as exc_info:
            load_json_schema(schema_file)
        assert "Invalid JSON" in str(exc_info.value)

    def test_load_schema_not_object(self, tmp_path: Path) -> None:
        """Test error when schema is not an object."""
        schema_file = tmp_path / "array.json"
        schema_file.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError) as exc_info:
            load_json_schema(schema_file)
        assert "object" in str(exc_info.value).lower()


class TestConfigError:
    """Tests for ConfigError class."""

    def test_error_without_details(self) -> None:
        """Test error message without details."""
        error = ConfigError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_error_with_details(self) -> None:
        """Test error message with details."""
        error = ConfigError("Something went wrong", "More info here")
        assert "Something went wrong" in str(error)
        assert "More info here" in str(error)
        assert "Details:" in str(error)

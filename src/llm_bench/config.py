"""Configuration loading and validation for LLM-Bench."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from yaml import SafeLoader

from llm_bench.data import DataError, load_test_cases
from llm_bench.models import BenchConfig, RunConfig, TestCase

# Security limits
MAX_CONFIG_FILE_SIZE = 10 * 1024 * 1024  # 10MB max for config files


class ConfigError(Exception):
    """Error raised when configuration is invalid."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.format_message())

    def format_message(self) -> str:
        """Format the error message with optional details."""
        if self.details:
            return f"{self.message}\n\nDetails:\n{self.details}"
        return self.message


def load_config(config_path: Path) -> BenchConfig:
    """Load and validate benchmark configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated BenchConfig instance.

    Raises:
        ConfigError: If the file cannot be read or validation fails.
    """
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")

    # Security: Check file size to prevent DoS via large files
    try:
        file_size = config_path.stat().st_size
        if file_size > MAX_CONFIG_FILE_SIZE:
            raise ConfigError(
                f"Configuration file too large: {file_size} bytes "
                f"(max: {MAX_CONFIG_FILE_SIZE} bytes)",
                f"File: {config_path}",
            )
    except OSError as e:
        raise ConfigError(
            f"Failed to access configuration file: {config_path}",
            f"OS error: {e.strerror} (errno {e.errno})",
        ) from None

    try:
        raw_content = config_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(
            f"Failed to read configuration file: {config_path}",
            f"OS error: {e.strerror} (errno {e.errno})",
        ) from None

    try:
        # Use SafeLoader explicitly to prevent code execution
        raw_config = yaml.load(raw_content, Loader=SafeLoader)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML syntax in configuration file: {config_path}", str(e)
        ) from None
    except RecursionError:
        raise ConfigError(
            "Configuration file contains deeply nested structures (potential DoS)",
            f"File: {config_path}",
        ) from None

    if not isinstance(raw_config, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML mapping (dictionary): {config_path}",
            f"Got {type(raw_config).__name__} instead",
        )

    return _parse_config(raw_config, config_path.parent)


def _parse_config(raw_config: dict[str, Any], base_path: Path) -> BenchConfig:
    """Parse and validate raw configuration dictionary.

    Args:
        raw_config: Raw configuration dictionary from YAML.
        base_path: Base path for resolving relative paths.

    Returns:
        Validated BenchConfig instance.

    Raises:
        ConfigError: If validation fails.
    """
    try:
        # Transform raw config to match model structure
        config_data = _transform_config(raw_config, base_path)
        return BenchConfig(**config_data)
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise ConfigError("Configuration validation failed", error_details) from None


def _validate_path_within_base(
    path: Path,
    resolved_base: Path,
    path_name: str,
) -> Path:
    """Validate that a resolved path is within the base directory only.

    Prevents path traversal attacks by ensuring resolved paths don't escape
    beyond the base directory. Parent directory access is not allowed.

    Args:
        path: Path to validate (already resolved).
        resolved_base: Pre-resolved base directory.
        path_name: Name of the path (for error messages).

    Returns:
        The validated path.

    Raises:
        ConfigError: If the path is outside the allowed directory.
    """
    # Security: Check for symlinks that might escape the base directory
    # Resolve the actual target if it's a symlink
    actual_path = path
    if path.is_symlink():
        try:
            actual_path = path.resolve(strict=True)
        except OSError as e:
            raise ConfigError(
                f"Security: {path_name} symlink target does not exist or is inaccessible",
                f"Path: {path}, Error: {e.strerror if hasattr(e, 'strerror') else str(e)}",
            ) from None

    # Compute relative path to check for ".." sequences
    try:
        relative = actual_path.relative_to(resolved_base)
        # Explicitly check for ".." in the relative path
        if ".." in str(relative):
            raise ConfigError(
                f"Security: {path_name} path contains '..' which is not allowed",
                f"Attempted path: {path}",
            )
        return actual_path
    except ValueError:
        pass

    # Path is outside allowed directory
    raise ConfigError(
        f"Security: {path_name} path {actual_path} is outside allowed directory "
        f"({resolved_base}). Path traversal is not allowed for security reasons."
    )


def _transform_config(raw_config: dict[str, Any], base_path: Path) -> dict[str, Any]:
    """Transform raw YAML config to match BenchConfig model structure.

    Args:
        raw_config: Raw configuration dictionary from YAML.
        base_path: Base path for resolving relative paths.

    Returns:
        Transformed configuration dictionary.
    """
    config_data: dict[str, Any] = {}

    # Pre-resolve base path once to avoid repeated resolution
    resolved_base = base_path.resolve()

    # Direct mappings
    if "name" in raw_config:
        config_data["name"] = raw_config["name"]

    if "system_prompt" in raw_config:
        config_data["system_prompt"] = raw_config["system_prompt"]

    if "models" in raw_config:
        config_data["models"] = raw_config["models"]

    # Handle schema path - resolve relative to config file
    if "schema" in raw_config:
        schema_path = Path(raw_config["schema"])
        if not schema_path.is_absolute():
            schema_path = (resolved_base / schema_path).resolve()
        else:
            schema_path = schema_path.resolve()
        # Validate path is within base directory
        config_data["schema_path"] = _validate_path_within_base(
            schema_path, resolved_base, "schema"
        )

    # Handle validators file - resolve relative to config file
    if "validators_file" in raw_config:
        validators_path = Path(raw_config["validators_file"])
        if not validators_path.is_absolute():
            validators_path = (resolved_base / validators_path).resolve()
        else:
            validators_path = validators_path.resolve()
        # Validate path is within base directory (critical for security)
        config_data["validators_file"] = _validate_path_within_base(
            validators_path, resolved_base, "validators_file"
        )

    # Handle test cases file
    loaded_test_cases: list[TestCase] = []
    if "test_cases_file" in raw_config:
        file_path = Path(raw_config["test_cases_file"])
        if not file_path.is_absolute():
            file_path = (resolved_base / file_path).resolve()
        else:
            file_path = file_path.resolve()
        # Validate path is within base directory
        config_data["test_cases_file"] = _validate_path_within_base(
            file_path, resolved_base, "test_cases_file"
        )

        try:
            loaded_test_cases = load_test_cases(file_path)
        except DataError as e:
            raise ConfigError(
                f"Failed to load test cases from file: {file_path}", str(e)
            ) from None

    # Handle nested config object or top-level run config fields
    if "config" in raw_config:
        raw_run_config = raw_config["config"]
        if isinstance(raw_run_config, dict):
            config_data["config"] = RunConfig(**raw_run_config)
    else:
        # Check for top-level run config fields
        run_config_fields = {}
        for field in ["concurrency", "temperature", "judge_model", "max_cost"]:
            if field in raw_config:
                run_config_fields[field] = raw_config[field]
        if run_config_fields:
            config_data["config"] = RunConfig(**run_config_fields)

    # Handle test cases
    test_cases: list[TestCase] = list(loaded_test_cases)

    if "test_cases" in raw_config:
        raw_test_cases = raw_config["test_cases"]
        if isinstance(raw_test_cases, list):
            for i, tc in enumerate(raw_test_cases):
                if isinstance(tc, dict):
                    try:
                        test_cases.append(TestCase(**tc))
                    except ValidationError as e:
                        raise ConfigError(
                            f"Invalid test case at index {i}",
                            _format_validation_errors(e),
                        ) from None
                else:
                    raise ConfigError(
                        f"Test case at index {i} must be a mapping",
                        f"Got {type(tc).__name__} instead",
                    )

    if test_cases:
        config_data["test_cases"] = test_cases

    return config_data


def _format_validation_errors(error: ValidationError) -> str:
    """Format Pydantic validation errors into readable messages.

    Args:
        error: Pydantic ValidationError instance.

    Returns:
        Formatted error string.
    """
    lines: list[str] = []
    for err in error.errors():
        location = " -> ".join(str(loc) for loc in err["loc"])
        message = err["msg"]
        lines.append(f"  - {location}: {message}")
    return "\n".join(lines)


def load_json_schema(schema_path: Path) -> dict[str, Any]:
    """Load a JSON schema file.

    Args:
        schema_path: Path to the JSON schema file.

    Returns:
        Parsed JSON schema as a dictionary.

    Raises:
        ConfigError: If the schema file cannot be read or parsed.
    """
    import json

    if not schema_path.exists():
        raise ConfigError(f"JSON schema file not found: {schema_path}")

    try:
        content = schema_path.read_text(encoding="utf-8")
        schema = json.loads(content)
    except OSError as e:
        raise ConfigError(
            f"Failed to read JSON schema file: {schema_path}", str(e)
        ) from None
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"Invalid JSON in schema file: {schema_path}", str(e)
        ) from None

    if not isinstance(schema, dict):
        raise ConfigError(
            "JSON schema must be an object",
            f"Got {type(schema).__name__} instead",
        )

    return schema

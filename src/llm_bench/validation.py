"""Validation pipeline for LLM-Bench."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepdiff import DeepDiff
from pydantic import ValidationError, create_model

from llm_bench.llm import LLMResponse, call_llm
from llm_bench.models import ValidationStatus

if TYPE_CHECKING:
    from llm_bench.cache import ResponseCache

# Cache for converted Pydantic models (schema hash -> model class)
_schema_model_cache: dict[str, type[Any]] = {}

# Cache for loaded validator modules (path -> validators dict)
_validator_module_cache: dict[str, dict[str, Callable[[str], Any]]] = {}

# Allowlist of safe modules for validator imports
ALLOWED_VALIDATOR_IMPORTS: frozenset[str] = frozenset(
    {
        "re",
        "json",
        "math",
        "string",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "operator",
        "typing",
    }
)

# Dangerous built-in functions that should be blocked
DANGEROUS_BUILTINS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "memoryview",
        "globals",
        "locals",
        "vars",
    }
)


class ValidatorSecurityError(Exception):
    """Raised when validator code contains unsafe operations."""

    pass


def _validate_validator_code(content: str, path: Path) -> None:
    """Validate that validator code only uses safe operations.

    This function performs static analysis on validator code to detect
    potentially dangerous operations before execution.

    Args:
        content: Python source code to validate.
        path: Path to the validator file (for error messages).

    Raises:
        ValidatorSecurityError: If unsafe operations are detected.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise ValidatorSecurityError(
            f"Syntax error in validator file {path}: {e}"
        ) from e

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name not in ALLOWED_VALIDATOR_IMPORTS:
                    raise ValidatorSecurityError(
                        f"Import of '{alias.name}' not allowed in validator file {path}. "
                        f"Allowed imports: {', '.join(sorted(ALLOWED_VALIDATOR_IMPORTS))}"
                    )

        if isinstance(node, ast.ImportFrom) and node.module:
            module_name = node.module.split(".")[0]
            if module_name not in ALLOWED_VALIDATOR_IMPORTS:
                raise ValidatorSecurityError(
                    f"Import from '{node.module}' not allowed in validator file {path}. "
                    f"Allowed imports: {', '.join(sorted(ALLOWED_VALIDATOR_IMPORTS))}"
                )

        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
                raise ValidatorSecurityError(
                    f"Dangerous function '{node.func.id}' not allowed in validator file {path}"
                )
            # Check for getattr tricks like getattr(__builtins__, 'eval')
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in DANGEROUS_BUILTINS
            ):
                raise ValidatorSecurityError(
                    f"Attempt to access dangerous function via getattr in {path}"
                )

        # Check for attribute access to dangerous names
        if isinstance(node, ast.Attribute) and node.attr in {
            "__code__",
            "__globals__",
            "__builtins__",
            "__subclasses__",
        }:
            raise ValidatorSecurityError(
                f"Access to '{node.attr}' not allowed in validator file {path}"
            )


@dataclass
class ValidationResult:
    """Result of validating an LLM response."""

    status: ValidationStatus
    passed: bool
    parsed_output: dict[str, Any] | None = None
    error_message: str | None = None
    diff_details: dict[str, Any] = field(default_factory=dict)
    used_fuzzy_match: bool = False


async def validate_response(
    raw_output: str,
    expected: dict[str, Any] | None,
    schema: dict[str, Any] | None = None,
    judge_model: str | None = None,
    cache: ResponseCache | None = None,
    regex_pattern: str | None = None,
    custom_validator: Callable[[str], bool | tuple[bool, str]] | None = None,
) -> ValidationResult:
    """Validate an LLM response through the waterfall pipeline.

    The validation pipeline has four stages:
    1. JSON Parse Check - Verify output is valid JSON
    2. Schema Validation - Validate against JSON schema (if provided)
    3. Strict Equality - Compare with expected output using DeepDiff
    4. Fuzzy Match (Optional) - Use LLM judge if strict equality fails

    Args:
        raw_output: Raw string output from the LLM.
        expected: Expected JSON output to compare against.
        schema: Optional JSON schema to validate against.
        judge_model: Optional LLM model to use as a judge.
        cache: Optional cache for judge responses.

    Returns:
        ValidationResult with status, pass/fail, and details.
    """
    # Freeform mode: No validation criteria defined, always pass
    # This allows users to inspect raw outputs without any validation
    if (
        expected is None
        and schema is None
        and regex_pattern is None
        and custom_validator is None
    ):
        return ValidationResult(
            status=ValidationStatus.PASSED,
            passed=True,
        )

    # Stage 0: Regex Check
    if regex_pattern:
        import re

        if not re.search(regex_pattern, raw_output, re.MULTILINE):
            return ValidationResult(
                status=ValidationStatus.FAILED_REGEX,
                passed=False,
                error_message=f"Output did not match regex pattern: {regex_pattern}",
            )
        # If regex passed and we don't expect JSON (no expected/schema), return pass
        if expected is None and schema is None and custom_validator is None:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                passed=True,
            )

    # Stage 0.5: Custom Validator
    if custom_validator:
        try:
            result = custom_validator(raw_output)
            # Handle return types: bool or (bool, str)
            if isinstance(result, tuple):
                passed, msg = result
            else:
                passed = bool(result)
                msg = "Custom validation failed"

            if not passed:
                return ValidationResult(
                    status=ValidationStatus.FAILED_CUSTOM,
                    passed=False,
                    error_message=msg,
                )

            # If custom validator passed and we don't expect JSON, return pass
            if expected is None and schema is None:
                return ValidationResult(
                    status=ValidationStatus.PASSED,
                    passed=True,
                )
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.FAILED_CUSTOM,
                passed=False,
                error_message=f"Validator raised exception: {str(e)}",
            )

    # Stage 1: JSON Parse Check
    parse_result = _validate_json_parse(raw_output)
    if not parse_result.passed:
        return parse_result

    parsed_output = parse_result.parsed_output
    assert parsed_output is not None  # For type checker

    # Stage 2: Schema Validation (if schema provided)
    if schema is not None:
        schema_result = _validate_schema(parsed_output, schema)
        if not schema_result.passed:
            schema_result.parsed_output = parsed_output
            return schema_result

    # Stage 3: DeepDiff Strict Equality
    if expected is None:
        return ValidationResult(
            status=ValidationStatus.PASSED,
            passed=True,
            parsed_output=parsed_output,
        )

    equality_result = _validate_equality(parsed_output, expected)
    if equality_result.passed:
        equality_result.parsed_output = parsed_output
        return equality_result

    # Stage 4: Fuzzy Match (LLM Judge)
    if judge_model:
        return await _validate_fuzzy_llm(
            parsed_output, expected, judge_model, cache, equality_result
        )

    equality_result.parsed_output = parsed_output
    return equality_result


async def _validate_fuzzy_llm(
    actual: dict[str, Any],
    expected: dict[str, Any],
    judge_model: str,
    cache: ResponseCache | None,
    original_failure: ValidationResult,
) -> ValidationResult:
    """Stage 4: Use LLM as judge to determine semantic equivalence.

    Args:
        actual: Actual parsed output.
        expected: Expected output.
        judge_model: Model to use for judging.
        cache: Cache for judge responses.
        original_failure: The failure result from strict equality check.

    Returns:
        ValidationResult with fuzzy match status.
    """
    system_prompt = (
        "You are a judge that determines if two JSON objects are semantically equivalent. "
        "Reply only with 'PASS' if they are equivalent, or 'FAIL' if they are not. "
        "Do not provide explanations."
    )
    # Sort keys for deterministic input
    user_input = (
        f"Expected: {json.dumps(expected, sort_keys=True)}\n"
        f"Actual: {json.dumps(actual, sort_keys=True)}"
    )

    try:
        response: LLMResponse | None = None

        # Check cache
        if cache:
            response = cache.get(
                model=judge_model,
                system_prompt=system_prompt,
                user_input=user_input,
                temperature=0.0,
            )

        if response is None:
            response = await call_llm(
                model=judge_model,
                system_prompt=system_prompt,
                user_input=user_input,
                temperature=0.0,
                stream=False,
            )
            if cache:
                cache.set(
                    model=judge_model,
                    system_prompt=system_prompt,
                    user_input=user_input,
                    temperature=0.0,
                    response=response,
                )

        content = response.content.strip().upper()

        if "PASS" in content:
            return ValidationResult(
                status=ValidationStatus.PASSED,
                passed=True,
                parsed_output=actual,
                used_fuzzy_match=True,
            )
        else:
            original_failure.status = ValidationStatus.FAILED_FUZZY
            original_failure.parsed_output = actual
            original_failure.error_message = (
                f"{original_failure.error_message}\n\n"
                f"Fuzzy match (LLM Judge {judge_model}) also failed."
            )
            return original_failure

    except Exception as e:
        # If judge fails, return original failure with note
        original_failure.error_message = (
            f"{original_failure.error_message}\n\n"
            f"Fuzzy match attempt failed with error: {str(e)}"
        )
        original_failure.parsed_output = actual
        return original_failure


def _validate_json_parse(raw_output: str) -> ValidationResult:
    """Stage 1: Validate that output is parseable JSON.

    Args:
        raw_output: Raw string output from the LLM.

    Returns:
        ValidationResult with parsed JSON or error details.
    """
    # Handle empty output
    if not raw_output or not raw_output.strip():
        return ValidationResult(
            status=ValidationStatus.FAILED_JSON_PARSE,
            passed=False,
            error_message="Empty output - expected JSON",
        )

    # Try to extract JSON from markdown code blocks
    cleaned_output = _extract_json_from_markdown(raw_output)

    try:
        parsed = json.loads(cleaned_output)

        # Ensure it's a dict (not a list or primitive)
        if not isinstance(parsed, dict):
            return ValidationResult(
                status=ValidationStatus.FAILED_JSON_PARSE,
                passed=False,
                error_message=f"Expected JSON object, got {type(parsed).__name__}",
            )

        return ValidationResult(
            status=ValidationStatus.PASSED,
            passed=True,
            parsed_output=parsed,
        )
    except json.JSONDecodeError as e:
        return ValidationResult(
            status=ValidationStatus.FAILED_JSON_PARSE,
            passed=False,
            error_message=f"Invalid JSON: {e.msg} at position {e.pos}",
        )


def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks if present.

    Handles common LLM output patterns like:
    - ```json\n{...}\n```
    - ```\n{...}\n```

    Args:
        text: Raw text that may contain markdown code blocks.

    Returns:
        Extracted JSON string or original text.
    """
    text = text.strip()

    # Check for ```json or ``` code blocks
    if text.startswith("```"):
        lines = text.split("\n")

        # Find the start of JSON content
        start_idx = 1 if lines[0] in ("```json", "```") else 0

        # Find the end (last ```)
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break

        # Extract content between markers
        if start_idx < end_idx:
            return "\n".join(lines[start_idx:end_idx]).strip()

    return text


def _validate_schema(
    parsed_output: dict[str, Any],
    schema: dict[str, Any],
) -> ValidationResult:
    """Stage 2: Validate parsed JSON against a JSON schema.

    Uses Pydantic to validate the output against the provided JSON schema.

    Args:
        parsed_output: Parsed JSON dictionary.
        schema: JSON schema to validate against.

    Returns:
        ValidationResult with validation status.
    """
    try:
        # Convert JSON schema to Pydantic model
        model = _json_schema_to_pydantic(schema)

        # Validate the output
        model(**parsed_output)

        return ValidationResult(
            status=ValidationStatus.PASSED,
            passed=True,
            parsed_output=parsed_output,
        )
    except ValidationError as e:
        errors = []
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            errors.append(f"{loc}: {error['msg']}")

        return ValidationResult(
            status=ValidationStatus.FAILED_SCHEMA,
            passed=False,
            error_message="Schema validation failed:\n" + "\n".join(errors),
        )
    except Exception as e:
        return ValidationResult(
            status=ValidationStatus.FAILED_SCHEMA,
            passed=False,
            error_message=f"Schema validation error: {e}",
        )


def _compute_schema_hash(schema: dict[str, Any]) -> str:
    """Compute a deterministic hash for a JSON schema.

    Args:
        schema: JSON schema dictionary.

    Returns:
        SHA-256 hash of the schema.
    """
    schema_str = json.dumps(schema, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(schema_str.encode("utf-8")).hexdigest()


def _json_schema_to_pydantic(schema: dict[str, Any]) -> type[Any]:
    """Convert a JSON schema to a Pydantic model.

    Supports basic JSON schema types: string, integer, number, boolean,
    object, and array. Results are cached by schema hash to avoid
    repeated model creation.

    Args:
        schema: JSON schema dictionary.

    Returns:
        Dynamically created Pydantic model class.
    """
    # Check cache first
    schema_hash = _compute_schema_hash(schema)
    if schema_hash in _schema_model_cache:
        return _schema_model_cache[schema_hash]

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = _json_type_to_python(prop_schema)
        is_required = prop_name in required

        if is_required:
            field_definitions[prop_name] = (python_type, ...)
        else:
            field_definitions[prop_name] = (python_type | None, None)

    model = create_model("DynamicSchema", **field_definitions)

    # Cache the model
    _schema_model_cache[schema_hash] = model

    return model  # type: ignore[no-any-return]


def _json_type_to_python(prop_schema: dict[str, Any]) -> type[Any]:
    """Convert JSON schema type to Python type.

    Args:
        prop_schema: Property schema from JSON schema.

    Returns:
        Corresponding Python type.
    """
    json_type = prop_schema.get("type", "any")

    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }

    return type_mapping.get(json_type, object)


def _validate_equality(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> ValidationResult:
    """Stage 3: Compare actual output to expected using DeepDiff.

    Uses DeepDiff to compare the outputs, ignoring order in lists
    and dictionaries.

    Args:
        actual: Actual parsed output from LLM.
        expected: Expected output to compare against.

    Returns:
        ValidationResult with diff details if mismatch.
    """
    diff = DeepDiff(
        expected,
        actual,
        ignore_order=True,
        report_repetition=True,
        verbose_level=2,
    )

    if not diff:
        # Empty diff means perfect match
        return ValidationResult(
            status=ValidationStatus.PASSED,
            passed=True,
        )

    # Convert DeepDiff to a regular dict for serialization
    diff_dict = dict(diff)

    # Create human-readable error message
    error_parts = []

    if "values_changed" in diff_dict:
        for path, change in diff_dict["values_changed"].items():
            old_val = change.get("old_value", "N/A")
            new_val = change.get("new_value", "N/A")
            error_parts.append(f"  {path}: expected {old_val!r}, got {new_val!r}")

    if "dictionary_item_added" in diff_dict:
        for path in diff_dict["dictionary_item_added"]:
            error_parts.append(f"  {path}: unexpected field")

    if "dictionary_item_removed" in diff_dict:
        for path in diff_dict["dictionary_item_removed"]:
            error_parts.append(f"  {path}: missing field")

    if "type_changes" in diff_dict:
        for path, change in diff_dict["type_changes"].items():
            old_type = change.get("old_type", "N/A")
            new_type = change.get("new_type", "N/A")
            error_parts.append(f"  {path}: type changed from {old_type} to {new_type}")

    error_message = "Output does not match expected:\n" + "\n".join(error_parts)

    return ValidationResult(
        status=ValidationStatus.FAILED_EQUALITY,
        passed=False,
        error_message=error_message,
        diff_details=diff_dict,
    )


def format_diff_for_display(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> tuple[str, str]:
    """Format expected and actual outputs for side-by-side display.

    Args:
        expected: Expected output dictionary.
        actual: Actual output dictionary.

    Returns:
        Tuple of (formatted_expected, formatted_actual) strings.
    """
    expected_str = json.dumps(expected, indent=2, sort_keys=True)
    actual_str = json.dumps(actual, indent=2, sort_keys=True)
    return expected_str, actual_str


def load_validators_module(path: Path) -> dict[str, Callable[[str], Any]]:
    """Load validator functions from a python file.

    **Security:** This function performs AST validation before executing code
    to block dangerous operations like exec, eval, file access, and unsafe imports.
    Only whitelisted safe modules can be imported.

    Results are cached by file path and modification time to avoid reloading
    unchanged modules.

    Args:
        path: Path to the python file. Should be validated to be within an allowed
            directory before calling this function.

    Returns:
        Dictionary of validator name to function.

    Raises:
        ImportError: If the file cannot be loaded.
        ValidatorSecurityError: If the code contains unsafe operations.
    """
    # Create cache key from path and modification time
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    cache_key = f"{path}:{mtime}"

    # Check cache first
    if cache_key in _validator_module_cache:
        return _validator_module_cache[cache_key]

    # Read and validate the code before execution
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ImportError(f"Could not read validators from {path}: {e}") from e

    # Security: Validate code before execution
    _validate_validator_code(content, path)

    spec = importlib.util.spec_from_file_location("validators", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validators from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract all callable functions
    validators = {
        name: func
        for name, func in vars(module).items()
        if callable(func) and not name.startswith("_")
    }

    # Cache the result
    _validator_module_cache[cache_key] = validators

    return validators

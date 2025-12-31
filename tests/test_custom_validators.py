"""Tests for Custom Python Validators."""

from pathlib import Path

import pytest

from llm_bench.models import ValidationStatus
from llm_bench.validation import load_validators_module, validate_response


class TestCustomValidators:
    """Tests for custom validator functions."""

    def test_load_validators(self, tmp_path: Path) -> None:
        """Test loading validators from a file."""
        validator_file = tmp_path / "validators.py"
        validator_file.write_text("""
def check_length(output):
    return len(output) > 5

def check_bullet_points(output):
    return output.count("- ") >= 3
""")

        validators = load_validators_module(validator_file)
        assert "check_length" in validators
        assert "check_bullet_points" in validators
        assert callable(validators["check_length"])

    @pytest.mark.asyncio
    async def test_custom_validator_pass(self) -> None:
        """Test custom validator passing."""

        def check_upper(output):
            return output.isupper()

        result = await validate_response(
            "HELLO WORLD", expected=None, custom_validator=check_upper
        )

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_custom_validator_fail(self) -> None:
        """Test custom validator failing."""

        def check_upper(output):
            return output.isupper()

        result = await validate_response(
            "Hello World", expected=None, custom_validator=check_upper
        )

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_CUSTOM
        assert "Custom validation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_custom_validator_fail_message(self) -> None:
        """Test custom validator failing with message."""

        def check_upper(output):
            if not output.isupper():
                return False, "Not uppercase"
            return True

        result = await validate_response(
            "Hello World", expected=None, custom_validator=check_upper
        )

        assert result.passed is False
        assert result.error_message == "Not uppercase"

    @pytest.mark.asyncio
    async def test_custom_validator_exception(self) -> None:
        """Test custom validator raising exception."""

        def check_error(_output):
            raise ValueError("Boom")

        result = await validate_response(
            "test", expected=None, custom_validator=check_error
        )

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_CUSTOM
        assert "Validator raised exception" in result.error_message

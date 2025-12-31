"""Tests for Regex Validation."""

import pytest

from llm_bench.models import ValidationStatus
from llm_bench.validation import validate_response


class TestRegexValidation:
    """Tests for regex pattern matching."""

    @pytest.mark.asyncio
    async def test_regex_pass(self) -> None:
        """Test that matching regex passes."""
        output = "Order ID: 123-45-6789"
        pattern = r"\d{3}-\d{2}-\d{4}"

        # If expected is None, we skip JSON parse if regex passes
        result = await validate_response(
            output,
            expected=None,
            regex_pattern=pattern
        )

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_regex_fail(self) -> None:
        """Test that non-matching regex fails."""
        output = '{"id": "abc-de-fghi"}'
        pattern = r"\d{3}-\d{2}-\d{4}"

        result = await validate_response(
            output,
            expected=None,
            regex_pattern=pattern
        )

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_REGEX
        assert "did not match regex" in result.error_message

    @pytest.mark.asyncio
    async def test_regex_and_expected_pass(self) -> None:
        """Test regex pass + expected pass."""
        output = '{"code": "XYZ-123"}'
        pattern = r"XYZ-\d{3}"
        expected = {"code": "XYZ-123"}

        result = await validate_response(
            output,
            expected=expected,
            regex_pattern=pattern
        )

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_regex_pass_equality_fail(self) -> None:
        """Test regex pass but equality fail."""
        output = '{"code": "XYZ-999"}'
        pattern = r"XYZ-\d{3}"
        expected = {"code": "XYZ-123"}

        result = await validate_response(
            output,
            expected=expected,
            regex_pattern=pattern
        )

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_EQUALITY

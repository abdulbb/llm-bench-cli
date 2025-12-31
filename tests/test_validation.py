"Tests for validation module."

from unittest.mock import MagicMock, patch

import pytest

from llm_bench.llm import LLMResponse
from llm_bench.models import LatencyMetrics, TokenUsage, ValidationStatus
from llm_bench.validation import (
    _extract_json_from_markdown,
    _validate_equality,
    _validate_json_parse,
    _validate_schema,
    format_diff_for_display,
    validate_response,
)

# Note: Common fixtures like sample_latency, sample_token_usage, and
# sample_llm_response are now available from conftest.py


class TestValidateJsonParse:
    """Tests for JSON parse validation (Stage 1)."""

    def test_valid_json_object(self) -> None:
        """Test parsing valid JSON object."""
        result = _validate_json_parse('{"key": "value"}')
        assert result.passed is True
        assert result.status == ValidationStatus.PASSED
        assert result.parsed_output == {"key": "value"}

    def test_valid_json_with_whitespace(self) -> None:
        """Test parsing JSON with surrounding whitespace."""
        result = _validate_json_parse('  \n{"key": "value"}\n  ')
        assert result.passed is True
        assert result.parsed_output == {"key": "value"}

    def test_empty_output(self) -> None:
        """Test empty output fails."""
        result = _validate_json_parse("")
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE
        assert "Empty" in (result.error_message or "")

    def test_whitespace_only(self) -> None:
        """Test whitespace-only output fails."""
        result = _validate_json_parse("   \n\t  ")
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE

    def test_invalid_json(self) -> None:
        """Test invalid JSON fails."""
        result = _validate_json_parse('{"key": value}')  # Missing quotes
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE
        assert "Invalid JSON" in (result.error_message or "")

    def test_json_array_rejected(self) -> None:
        """Test JSON array is rejected (we expect objects)."""
        result = _validate_json_parse("[1, 2, 3]")
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE
        assert "Expected JSON object" in (result.error_message or "")

    def test_json_primitive_rejected(self) -> None:
        """Test JSON primitive is rejected."""
        result = _validate_json_parse('"just a string"')
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE

    def test_json_in_markdown_code_block(self) -> None:
        """Test extracting JSON from markdown code block."""
        raw = '```json\n{"key": "value"}\n```'
        result = _validate_json_parse(raw)
        assert result.passed is True
        assert result.parsed_output == {"key": "value"}

    def test_json_in_plain_code_block(self) -> None:
        """Test extracting JSON from plain code block."""
        raw = '```\n{"key": "value"}\n```'
        result = _validate_json_parse(raw)
        assert result.passed is True
        assert result.parsed_output == {"key": "value"}

    def test_nested_json(self) -> None:
        """Test parsing nested JSON."""
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = _validate_json_parse(raw)
        assert result.passed is True
        assert result.parsed_output == {"outer": {"inner": [1, 2, 3]}}


class TestExtractJsonFromMarkdown:
    """Tests for markdown extraction helper."""

    def test_no_markdown(self) -> None:
        """Test plain JSON is returned as-is."""
        result = _extract_json_from_markdown('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_code_block(self) -> None:
        """Test extracting from ```json block."""
        raw = '```json\n{"key": "value"}\n```'
        result = _extract_json_from_markdown(raw)
        assert result == '{"key": "value"}'

    def test_plain_code_block(self) -> None:
        """Test extracting from ``` block."""
        raw = '```\n{"key": "value"}\n```'
        result = _extract_json_from_markdown(raw)
        assert result == '{"key": "value"}'

    def test_multiline_json(self) -> None:
        """Test extracting multiline JSON."""
        raw = '```json\n{\n  "key": "value"\n}\n```'
        result = _extract_json_from_markdown(raw)
        assert result == '{\n  "key": "value"\n}'


class TestValidateSchema:
    """Tests for schema validation (Stage 2)."""

    def test_valid_against_schema(self) -> None:
        """Test output matching schema passes."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        output = {"name": "Alice", "age": 30}
        result = _validate_schema(output, schema)
        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    def test_missing_required_field(self) -> None:
        """Test missing required field fails."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        output = {"age": 30}  # Missing 'name'
        result = _validate_schema(output, schema)
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_SCHEMA
        assert "name" in (result.error_message or "").lower()

    def test_wrong_type(self) -> None:
        """Test wrong type fails."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        output = {"count": "not an integer"}
        result = _validate_schema(output, schema)
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_SCHEMA

    def test_optional_field_missing(self) -> None:
        """Test optional field can be missing."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        output = {"name": "Alice"}  # nickname is optional
        result = _validate_schema(output, schema)
        assert result.passed is True

    def test_extra_fields_allowed(self) -> None:
        """Test extra fields are allowed by default."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        output = {"name": "Alice", "extra": "field"}
        result = _validate_schema(output, schema)
        assert result.passed is True


class TestValidateEquality:
    """Tests for DeepDiff equality validation (Stage 3)."""

    def test_exact_match(self) -> None:
        """Test exact match passes."""
        expected = {"key": "value", "count": 42}
        actual = {"key": "value", "count": 42}
        result = _validate_equality(actual, expected)
        assert result.passed is True
        assert result.status == ValidationStatus.PASSED
        assert result.diff_details == {}

    def test_different_order_dict(self) -> None:
        """Test different key order is ignored."""
        expected = {"a": 1, "b": 2}
        actual = {"b": 2, "a": 1}
        result = _validate_equality(actual, expected)
        assert result.passed is True

    def test_different_order_list(self) -> None:
        """Test different list order is ignored."""
        expected = {"items": [1, 2, 3]}
        actual = {"items": [3, 1, 2]}
        result = _validate_equality(actual, expected)
        assert result.passed is True

    def test_value_mismatch(self) -> None:
        """Test value mismatch fails."""
        expected = {"key": "expected"}
        actual = {"key": "actual"}
        result = _validate_equality(actual, expected)
        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_EQUALITY
        assert "values_changed" in result.diff_details

    def test_missing_field(self) -> None:
        """Test missing field fails."""
        expected = {"a": 1, "b": 2}
        actual = {"a": 1}
        result = _validate_equality(actual, expected)
        assert result.passed is False
        assert "dictionary_item_removed" in result.diff_details

    def test_extra_field(self) -> None:
        """Test extra field fails."""
        expected = {"a": 1}
        actual = {"a": 1, "b": 2}
        result = _validate_equality(actual, expected)
        assert result.passed is False
        assert "dictionary_item_added" in result.diff_details

    def test_type_change(self) -> None:
        """Test type change fails."""
        expected = {"value": 42}
        actual = {"value": "42"}
        result = _validate_equality(actual, expected)
        assert result.passed is False
        assert "type_changes" in result.diff_details

    def test_nested_difference(self) -> None:
        """Test nested difference is detected."""
        expected = {"outer": {"inner": "expected"}}
        actual = {"outer": {"inner": "actual"}}
        result = _validate_equality(actual, expected)
        assert result.passed is False
        assert result.error_message is not None
        assert "inner" in result.error_message


class TestValidateResponse:
    """Tests for the full validation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_pass(self) -> None:
        """Test successful validation through all stages."""
        raw_output = '{"sentiment": "positive", "score": 0.95}'
        expected = {"sentiment": "positive", "score": 0.95}
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["sentiment", "score"],
        }

        result = await validate_response(raw_output, expected, schema)

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED
        assert result.parsed_output == expected

    @pytest.mark.asyncio
    async def test_pipeline_fails_at_json_parse(self) -> None:
        """Test pipeline fails at JSON parse stage."""
        raw_output = "not json"
        expected = {"key": "value"}

        result = await validate_response(raw_output, expected)

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_JSON_PARSE

    @pytest.mark.asyncio
    async def test_pipeline_fails_at_schema(self) -> None:
        """Test pipeline fails at schema validation stage."""
        raw_output = '{"wrong": "structure"}'
        expected = {"name": "value"}
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        result = await validate_response(raw_output, expected, schema)

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_SCHEMA

    @pytest.mark.asyncio
    async def test_pipeline_fails_at_equality(self) -> None:
        """Test pipeline fails at equality stage."""
        raw_output = '{"key": "wrong"}'
        expected = {"key": "expected"}

        result = await validate_response(raw_output, expected)

        assert result.passed is False
        assert result.status == ValidationStatus.FAILED_EQUALITY

    @pytest.mark.asyncio
    async def test_pipeline_without_schema(self) -> None:
        """Test pipeline works without schema."""
        raw_output = '{"key": "value"}'
        expected = {"key": "value"}

        result = await validate_response(raw_output, expected, schema=None)

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    @pytest.mark.asyncio
    async def test_pipeline_with_markdown_output(self) -> None:
        """Test pipeline handles markdown-wrapped JSON."""
        raw_output = '```json\n{"key": "value"}\n```'
        expected = {"key": "value"}

        result = await validate_response(raw_output, expected)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pipeline_without_expected(self) -> None:
        """Test pipeline works without expected output (freeform mode).

        When no validation criteria are provided (no expected, no schema,
        no regex, no custom validator), the pipeline passes immediately
        without parsing JSON - this is "freeform" mode for manual inspection.
        """
        raw_output = '{"key": "value"}'
        expected = None

        result = await validate_response(raw_output, expected)

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED
        # In freeform mode, parsed_output is None since we skip JSON parsing
        assert result.parsed_output is None

    @pytest.mark.asyncio
    async def test_pipeline_with_schema_but_no_expected(self) -> None:
        """Test pipeline validates schema even without expected output."""
        raw_output = '{"key": "value"}'
        expected = None
        schema = {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        }

        result = await validate_response(raw_output, expected, schema=schema)

        assert result.passed is True
        assert result.status == ValidationStatus.PASSED
        # With schema, JSON is parsed and validated
        assert result.parsed_output == {"key": "value"}


class TestFuzzyMatch:
    """Tests for fuzzy matching validation (Stage 4)."""

    # Using fixtures from conftest.py for mock responses
    @pytest.fixture
    def mock_judge_response_pass(
        self, sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
    ) -> LLMResponse:
        """Mock judge response saying PASS."""
        return LLMResponse(
            content="PASS",
            latency=sample_latency,
            token_usage=sample_token_usage,
            cost_usd=0.0001,
            model="judge-model",
        )

    @pytest.fixture
    def mock_judge_response_fail(
        self, sample_latency: LatencyMetrics, sample_token_usage: TokenUsage
    ) -> LLMResponse:
        """Mock judge response saying FAIL."""
        return LLMResponse(
            content="FAIL",
            latency=sample_latency,
            token_usage=sample_token_usage,
            cost_usd=0.0001,
            model="judge-model",
        )

    @pytest.mark.asyncio
    async def test_fuzzy_match_triggered_on_equality_failure(
        self, mock_judge_response_pass: LLMResponse
    ) -> None:
        """Test that fuzzy match is triggered when equality fails."""
        raw_output = '{"key": "slightly different"}'
        expected = {"key": "value"}

        with patch(
            "llm_bench.validation.call_llm", return_value=mock_judge_response_pass
        ) as mock_call:
            result = await validate_response(
                raw_output, expected, judge_model="gpt-3.5-turbo"
            )

            assert result.passed is True
            assert result.status == ValidationStatus.PASSED
            assert result.used_fuzzy_match is True
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_fuzzy_match_fails(
        self, mock_judge_response_fail: LLMResponse
    ) -> None:
        """Test that fuzzy match failure results in failed status."""
        raw_output = '{"key": "completely different"}'
        expected = {"key": "value"}

        with patch(
            "llm_bench.validation.call_llm", return_value=mock_judge_response_fail
        ):
            result = await validate_response(
                raw_output, expected, judge_model="gpt-3.5-turbo"
            )

            assert result.passed is False
            assert result.status == ValidationStatus.FAILED_FUZZY
            assert result.used_fuzzy_match is False
            assert "Fuzzy match" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_fuzzy_match_skipped_if_no_judge(self) -> None:
        """Test that fuzzy match is skipped if judge_model is None."""
        raw_output = '{"key": "different"}'
        expected = {"key": "value"}

        with patch("llm_bench.validation.call_llm") as mock_call:
            result = await validate_response(raw_output, expected, judge_model=None)

            assert result.passed is False
            assert result.status == ValidationStatus.FAILED_EQUALITY
            mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_fuzzy_match_uses_cache(
        self, mock_judge_response_pass: LLMResponse
    ) -> None:
        """Test that fuzzy match uses cache."""
        raw_output = '{"key": "val"}'
        expected = {"key": "value"}

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.get.return_value = mock_judge_response_pass

        with patch("llm_bench.validation.call_llm") as mock_call:
            result = await validate_response(
                raw_output, expected, judge_model="gpt-3.5-turbo", cache=mock_cache
            )

            assert result.passed is True
            mock_cache.get.assert_called_once()
            mock_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_fuzzy_match_sets_cache(
        self, mock_judge_response_pass: LLMResponse
    ) -> None:
        """Test that fuzzy match sets cache on miss."""
        raw_output = '{"key": "val"}'
        expected = {"key": "value"}

        # Mock cache miss
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch(
            "llm_bench.validation.call_llm", return_value=mock_judge_response_pass
        ):
            await validate_response(
                raw_output, expected, judge_model="gpt-3.5-turbo", cache=mock_cache
            )

            mock_cache.set.assert_called_once()


class TestFormatDiffForDisplay:
    """Tests for diff formatting helper."""

    def test_format_diff(self) -> None:
        """Test formatting diff for display."""
        expected = {"b": 2, "a": 1}
        actual = {"a": 1, "b": 2}

        expected_str, actual_str = format_diff_for_display(expected, actual)

        # Both should be valid JSON with sorted keys
        assert '"a": 1' in expected_str
        assert '"b": 2' in expected_str
        assert '"a": 1' in actual_str
        assert '"b": 2' in actual_str

"""Tests for NLP Metrics."""

from llm_bench.metrics import calculate_rouge_l


class TestMetrics:
    """Tests for metric calculations."""

    def test_rouge_l_exact_match(self) -> None:
        """Test ROUGE-L with identical strings."""
        s = "the quick brown fox"
        assert calculate_rouge_l(s, s) == 1.0

    def test_rouge_l_no_match(self) -> None:
        """Test ROUGE-L with no overlap."""
        assert calculate_rouge_l("abc", "xyz") == 0.0

    def test_rouge_l_partial_match(self) -> None:
        """Test ROUGE-L with partial overlap."""
        # LCS: "quick brown" (2 words)
        # Cand: 4 words, Ref: 4 words.
        # Prec: 2/4 = 0.5, Rec: 2/4 = 0.5 -> F1 = 0.5
        cand = "the quick brown fox"
        ref = "a quick brown dog"
        score = calculate_rouge_l(cand, ref)
        assert abs(score - 0.5) < 0.001

    def test_rouge_l_empty(self) -> None:
        """Test ROUGE-L with empty inputs."""
        assert calculate_rouge_l("", "ref") == 0.0
        assert calculate_rouge_l("cand", "") == 0.0

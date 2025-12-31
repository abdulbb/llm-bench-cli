"""NLP Metrics for text evaluation."""

def calculate_rouge_l(candidate: str, reference: str) -> float:
    """Calculate ROUGE-L score (Longest Common Subsequence).

    Returns:
        F1 score between 0.0 and 1.0.
    """
    if not candidate or not reference:
        return 0.0

    # Simple tokenization by whitespace
    cand_tokens = candidate.strip().split()
    ref_tokens = reference.strip().split()

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(cand_tokens, ref_tokens)

    # ROUGE-L stats
    prec = lcs / len(cand_tokens) if cand_tokens else 0.0
    rec = lcs / len(ref_tokens) if ref_tokens else 0.0

    if prec + rec == 0:
        return 0.0

    f1 = 2 * ((prec * rec) / (prec + rec))
    return f1


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Compute the length of the LCS of two sequences."""
    m = len(x)
    n = len(y)

    # We only need two rows to save memory
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = list(curr)

    return curr[n]

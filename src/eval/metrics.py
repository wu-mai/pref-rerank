"""
Evaluation metrics for personalized evidence reranking.

All metrics are session-level: we compute per-session scores, then average across
sessions. Statistical significance via paired bootstrap 95% CI.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Per-session metrics
# ---------------------------------------------------------------------------


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain @ k."""
    return sum(
        gain / math.log2(i + 2)  # rank i -> position i+1 -> log2(i+2)
        for i, gain in enumerate(gains[:k])
    )


def ndcg_at_k(ranked_relevance: Sequence[float], k: int = 10) -> float:
    """nDCG@k for one session: DCG@k / iDCG@k.

    ranked_relevance: relevance scores sorted by the model's predicted ranking.
    Ideal ranking = sorted(ranked_relevance, reverse=True).
    """
    if not ranked_relevance:
        return 0.0
    dcg = dcg_at_k(ranked_relevance, k)
    ideal = sorted(ranked_relevance, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(ranked_relevance: Sequence[float]) -> float:
    """Reciprocal rank of the first relevant doc."""
    for i, rel in enumerate(ranked_relevance):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_relevance: Sequence[float], total_relevant: int, k: int = 10) -> float:
    """Recall@k = (# relevant in top-k) / total_relevant."""
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for rel in ranked_relevance[:k] if rel > 0)
    return hits / total_relevant


# ---------------------------------------------------------------------------
# Aggregated session-level metrics with paired bootstrap CI
# ---------------------------------------------------------------------------


@dataclass
class SessionEval:
    """Result of evaluating one session under one system."""

    session_id: str
    ndcg10: float
    mrr_val: float
    recall10: float
    recall20: float


def aggregate_session_evals(evals: list[SessionEval]) -> dict[str, float]:
    """Mean across sessions."""
    if not evals:
        return {"n_sessions": 0, "ndcg10": 0.0, "mrr": 0.0, "recall10": 0.0, "recall20": 0.0}
    return {
        "n_sessions": len(evals),
        "ndcg10": float(np.mean([e.ndcg10 for e in evals])),
        "mrr": float(np.mean([e.mrr_val for e in evals])),
        "recall10": float(np.mean([e.recall10 for e in evals])),
        "recall20": float(np.mean([e.recall20 for e in evals])),
    }


def paired_bootstrap_ci(
    system_a: list[float],
    system_b: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on the per-session difference.

    Returns (mean_diff, ci_low, ci_high). If ci_low > 0 or ci_high < 0, the
    difference is statistically significant at the (1 - ci) level.
    """
    assert len(system_a) == len(system_b), "Paired comparison requires equal-length lists"
    if not system_a:
        return 0.0, 0.0, 0.0

    rng = random.Random(seed)
    n = len(system_a)
    diffs = [a - b for a, b in zip(system_a, system_b)]
    mean_diff = sum(diffs) / n

    boot_means: list[float] = []
    indices = list(range(n))
    for _ in range(n_bootstrap):
        sample_sum = 0.0
        for _ in range(n):
            idx = rng.choice(indices)
            sample_sum += diffs[idx]
        boot_means.append(sample_sum / n)

    boot_means.sort()
    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap) - 1
    return mean_diff, boot_means[lo_idx], boot_means[hi_idx]


def stratified_eval(
    evals: list[SessionEval],
    high_rank_session_ids: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Split session evals into high-rank vs low-rank strata.

    high_rank_session_ids: sessions whose eval click was in top-3 SERP.
    """
    high = [e for e in evals if e.session_id in high_rank_session_ids]
    low = [e for e in evals if e.session_id not in high_rank_session_ids]
    return aggregate_session_evals(high), aggregate_session_evals(low)


# ---------------------------------------------------------------------------
# Cohen's Kappa (for LLM schema validation study, Block 7)
# ---------------------------------------------------------------------------


def cohen_kappa(annotator1: Sequence[int], annotator2: Sequence[int]) -> float:
    """Cohen's Kappa between two annotators (integer labels per item).

    Returns Kappa in [-1, 1]; ≥0.7 = substantial agreement.
    """
    assert len(annotator1) == len(annotator2)
    n = len(annotator1)
    if n == 0:
        return 0.0
    labels = sorted(set(annotator1) | set(annotator2))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)

    # Observed agreement
    obs = sum(1 for a, b in zip(annotator1, annotator2) if a == b) / n

    # Expected agreement (from marginal distributions)
    marg1 = np.zeros(k)
    marg2 = np.zeros(k)
    for a, b in zip(annotator1, annotator2):
        marg1[label_to_idx[a]] += 1
        marg2[label_to_idx[b]] += 1
    marg1 /= n
    marg2 /= n
    exp = float(np.sum(marg1 * marg2))

    if exp == 1.0:
        return 1.0
    return (obs - exp) / (1 - exp)

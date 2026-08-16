"""Rank stability under perturbed criterion weights.

Outranking methods are known to be sensitive to their weight vector, and a
ranking derived from expert judgement inherits the imprecision of that
judgement. This module re-runs the outranking many times under perturbed
weights and reports how often each alternative holds each rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StabilityReport", "rank_stability"]


@dataclass
class StabilityReport:
    """Distribution of ranks across simulations.

    Attributes
    ----------
    ranks : ndarray of shape (n_simulations, n_alternatives)
        Rank of each alternative in each simulation; 1 is best.
    rank_counts : ndarray of shape (n_alternatives, n_alternatives)
        ``rank_counts[a, r]`` counts how often alternative ``a`` took rank
        ``r + 1``.
    mean_rank : ndarray of shape (n_alternatives,)
    top1_frequency : ndarray of shape (n_alternatives,)
        Proportion of simulations in which each alternative ranked first.
    rank_reversal_rate : float
        Proportion of simulations whose full ordering differs from the
        unperturbed one.
    alternatives : list of str or None
    """

    ranks: np.ndarray
    rank_counts: np.ndarray
    mean_rank: np.ndarray
    top1_frequency: np.ndarray
    rank_reversal_rate: float
    alternatives: list | None = None

    def summary(self) -> str:
        """Return a plain-text table of the stability results."""
        n = self.mean_rank.size
        labels = self.alternatives or [f"A{i + 1}" for i in range(n)]
        width = max(len(str(x)) for x in labels)

        lines = [
            (
                f"rank reversal rate: {self.rank_reversal_rate:.1%} "
                f"of {self.ranks.shape[0]} simulations"
            ),
            "",
            f"{'alternative':<{width}}  mean rank  P(rank 1)  modal rank",
        ]
        for i, label in enumerate(labels):
            modal = int(np.argmax(self.rank_counts[i])) + 1
            lines.append(
                f"{label!s:<{width}}  {self.mean_rank[i]:>9.2f}  "
                f"{self.top1_frequency[i]:>9.1%}  {modal:>10d}"
            )
        return "\n".join(lines)


def rank_stability(
    decision_matrix,
    weights,
    ranker,
    n: int = 1000,
    scale: float = 0.10,
    method: str = "normal",
    random_state=None,
    alternatives=None,
) -> StabilityReport:
    """Re-rank under perturbed weights and summarise the outcome.

    Parameters
    ----------
    decision_matrix : array-like, shape (n_alternatives, n_criteria)
    weights : array-like, shape (n_criteria,)
        The unperturbed weights.
    ranker : Promethee
        The configured outranking object; reused unchanged each simulation.
    n : int
        Number of simulations.
    scale : float
        Perturbation magnitude. Under ``"normal"`` it is the relative
        standard deviation applied to each weight; under ``"dirichlet"`` it
        controls the concentration, with smaller values giving tighter
        samples around the nominal weights.
    method : {"normal", "dirichlet"}
        ``"normal"`` perturbs each weight independently and renormalises;
        ``"dirichlet"`` samples from the simplex directly, which respects the
        sum-to-one constraint without truncation.
    random_state : int or numpy.random.Generator, optional
    alternatives : sequence of str, optional
        Labels carried through to the report.
    """
    X = np.asarray(decision_matrix, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if w.sum() <= 0:
        raise ValueError("weights must not sum to zero")
    w = w / w.sum()
    if scale <= 0:
        raise ValueError("scale must be positive")

    rng = np.random.default_rng(random_state)
    n_alt = X.shape[0]

    baseline_order = tuple(ranker.rank(X, w).order.tolist())

    ranks = np.empty((n, n_alt), dtype=int)
    reversals = 0

    for sim in range(n):
        w_sim = _perturb(w, scale, method, rng)
        order = ranker.rank(X, w_sim).order
        if tuple(order.tolist()) != baseline_order:
            reversals += 1
        # order lists alternative indices best-to-worst; invert to get ranks.
        ranks[sim, order] = np.arange(1, n_alt + 1)

    rank_counts = np.zeros((n_alt, n_alt), dtype=int)
    for a in range(n_alt):
        counts = np.bincount(ranks[:, a] - 1, minlength=n_alt)
        rank_counts[a] = counts

    return StabilityReport(
        ranks=ranks,
        rank_counts=rank_counts,
        mean_rank=ranks.mean(axis=0),
        top1_frequency=rank_counts[:, 0] / float(n),
        rank_reversal_rate=reversals / float(n),
        alternatives=list(alternatives) if alternatives is not None else None,
    )


def _perturb(w, scale, method, rng):
    if method == "normal":
        noise = rng.normal(loc=1.0, scale=scale, size=w.size)
        w_sim = np.clip(w * noise, 1e-12, None)
        return w_sim / w_sim.sum()

    if method == "dirichlet":
        concentration = max((1.0 - scale**2) / (scale**2), 1e-6)
        alpha = np.clip(w * concentration, 1e-6, None)
        return rng.dirichlet(alpha)

    raise ValueError(f"unknown perturbation method {method!r}")

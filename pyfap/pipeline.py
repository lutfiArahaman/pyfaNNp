"""The coupled pipeline: fuzzy weights, neural surrogate, outranking.

``DecisionPipeline`` composes the three stages so that a complete analysis is
one call and one object. The stages remain usable on their own; nothing here
is required in order to use :class:`~pyfap.fahp.FAHP`,
:class:`~pyfap.ann.ANNSurrogate` or :class:`~pyfap.promethee.Promethee`
independently.
"""

from __future__ import annotations

import numpy as np

from .promethee import Promethee
from .sensitivity import rank_stability

__all__ = ["DecisionPipeline", "DecisionResult"]


class DecisionResult:
    """Outcome of a fitted pipeline.

    Attributes
    ----------
    weights : ndarray of shape (n_criteria,)
    consistency_ratio : float or None
    positive_flow, negative_flow, net_flow : ndarray of shape (n_alternatives,)
    ranking : list
        Alternative labels, best to worst.
    surrogate_score : float or None
        Training coefficient of determination of the neural surrogate.
    """

    def __init__(
        self,
        weights,
        flows,
        alternatives,
        criteria,
        decision_matrix,
        ranker,
        consistency_ratio=None,
        surrogate=None,
    ):
        self.weights = weights
        self.consistency_ratio = consistency_ratio
        self.alternatives = list(alternatives)
        self.criteria = list(criteria)

        self.positive_flow = flows.positive_flow
        self.negative_flow = flows.negative_flow
        self.net_flow = flows.net_flow
        self.preference_index = flows.preference_index

        self._flows = flows
        self._X = decision_matrix
        self._ranker = ranker
        self._surrogate = surrogate

    # ---------------------------------------------------------------- views

    @property
    def ranking(self) -> list:
        """Alternative labels ordered best to worst by net flow."""
        return [self.alternatives[i] for i in self._flows.order]

    @property
    def partial_order(self) -> np.ndarray:
        """PROMETHEE I relation matrix; see :meth:`Promethee.partial_order`."""
        return Promethee.partial_order(self._flows)

    @property
    def surrogate_score(self):
        return None if self._surrogate is None else self._surrogate.train_score_

    def to_frame(self):
        """Return the flows as a pandas DataFrame, if pandas is installed."""
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "to_frame() requires pandas; install it with `pip install pandas`"
            ) from exc
        return pd.DataFrame(
            {
                "phi_plus": self.positive_flow,
                "phi_minus": self.negative_flow,
                "phi": self.net_flow,
            },
            index=self.alternatives,
        ).sort_values("phi", ascending=False)

    # -------------------------------------------------------------- methods

    def stability(self, n: int = 1000, scale: float = 0.10, **kwargs):
        """Rank distribution under perturbed weights.

        See :func:`pyfap.sensitivity.rank_stability` for the full signature.
        """
        return rank_stability(
            self._X,
            self.weights,
            self._ranker,
            n=n,
            scale=scale,
            alternatives=self.alternatives,
            **kwargs,
        )

    def predict(self, decision_matrix) -> np.ndarray:
        """Score new alternatives with the fitted neural surrogate."""
        if self._surrogate is None:
            raise RuntimeError(
                "this pipeline was built without a surrogate; pass one to "
                "DecisionPipeline(surrogate=...) to enable predict()"
            )
        return self._surrogate.predict(decision_matrix)

    def __repr__(self) -> str:
        best = self.ranking[0] if self.alternatives else "?"
        cr = (
            "n/a"
            if self.consistency_ratio is None
            else f"{self.consistency_ratio:.3f}"
        )
        return (
            f"<DecisionResult n_alternatives={len(self.alternatives)} "
            f"n_criteria={len(self.criteria)} CR={cr} best={best!r}>"
        )


class DecisionPipeline:
    """Chain fuzzy weight derivation, a neural surrogate and outranking.

    Parameters
    ----------
    weights : FAHP
        Weight-derivation stage. Any object exposing ``derive(judgments)``
        and, optionally, ``consistency_ratio_`` is accepted.
    surrogate : ANNSurrogate, optional
        Trained on the exact flows during :meth:`fit`. If omitted, the
        pipeline is a plain fuzzy AHP and PROMETHEE analysis, which is the
        baseline the coupled pipeline is compared against.
    ranker : Promethee, optional
        Defaults to ``Promethee()``.
    """

    def __init__(self, weights, surrogate=None, ranker=None):
        if not hasattr(weights, "derive"):
            raise TypeError("weights must expose a derive(judgments) method")
        self.weights = weights
        self.surrogate = surrogate
        self.ranker = ranker if ranker is not None else Promethee()

        self._fitted = False
        self._weights_ = None
        self._flows_ = None
        self._X_ = None
        self._alternatives_ = None
        self._criteria_ = None

    # ------------------------------------------------------------------ API

    def fit(
        self,
        judgments,
        decision_matrix,
        alternatives=None,
        criteria=None,
    ) -> DecisionPipeline:
        """Derive weights, compute exact flows and train the surrogate.

        Parameters
        ----------
        judgments : array-like, shape (n_criteria, n_criteria, 3)
        decision_matrix : array-like, shape (n_alternatives, n_criteria)
        alternatives, criteria : sequence of str, optional
            Labels; positional defaults are generated when omitted.
        """
        X = np.asarray(decision_matrix, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"decision_matrix must be 2-D; got shape {X.shape}")
        n_alt, n_crit = X.shape

        J = np.asarray(judgments, dtype=float)
        if J.shape[0] != n_crit:
            raise ValueError(
                f"judgments describes {J.shape[0]} criteria but the decision "
                f"matrix has {n_crit} columns"
            )

        self._alternatives_ = _labels(alternatives, n_alt, "A")
        self._criteria_ = _labels(criteria, n_crit, "C")

        self._weights_ = self.weights.derive(J)
        self._flows_ = self.ranker.rank(X, self._weights_)
        self._X_ = X

        if self.surrogate is not None:
            self.surrogate.fit(X, self._flows_.net_flow)

        self._fitted = True
        return self

    def rank(self) -> DecisionResult:
        """Return the :class:`DecisionResult` for the fitted problem."""
        if not self._fitted:
            raise RuntimeError("call fit() before rank()")
        return DecisionResult(
            weights=self._weights_,
            flows=self._flows_,
            alternatives=self._alternatives_,
            criteria=self._criteria_,
            decision_matrix=self._X_,
            ranker=self.ranker,
            consistency_ratio=getattr(self.weights, "consistency_ratio_", None),
            surrogate=self.surrogate,
        )

    def fit_rank(self, judgments, decision_matrix, **kwargs) -> DecisionResult:
        """Convenience for ``fit(...).rank()``."""
        return self.fit(judgments, decision_matrix, **kwargs).rank()


def _labels(given, n, prefix):
    if given is None:
        return [f"{prefix}{i + 1}" for i in range(n)]
    seq = list(given)
    if len(seq) != n:
        raise ValueError(f"expected {n} labels, got {len(seq)}")
    return seq

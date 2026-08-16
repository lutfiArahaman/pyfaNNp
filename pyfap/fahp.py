"""Fuzzy analytic hierarchy process: criterion weights from fuzzy judgements.

Two aggregation methods are provided:

``extent_analysis``
    Chang (1996). Computes fuzzy synthetic extents, then the degree of
    possibility that each extent exceeds every other, and takes the minimum.
    Widely used, but note the known criticism that it can assign a weight of
    exactly zero to a criterion that carries real information; see
    ``FAHP.derive`` for the warning this raises.

``geometric_mean``
    Buckley (1985). Takes the fuzzy geometric mean of each row, normalises
    within the fuzzy domain, then defuzzifies. Slower to state but does not
    produce zero weights.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._fuzzy import SAATY_RI, as_tfn_matrix, defuzzify

__all__ = ["FAHP"]


class FAHP:
    """Derive crisp criterion weights from a fuzzy pairwise comparison matrix.

    Parameters
    ----------
    method : {"extent_analysis", "geometric_mean"}
        Aggregation method; see the module docstring.
    consistency_check : bool
        If True, compute the consistency ratio of the defuzzified matrix and
        warn when it exceeds ``consistency_threshold``.
    consistency_threshold : float
        Conventional acceptance limit for the consistency ratio.
    defuzzify_method : {"centroid", "graded_mean"}
        Used both for the consistency check and, under ``geometric_mean``,
        to collapse the fuzzy weights.

    Attributes set by :meth:`derive`
    --------------------------------
    weights_ : ndarray of shape (n_criteria,)
    consistency_ratio_ : float or None
    consistency_index_ : float or None
    lambda_max_ : float or None
    """

    def __init__(
        self,
        method: str = "extent_analysis",
        consistency_check: bool = True,
        consistency_threshold: float = 0.10,
        defuzzify_method: str = "centroid",
    ):
        if method not in ("extent_analysis", "geometric_mean"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.consistency_check = consistency_check
        self.consistency_threshold = consistency_threshold
        self.defuzzify_method = defuzzify_method

        self.weights_ = None
        self.consistency_ratio_ = None
        self.consistency_index_ = None
        self.lambda_max_ = None

    # ------------------------------------------------------------------ API

    def derive(self, judgments) -> np.ndarray:
        """Return normalised crisp weights for a fuzzy comparison matrix."""
        J = as_tfn_matrix(judgments)

        if self.consistency_check:
            self._check_consistency(J)

        if self.method == "extent_analysis":
            w = self._extent_analysis(J)
        else:
            w = self._geometric_mean(J)

        if np.any(w <= 0):
            warnings.warn(
                "at least one criterion received a weight of zero. This is a "
                "known artefact of extent analysis, not necessarily a "
                "statement that the criterion is irrelevant; consider "
                "method='geometric_mean'.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.weights_ = w
        return w

    # ------------------------------------------------------------ internals

    @staticmethod
    def _extent_analysis(J: np.ndarray) -> np.ndarray:
        n = J.shape[0]

        # Fuzzy synthetic extent per row. The inverse of a TFN reverses the
        # bounds, hence the index crossing below.
        row = J.sum(axis=1)          # (n, 3)
        total = row.sum(axis=0)      # (3,)
        S = np.empty((n, 3))
        S[:, 0] = row[:, 0] / total[2]
        S[:, 1] = row[:, 1] / total[1]
        S[:, 2] = row[:, 2] / total[0]

        # Degree of possibility V(S_i >= S_j) for every ordered pair.
        d = np.ones(n)
        for i in range(n):
            v_min = 1.0
            for j in range(n):
                if i == j:
                    continue
                v_min = min(v_min, _degree_of_possibility(S[i], S[j]))
            d[i] = v_min

        if d.sum() <= 0:
            raise ValueError(
                "all degrees of possibility are zero; the judgement matrix "
                "carries no discriminating information"
            )
        return d / d.sum()

    def _geometric_mean(self, J: np.ndarray) -> np.ndarray:
        n = J.shape[0]

        # Componentwise fuzzy geometric mean of each row.
        r = np.exp(np.log(J).mean(axis=1))   # (n, 3), == prod(J, axis=1)**(1/n)

        rsum = r.sum(axis=0)                 # (3,)
        w_fuzzy = np.empty((n, 3))
        w_fuzzy[:, 0] = r[:, 0] / rsum[2]
        w_fuzzy[:, 1] = r[:, 1] / rsum[1]
        w_fuzzy[:, 2] = r[:, 2] / rsum[0]

        w = defuzzify(w_fuzzy, method=self.defuzzify_method)
        return w / w.sum()

    def _check_consistency(self, J: np.ndarray) -> None:
        n = J.shape[0]
        crisp = defuzzify(J, method=self.defuzzify_method)

        eigenvalues = np.linalg.eigvals(crisp)
        lam = float(np.max(eigenvalues.real))
        self.lambda_max_ = lam

        if n < 3:
            self.consistency_index_ = 0.0
            self.consistency_ratio_ = 0.0
            return

        ci = (lam - n) / (n - 1)
        ri = SAATY_RI.get(n)
        self.consistency_index_ = float(ci)

        if ri is None or ri == 0:
            self.consistency_ratio_ = None
            warnings.warn(
                f"no random consistency index tabulated for n={n}; "
                "consistency ratio not computed",
                RuntimeWarning,
                stacklevel=3,
            )
            return

        cr = float(ci / ri)
        self.consistency_ratio_ = cr
        if cr > self.consistency_threshold:
            warnings.warn(
                f"consistency ratio {cr:.3f} exceeds the conventional "
                f"threshold of {self.consistency_threshold:.2f}; the "
                "judgements should be revisited before the weights are used",
                RuntimeWarning,
                stacklevel=3,
            )


def _degree_of_possibility(si: np.ndarray, sj: np.ndarray) -> float:
    """V(S_i >= S_j) for triangular fuzzy numbers ``si``, ``sj``."""
    li, mi, ui = si
    lj, mj, uj = sj

    if mi >= mj:
        return 1.0
    if lj >= ui:
        return 0.0

    denominator = (mi - ui) - (mj - lj)
    if denominator == 0:
        return 0.0
    return float((lj - ui) / denominator)

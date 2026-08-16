"""PROMETHEE outranking.

Implements the six preference functions of Brans and Vincke (1985), the
positive, negative and net flows, the PROMETHEE II complete ranking and the
PROMETHEE I partial relation.

Naming note
-----------
Brans' type 3 (``v-shape``) uses only the preference threshold ``p``, and his
type 5 (``linear``, or "V-shape with indifference") uses both ``q`` and ``p``.
Here ``v-shape`` accepts both and reduces to type 3 exactly when ``q == 0``,
so a single name covers both cases; ``linear`` is an alias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Promethee", "PrometheeFlows", "PREFERENCE_FUNCTIONS"]

PREFERENCE_FUNCTIONS = (
    "usual",
    "u-shape",
    "v-shape",
    "level",
    "linear",
    "gaussian",
)


@dataclass
class PrometheeFlows:
    """Result of an outranking."""

    positive_flow: np.ndarray
    negative_flow: np.ndarray
    net_flow: np.ndarray
    preference_index: np.ndarray

    @property
    def order(self) -> np.ndarray:
        """Indices of alternatives, best to worst by net flow."""
        return np.argsort(-self.net_flow, kind="stable")


class Promethee:
    """PROMETHEE I and II outranking.

    Parameters
    ----------
    version : {"I", "II"}
        Retained for reporting; both flow sets are always computed.
    preference : str or sequence of str
        One of :data:`PREFERENCE_FUNCTIONS`, or one name per criterion.
    q, p, s : float or sequence of float
        Indifference threshold, preference threshold and Gaussian parameter.
        Scalars are broadcast across criteria.
    criteria_types : sequence, optional
        ``+1`` / ``"max"`` for criteria to maximise, ``-1`` / ``"min"`` for
        criteria to minimise. Defaults to maximising every criterion.
    """

    def __init__(
        self,
        version: str = "II",
        preference: str = "v-shape",
        q=0.0,
        p=1.0,
        s=None,
        criteria_types=None,
    ):
        if version not in ("I", "II"):
            raise ValueError(f"version must be 'I' or 'II'; got {version!r}")
        self.version = version
        self.preference = preference
        self.q = q
        self.p = p
        self.s = s
        self.criteria_types = criteria_types

    # ------------------------------------------------------------------ API

    def rank(self, decision_matrix, weights) -> PrometheeFlows:
        """Compute outranking flows for a decision matrix.

        Parameters
        ----------
        decision_matrix : array-like, shape (n_alternatives, n_criteria)
        weights : array-like, shape (n_criteria,)
            Need not be normalised; they are normalised internally.
        """
        X = np.asarray(decision_matrix, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"decision_matrix must be 2-D; got shape {X.shape}")
        n_alt, n_crit = X.shape
        if n_alt < 2:
            raise ValueError("at least two alternatives are required")

        w = np.asarray(weights, dtype=float).ravel()
        if w.size != n_crit:
            raise ValueError(
                f"weights has length {w.size} but the decision matrix has "
                f"{n_crit} criteria"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if w.sum() <= 0:
            raise ValueError("weights must not sum to zero")
        w = w / w.sum()

        kinds = _broadcast_str(self.preference, n_crit, "preference")
        qs = _broadcast_num(self.q, n_crit, "q")
        ps = _broadcast_num(self.p, n_crit, "p")
        ss = _broadcast_num(self.s, n_crit, "s", allow_none=True)
        types = _criteria_types(self.criteria_types, n_crit)

        # Aggregated preference index pi[a, b].
        pi = np.zeros((n_alt, n_alt), dtype=float)
        for k in range(n_crit):
            col = X[:, k] * types[k]
            deviation = col[:, None] - col[None, :]
            pi += w[k] * _preference(
                deviation, kinds[k], qs[k], ps[k], ss[k], criterion=k
            )
        np.fill_diagonal(pi, 0.0)

        denominator = n_alt - 1
        phi_plus = pi.sum(axis=1) / denominator
        phi_minus = pi.sum(axis=0) / denominator

        return PrometheeFlows(
            positive_flow=phi_plus,
            negative_flow=phi_minus,
            net_flow=phi_plus - phi_minus,
            preference_index=pi,
        )

    @staticmethod
    def partial_order(flows: PrometheeFlows, tol: float = 1e-12) -> np.ndarray:
        """PROMETHEE I relation matrix.

        Returns an ``(n, n)`` array of strings where entry ``[a, b]`` is
        ``"P"`` if ``a`` outranks ``b``, ``"I"`` if they are indifferent,
        ``"R"`` if they are incomparable, and ``""`` on the diagonal.
        """
        plus = flows.positive_flow
        minus = flows.negative_flow
        n = plus.size

        relation = np.full((n, n), "", dtype=object)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                dp = plus[a] - plus[b]
                dm = minus[a] - minus[b]
                plus_ge = dp > -tol
                minus_le = dm < tol
                strict = abs(dp) > tol or abs(dm) > tol

                if plus_ge and minus_le and strict:
                    relation[a, b] = "P"
                elif abs(dp) <= tol and abs(dm) <= tol:
                    relation[a, b] = "I"
                elif (dp < -tol and dm < -tol) or (dp > tol and dm > tol):
                    relation[a, b] = "R"
                else:
                    relation[a, b] = "R" if not (plus_ge and minus_le) else "P"
        return relation


# ---------------------------------------------------------------- internals


def _preference(d, kind, q, p, s, criterion=None):
    """Vectorised preference function applied to a deviation matrix."""
    where = f" for criterion {criterion}" if criterion is not None else ""

    if kind == "usual":
        return (d > 0).astype(float)

    if kind == "u-shape":
        return (d > q).astype(float)

    if kind in ("v-shape", "linear"):
        if p <= q:
            raise ValueError(
                f"p must be strictly greater than q{where}; got q={q}, p={p}"
            )
        out = np.zeros_like(d, dtype=float)
        mid = (d > q) & (d <= p)
        out[mid] = (d[mid] - q) / (p - q)
        out[d > p] = 1.0
        return out

    if kind == "level":
        if p < q:
            raise ValueError(f"p must be at least q{where}; got q={q}, p={p}")
        out = np.zeros_like(d, dtype=float)
        out[(d > q) & (d <= p)] = 0.5
        out[d > p] = 1.0
        return out

    if kind == "gaussian":
        if s is None or s <= 0:
            raise ValueError(
                f"the gaussian preference function requires s > 0{where}"
            )
        out = np.zeros_like(d, dtype=float)
        pos = d > 0
        out[pos] = 1.0 - np.exp(-(d[pos] ** 2) / (2.0 * s**2))
        return out

    raise ValueError(
        f"unknown preference function {kind!r}{where}; "
        f"expected one of {PREFERENCE_FUNCTIONS}"
    )


def _broadcast_num(value, n, name, allow_none=False):
    if value is None:
        if allow_none:
            return [None] * n
        raise ValueError(f"{name} must not be None")
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return [float(arr[0])] * n
    if arr.size != n:
        raise ValueError(f"{name} must be a scalar or have length {n}")
    return [float(v) for v in arr]


def _broadcast_str(value, n, name):
    if isinstance(value, str):
        return [value] * n
    seq = list(value)
    if len(seq) != n:
        raise ValueError(f"{name} must be a single name or have length {n}")
    return seq


def _criteria_types(types, n):
    if types is None:
        return np.ones(n, dtype=float)
    out = np.empty(n, dtype=float)
    seq = list(types)
    if len(seq) != n:
        raise ValueError(f"criteria_types must have length {n}")
    for i, t in enumerate(seq):
        if isinstance(t, str):
            key = t.strip().lower()
            if key in ("max", "benefit", "+"):
                out[i] = 1.0
            elif key in ("min", "cost", "-"):
                out[i] = -1.0
            else:
                raise ValueError(f"unknown criterion type {t!r}")
        else:
            out[i] = 1.0 if float(t) >= 0 else -1.0
    return out

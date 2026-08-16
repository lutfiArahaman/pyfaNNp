"""Triangular fuzzy number helpers.

A triangular fuzzy number (TFN) is stored as ``(l, m, u)`` with ``l <= m <= u``.
A fuzzy pairwise comparison matrix is therefore an array of shape
``(n, n, 3)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SAATY_RI",
    "as_tfn_matrix",
    "defuzzify",
    "from_saaty",
    "is_reciprocal",
]

#: Saaty's random consistency index, indexed by matrix order.
SAATY_RI = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}


def as_tfn_matrix(judgments) -> np.ndarray:
    """Validate and return a fuzzy pairwise comparison matrix.

    Parameters
    ----------
    judgments : array-like, shape (n, n, 3)
        Triangular fuzzy pairwise comparisons.

    Returns
    -------
    ndarray of shape (n, n, 3), dtype float
    """
    J = np.asarray(judgments, dtype=float)
    if J.ndim != 3 or J.shape[0] != J.shape[1] or J.shape[2] != 3:
        raise ValueError(
            f"judgments must have shape (n, n, 3); got {J.shape}. "
            "Use from_saaty() to build one from a crisp Saaty matrix."
        )
    if np.any(J[..., 0] > J[..., 1]) or np.any(J[..., 1] > J[..., 2]):
        raise ValueError("each triangular fuzzy number must satisfy l <= m <= u")
    if np.any(J <= 0):
        raise ValueError("pairwise comparison values must be strictly positive")
    return J


def defuzzify(tfn, method: str = "centroid") -> np.ndarray:
    """Collapse triangular fuzzy numbers to crisp values.

    Parameters
    ----------
    tfn : array-like, last axis of length 3
    method : {"centroid", "graded_mean"}
        ``centroid`` uses ``(l + m + u) / 3``; ``graded_mean`` uses
        ``(l + 4m + u) / 6``, which weights the modal value more heavily.
    """
    T = np.asarray(tfn, dtype=float)
    if T.shape[-1] != 3:
        raise ValueError("last axis must have length 3")
    if method == "centroid":
        return T.sum(axis=-1) / 3.0
    if method == "graded_mean":
        return (T[..., 0] + 4.0 * T[..., 1] + T[..., 2]) / 6.0
    raise ValueError(f"unknown defuzzification method {method!r}")


def from_saaty(matrix, spread: float = 1.0, ceiling: float = 9.0) -> np.ndarray:
    """Build a fuzzy comparison matrix from a crisp Saaty matrix.

    Each above-diagonal entry ``v`` becomes ``(v - spread, v, v + spread)``,
    clipped to ``[1 / ceiling, ceiling]``; diagonal entries become
    ``(1, 1, 1)``; below-diagonal entries are the fuzzy reciprocals
    ``(1/u, 1/m, 1/l)`` of their transpose, so the result is reciprocal by
    construction regardless of the input's exact reciprocity.

    This is a convenience for turning an existing crisp AHP matrix into a
    fuzzy one. When judgements are elicited directly on a linguistic scale,
    build the ``(n, n, 3)`` array from that scale instead and pass it to
    :class:`~pyfap.fahp.FAHP` unchanged.
    """
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix must be square; got {A.shape}")
    if np.any(A <= 0):
        raise ValueError("Saaty values must be strictly positive")

    n = A.shape[0]
    J = np.empty((n, n, 3), dtype=float)
    lo = 1.0 / ceiling

    for i in range(n):
        J[i, i] = (1.0, 1.0, 1.0)
        for j in range(i + 1, n):
            v = A[i, j]
            tfn = np.clip((v - spread, v, v + spread), lo, ceiling)
            tfn = np.sort(tfn)  # guard against clipping inverting the order
            J[i, j] = tfn
            J[j, i] = (1.0 / tfn[2], 1.0 / tfn[1], 1.0 / tfn[0])
    return J


def is_reciprocal(judgments, tol: float = 1e-8) -> bool:
    """Return True if ``J[j, i] == (1/u, 1/m, 1/l)`` of ``J[i, j]``."""
    J = as_tfn_matrix(judgments)
    return bool(np.allclose(J, 1.0 / J.transpose(1, 0, 2)[..., ::-1], atol=tol))

"""Decision matrix preprocessing.

PROMETHEE thresholds are expressed in each criterion's own units, so a
scalar ``q`` or ``p`` applied across criteria only makes sense once those
criteria share a scale. Either supply per-criterion thresholds to
:class:`~pyfap.promethee.Promethee`, or normalise the decision matrix first
with :func:`minmax_normalize`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["minmax_normalize"]


def minmax_normalize(decision_matrix) -> np.ndarray:
    """Scale each criterion to ``[0, 1]``.

    Constant criteria are mapped to zero, since they cannot discriminate
    between alternatives. Criterion direction is *not* applied here; pass
    ``criteria_types`` to :class:`~pyfap.promethee.Promethee` instead, so
    that the direction of each criterion stays visible in the configuration
    rather than being baked into the data.
    """
    X = np.asarray(decision_matrix, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"decision_matrix must be 2-D; got shape {X.shape}")

    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    constant = span == 0

    out = np.zeros_like(X)
    out[:, ~constant] = (X[:, ~constant] - lo[~constant]) / span[~constant]
    return out

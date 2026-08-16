"""Illustrative decision problems.

.. warning::

   ``load_demo()`` returns **synthetic data invented for this scaffold**. It
   exists so that the example script runs out of the box and so the test
   suite has a fixture. It is not a published benchmark and must not be
   described as one.

   The example section of the paper requires a *published* decision problem
   whose original weights and ranking are available, so that the comparison
   is reproducible and independent of any single study. Add such a dataset as
   ``load_<name>()`` alongside this one, with a full citation in its
   docstring, and cite it in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._fuzzy import from_saaty

__all__ = ["DecisionProblem", "load_demo"]


@dataclass
class DecisionProblem:
    """A decision problem: fuzzy judgements plus a decision matrix."""

    judgments: np.ndarray
    decision_matrix: np.ndarray
    alternatives: list = field(default_factory=list)
    criteria: list = field(default_factory=list)
    criteria_types: list = field(default_factory=list)
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"<DecisionProblem {len(self.alternatives)} alternatives x "
            f"{len(self.criteria)} criteria>"
        )


def load_demo() -> DecisionProblem:
    """Return a small synthetic supplier-selection problem.

    Six suppliers assessed on four criteria: unit cost (minimised), a quality
    score, on-time delivery rate, and a service rating. The pairwise
    judgements are deliberately close to consistent so that the consistency
    check passes and the example runs without warnings.

    Synthetic. See the module warning.
    """
    # Crisp Saaty judgements over the four criteria, fuzzified with a spread
    # of one scale point either side of the modal value.
    saaty = np.array(
        [
            [1.0,     2.0,     3.0,     5.0],      # cost
            [1 / 2.0, 1.0,     2.0,     4.0],      # quality
            [1 / 3.0, 1 / 2.0, 1.0,     3.0],      # delivery
            [1 / 5.0, 1 / 4.0, 1 / 3.0, 1.0],      # service
        ]
    )
    judgments = from_saaty(saaty, spread=1.0)

    decision_matrix = np.array(
        [
            [ 48.0, 7.2, 0.91, 6.5],
            [ 61.0, 8.9, 0.96, 8.1],
            [ 43.0, 6.1, 0.84, 5.9],
            [ 55.0, 8.2, 0.93, 7.4],
            [ 67.0, 9.4, 0.98, 8.8],
            [ 51.0, 7.7, 0.88, 7.0],
        ]
    )

    return DecisionProblem(
        judgments=judgments,
        decision_matrix=decision_matrix,
        alternatives=["S1", "S2", "S3", "S4", "S5", "S6"],
        criteria=["cost", "quality", "delivery", "service"],
        criteria_types=["min", "max", "max", "max"],
        description="Synthetic supplier selection; not a published benchmark.",
    )

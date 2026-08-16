"""pyFAP: coupled fuzzy AHP, neural network and PROMETHEE decision analysis.

The package expresses a three-stage decision analysis as a single scripted
pipeline: fuzzy pairwise judgements become criterion weights, the weights
drive a PROMETHEE outranking, and a neural surrogate is trained against the
resulting flows so that large alternative sets can be scored without
re-running the exact outranking.

Basic use::

    from pyfap import FAHP, ANNSurrogate, Promethee, DecisionPipeline
    from pyfap.datasets import load_demo

    problem = load_demo()

    pipe = DecisionPipeline(
        weights=FAHP(method="extent_analysis", consistency_check=True),
        surrogate=ANNSurrogate(hidden=(32, 16), epochs=300, random_state=0),
        ranker=Promethee(version="II", preference="v-shape", q=0.1, p=0.5,
                         criteria_types=problem.criteria_types),
    )

    result = pipe.fit(
        judgments=problem.judgments,
        decision_matrix=problem.decision_matrix,
    ).rank()

    print(result.ranking)
"""

from ._fuzzy import defuzzify, from_saaty
from .ann import ANNSurrogate
from .fahp import FAHP
from .pipeline import DecisionPipeline, DecisionResult
from .preprocessing import minmax_normalize
from .promethee import Promethee, PrometheeFlows
from .sensitivity import StabilityReport, rank_stability

__version__ = "0.1.0"

__all__ = [
    "FAHP",
    "ANNSurrogate",
    "DecisionPipeline",
    "DecisionResult",
    "Promethee",
    "PrometheeFlows",
    "StabilityReport",
    "__version__",
    "defuzzify",
    "from_saaty",
    "minmax_normalize",
    "rank_stability",
]

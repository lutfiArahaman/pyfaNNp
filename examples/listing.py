"""Runnable version of the code listing in paper.md.

    python examples/listing.py

The listing in the paper is reproduced verbatim below. The only additions
here are loading the data, normalising the decision matrix, and printing the
results.

Why the normalisation: PROMETHEE thresholds are expressed in each criterion's
own units, so the scalar ``q=0.1, p=0.5`` in the paper listing presupposes
that the criteria share a scale. The demo problem has cost in the tens and
delivery rate in [0, 1], so it is normalised first. The alternative — and the
better choice for a real study — is to pass per-criterion thresholds, shown
at the bottom of this file.
"""

import numpy as np

from pyfap import ANNSurrogate, DecisionPipeline, FAHP, Promethee
from pyfap.datasets import load_demo
from pyfap.preprocessing import minmax_normalize

problem = load_demo()
J = problem.judgments
X = minmax_normalize(problem.decision_matrix)

# --------------------------------------------------------------------------
# The paper listing.
# --------------------------------------------------------------------------

pipe = DecisionPipeline(
    weights=FAHP(method="extent_analysis", consistency_check=True),
    surrogate=ANNSurrogate(hidden=(32, 16), epochs=300, random_state=0),
    ranker=Promethee(
        version="II",
        preference="v-shape",
        q=0.1,
        p=0.5,
        criteria_types=problem.criteria_types,
    ),
)

result = pipe.fit(
    judgments=J,
    decision_matrix=X,
    # The paper listing omits these two label arguments; everything else in
    # this block is verbatim. Without them the alternatives are reported as
    # A1..A6 rather than by name.
    alternatives=problem.alternatives,
    criteria=problem.criteria,
).rank()

result.weights            # crisp criterion weights
result.consistency_ratio  # CR of the defuzzified judgement matrix
result.net_flow           # PROMETHEE II net flow per alternative
result.ranking            # complete ordering
result.stability(n=1000)  # rank distribution under perturbed weights

# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------

print(problem.description)
print(f"\n{result!r}\n")

print("criterion weights")
for name, w in zip(problem.criteria, result.weights):
    print(f"  {name:<9} {w:.4f}")
print(f"  consistency ratio: {result.consistency_ratio:.4f}")

print("\nPROMETHEE II flows")
print(f"  {'alt':<5}{'phi+':>9}{'phi-':>9}{'phi':>9}")
for i, alt in enumerate(result.alternatives):
    print(
        f"  {alt:<5}{result.positive_flow[i]:>9.4f}"
        f"{result.negative_flow[i]:>9.4f}{result.net_flow[i]:>9.4f}"
    )
print(f"\n  net flows sum to {result.net_flow.sum():.2e} (should be ~0)")
print(f"  ranking: {' > '.join(result.ranking)}")

print(f"\nsurrogate training R^2: {result.surrogate_score:.4f}")
print("surrogate vs exact net flow")
predicted = result.predict(X)
for i, alt in enumerate(result.alternatives):
    print(
        f"  {alt:<5} exact {result.net_flow[i]:>8.4f}"
        f"   predicted {predicted[i]:>8.4f}"
    )

print("\nrank stability under +/-10% weight perturbation")
print(result.stability(n=1000, random_state=0).summary())

# --------------------------------------------------------------------------
# Per-criterion thresholds on the raw, unnormalised matrix. Preferable in a
# real study: each threshold is stated in the units the decision maker
# actually reasons about.
# --------------------------------------------------------------------------

raw = problem.decision_matrix
ranker = Promethee(
    version="II",
    preference="v-shape",
    q=[2.0, 0.3, 0.01, 0.3],     # indifference, per criterion
    p=[10.0, 1.5, 0.06, 1.5],    # preference, per criterion
    criteria_types=problem.criteria_types,
)
raw_result = DecisionPipeline(
    weights=FAHP(method="geometric_mean"),
    ranker=ranker,
).fit_rank(
    judgments=J,
    decision_matrix=raw,
    alternatives=problem.alternatives,
    criteria=problem.criteria,
)

print("\nraw scales, per-criterion thresholds, geometric-mean FAHP")
print(f"  ranking: {' > '.join(raw_result.ranking)}")
print(f"  weights: {np.round(raw_result.weights, 4)}")

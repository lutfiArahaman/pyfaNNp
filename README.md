# pyFAP

Coupled fuzzy AHP, neural network and PROMETHEE decision analysis, as a single
scripted pipeline.

> **Status: scaffold.** This code was written to make the listing in
> `paper.md` executable and to give the test suite something to assert
> against. It has not yet been validated against published worked examples —
> see [Before publishing](#before-publishing).

## Why

A hybrid multi-criteria analysis is usually run across three tools: a
spreadsheet for the fuzzy weights, a script for the predictive model, and
desktop software for the outranking, with results transcribed by hand between
them. The analysis then cannot be re-executed from one command, and the
sensitivity of the ranking to the expert judgements behind the weights is
rarely examined. pyFAP puts the three stages behind one object so that the
whole analysis is a script.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```python
from pyfap import FAHP, ANNSurrogate, Promethee, DecisionPipeline
from pyfap.datasets import load_demo
from pyfap.preprocessing import minmax_normalize

problem = load_demo()

pipe = DecisionPipeline(
    weights=FAHP(method="extent_analysis", consistency_check=True),
    surrogate=ANNSurrogate(hidden=(32, 16), epochs=300, random_state=0),
    ranker=Promethee(version="II", preference="v-shape", q=0.1, p=0.5,
                     criteria_types=problem.criteria_types),
)

result = pipe.fit(
    judgments=problem.judgments,
    decision_matrix=minmax_normalize(problem.decision_matrix),
    alternatives=problem.alternatives,
    criteria=problem.criteria,
).rank()

print(result.ranking)
print(result.stability(n=1000, random_state=0).summary())
```

Run the full example:

```bash
python examples/listing.py
```

## API

| Object | Purpose |
| --- | --- |
| `FAHP` | Crisp weights from a fuzzy pairwise matrix, by Chang's extent analysis or Buckley's geometric mean, with a consistency ratio |
| `Promethee` | Six preference functions, φ⁺/φ⁻/φ flows, PROMETHEE II ranking and PROMETHEE I partial relation |
| `ANNSurrogate` | Feed-forward network trained on the exact flows, for scoring new alternatives |
| `DecisionPipeline` | Composes the three; `fit()` then `rank()` |
| `DecisionResult` | Weights, flows, ranking, partial order, `stability()`, `predict()` |
| `rank_stability` | Monte Carlo rank distribution under perturbed weights |
| `from_saaty` | Fuzzify an existing crisp Saaty matrix |
| `minmax_normalize` | Scale criteria to `[0, 1]` |

Every stage works standalone; `DecisionPipeline` is a convenience, not a
requirement.

## Notes on the methods

- **Extent analysis can return a zero weight** for a criterion that carries
  real information. This is a documented property of the method, not a bug;
  `FAHP` warns when it happens and `method="geometric_mean"` avoids it.
- **`preference="v-shape"` accepts both `q` and `p`.** It reduces to Brans'
  type 3 exactly when `q == 0` and is otherwise his type 5 (V-shape with
  indifference). `"linear"` is an alias.
- **Thresholds are in each criterion's own units.** A scalar `q` or `p`
  across criteria presupposes a shared scale — either normalise first, or
  pass per-criterion thresholds.
- **The surrogate approximates the outranking, it does not replace it.** The
  exact flows stay on the result object and the training R² is reported.

## Before publishing

Required before this supports the claims in `paper.md`:

- [ ] **Validate against published worked examples.** Reproduce the weights
      from Chang (1996) and Buckley (1985) and the flows from Brans & Vincke
      (1985) to a stated tolerance. This is the strongest correctness
      evidence a software paper can offer, and none of it exists yet.
- [ ] **Replace `load_demo()`.** It is synthetic data invented for this
      scaffold, not a benchmark. The paper's example needs a *published*
      decision problem with a known ranking.
- [ ] **Establish the surrogate's value empirically.** Six alternatives
      cannot demonstrate it. Show Spearman ρ against exact flows on a
      held-out split as a function of training size, with wall-clock cost.
- [ ] **Check the Related Work claims.** Install `pymcdm`,
      `scikit-criteria` and `pyDecision` and confirm what each already does.
- [ ] Choose a license, add CI, archive a release for a DOI.

## References

- Buckley, J.J. (1985) Fuzzy hierarchical analysis. *Fuzzy Sets and Systems*
  17(3), 233–247.
- Brans, J.P. & Vincke, P. (1985) A preference ranking organisation method.
  *Management Science* 31(6), 647–656.
- Chang, D.-Y. (1996) Applications of the extent analysis method on fuzzy
  AHP. *European Journal of Operational Research* 95(3), 649–655.

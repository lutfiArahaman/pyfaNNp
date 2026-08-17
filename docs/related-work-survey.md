# Related work survey

Evidence behind the Related Work section of `paper.md`. Produced by
`tools/survey_related_work.py`, run from the `Related work survey` workflow
(`workflow_dispatch`). Repeat it before submission if time has passed — these
packages move.

**Surveyed:** 16 August 2026, Python 3.12.13.

## Versions

| Package | Version | Home |
| --- | --- | --- |
| pymcdm | 1.4.0 | github.com/kotbaton/pymcdm |
| scikit-criteria | 0.10 | scikit-criteria.quatrope.org |
| pyDecision | 5.1.1 | github.com/Valdecy/pyDecisions |
| scikit-fuzzy | 0.5.0 | github.com/scikit-fuzzy/scikit-fuzzy |
| simpful | 2.12.0 | github.com/aresio/simpful |

## The claim under test

*Does any package already chain a fuzzy AHP weight derivation to a PROMETHEE
outranking?*

| Package | Fuzzy AHP | PROMETHEE | Verdict |
| --- | --- | --- | --- |
| pymcdm | no | yes | PROMETHEE only |
| scikit-criteria | no | no | neither |
| **pyDecision** | **yes** | **yes** | **both** |
| scikit-fuzzy | no | no | neither |
| simpful | no | no | neither |

**pyDecision has both.** The paper must concede this explicitly rather than
leave it to be discovered.

## What each actually provides

### pyDecision 5.1.1 — the closest existing work

```
pyDecision.algorithm.fuzzy_ahp_method(dataset)
pyDecision.algorithm.ahp_method(dataset, wd='m')
pyDecision.algorithm.promethee_i(dataset, W, Q, S, P, F, graph=False)
pyDecision.algorithm.promethee_ii(dataset, W, Q, S, P, F, sort=True, topn=0,
                                  graph=False, verbose=True)
```

Also `promethee_iii` … `promethee_vi`, `promethee_gaia`, `ec_promethee`,
`fuzzy_ahp_ppf`, `ppf_ahp_method`.

- Both stages are present, so the "these methods don't exist in Python" claim
  is **false** and must not be made.
- They are plain functions over arrays, returning tuples. None of them carry
  state; nothing connects the output of `fuzzy_ahp_method` to the `W`
  argument of `promethee_ii` except the analyst.
- No docstrings on the inspected functions.
- No machine-learning component, no sensitivity analysis.

### pymcdm 1.4.0

```
pymcdm.methods.PROMETHEE_II(preference_function, p=None, q=None)
    preference functions: 'usual', 'ushape', 'vshape', 'level', 'vshape_2'
pymcdm.weights.subjective.AHP(ranking=None, scoring=None, object_names=None,
                              matrix=None, filename=None)
pymcdm.helpers.param_sensitivity(Method, matrix, weights, types,
                                 param_name, param_values, **init_kwargs)
```

- AHP is **crisp**, not fuzzy, though it does compute consistency using
  tabulated RI values.
- Five preference functions, not the six of Brans and Vincke.
- `param_sensitivity` sweeps **one named parameter** over a supplied list of
  values. It is not a sample over the weight vector, so it does not answer
  "how much of this ranking survives the imprecision in the judgements?"
- Well documented, with literature references in the docstrings.

### scikit-criteria 0.10

Aggregation methods: `aras, cocoso, codas, copras, edas, electre, ervd,
mabac, mairca, marcos, moora, ocra, probid, ram, rim, similarity, simple,
simus, spotis, topsis, vikor, waspas`.

- **Neither AHP nor PROMETHEE.** The draft's grouping of it with the other
  two was wrong.
- Does have `skcriteria.pipeline` / `skcriteria.pipelines` — a pipeline
  abstraction — and `skcriteria.ranksrev`, a rank-reversal module. So neither
  "composition" nor "rank-reversal analysis" is novel in itself; what is
  particular to pyFAP is the set of methods composed.

### scikit-fuzzy 0.5.0 and simpful 2.12.0

Fuzzy machinery only: `defuzz`, `defuzzify`, `Triangular_MF`,
`TrapezoidFuzzySet`, `Trapezoidal_MF`. No pairwise-comparison weight
derivation, no consistency diagnostics, no outranking. The draft's
characterisation of these two was accurate.

## Cross-check: do the numbers agree?

Run after the survey, by `tools/crosscheck.py` and then as assertions in
`tests/test_crosscheck.py`. **54 assertions, all passing.**

| Quantity | Compared against | Result |
| --- | --- | --- |
| Fuzzy AHP weights (geometric mean) | `pyDecision.fuzzy_ahp_method` | agree to 1e-10, orders 3–6 |
| PROMETHEE II net flows | `pyDecision.promethee_ii` | agree to 1e-10, all six preference functions |
| PROMETHEE II net flows | `pymcdm.methods.PROMETHEE_II` | agree to 1e-10, five preference functions |
| Mixed per-criterion functions | `pyDecision.promethee_ii` | agree to 1e-10 |

Reading `pyDecision`'s `preference_degree` source confirmed the mapping:

| Brans type | pyFAP | pyDecision | pymcdm |
| --- | --- | --- | --- |
| 1 usual | `"usual"` | `t1` | `usual` |
| 2 U-shape | `"u-shape"` (q) | `t2` | `ushape` |
| 3 V-shape | `"v-shape"`, q=0 (p) | `t3` | `vshape` |
| 4 level | `"level"` (q, p) | `t4` | `level` |
| 5 linear | `"v-shape"`, q>0 (q, p) | `t5` | `vshape_2` |
| 6 Gaussian | `"gaussian"` (s) | `t6` | — |

`pyDecision` has a seventh function with no counterpart here.

### The one difference found

Consistency ratios do not match, and the reason is two deliberate choices,
not an error. `pyDecision` defuzzifies with the graded mean
`(l + 4m + u)/6` and takes lambda-max as the mean of the ratios
`(Aw)_i / w_i`, the standard AHP approximation. pyFAP defuzzifies with the
centroid `(l + m + u)/3` and computes the true principal eigenvalue. Both
are defensible. The weights, which are what the ranking depends on, agree
exactly. `test_crosscheck.py` pins the discrepancy so it stays a known
difference.

## Consequences for the paper

1. **Concede pyDecision up front.** It provides both stages. The contribution
   is the composition and the surrogate, not the availability of the methods.
2. **Drop any claim that sensitivity analysis is novel.** pymcdm ships a
   parameter sweep and scikit-criteria ships a rank-reversal module. The
   distinction is Monte Carlo over the *weight vector* rather than a sweep
   over one parameter, and it is a distinction of kind, not of existence.
3. **Drop any claim that a pipeline abstraction is novel.** scikit-criteria
   has one.
4. **Fix the scikit-criteria sentence.** It has neither AHP nor PROMETHEE.
5. **The surrogate stands.** No surveyed package contains any
   machine-learning component. This is the one unambiguously distinct
   element, which is a further reason the surrogate subsection has to carry
   its weight.

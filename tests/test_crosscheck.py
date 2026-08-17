"""External validation against independently written implementations.

These tests compare pyFAP against `pyDecision` and `pymcdm` on the same
inputs. Agreement with code written by other people, from the same published
specifications, is the strongest correctness evidence available while
`test_published.py` remains unfilled -- it is not the same as agreeing with
the source papers, but two independent implementations converging on the
same numbers is hard to achieve by coincidence.

The comparison packages are not dependencies of pyFAP, so the whole module
skips unless they are installed:

    pip install pyDecision pymcdm
    pytest tests/test_crosscheck.py -v

CI runs it from the `Related work survey` workflow, which installs them.

Verified mapping of preference functions, read from pyDecision's
`preference_degree` source and pymcdm's documented names:

    Brans type   pyFAP                     pyDecision   pymcdm
    1  usual     "usual"                   t1           usual
    2  U-shape   "u-shape"       (q)       t2           ushape
    3  V-shape   "v-shape", q=0  (p)       t3           vshape
    4  level     "level"         (q, p)    t4           level
    5  linear    "v-shape", q>0  (q, p)    t5           vshape_2
    6  Gaussian  "gaussian"      (s)       t6           --

pyDecision additionally provides a `t7` with no counterpart here.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfap import FAHP, Promethee, from_saaty

pytest.importorskip("pyDecision", reason="comparison package not installed")
pytest.importorskip("pymcdm", reason="comparison package not installed")

TOLERANCE = 1e-10


def random_saaty(rng, n):
    """A reciprocal crisp comparison matrix on the Saaty scale."""
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = float(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]))
            matrix[i, j] = value
            matrix[j, i] = 1.0 / value
    return matrix


def to_pydecision_tfn(judgments):
    """pyDecision takes a list of rows of (l, m, u) tuples."""
    n = judgments.shape[0]
    return [
        [tuple(float(x) for x in judgments[i, j]) for j in range(n)]
        for i in range(n)
    ]


class TestFuzzyAhpAgainstPyDecision:
    """pyDecision's `fuzzy_ahp_method` is Buckley's geometric mean: row-wise
    fuzzy products, nth root, normalisation with crossed bounds, centroid
    defuzzification, then renormalisation. That is the same construction as
    ``FAHP(method="geometric_mean")``, so the weights must agree.
    """

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    @pytest.mark.parametrize("seed", range(4))
    def test_weights_agree(self, n, seed):
        from pyDecision.algorithm import fuzzy_ahp_method

        rng = np.random.default_rng(1000 + seed)
        judgments = from_saaty(random_saaty(rng, n), spread=1.0)

        ours = FAHP(
            method="geometric_mean", consistency_check=False
        ).derive(judgments)
        _, _, theirs, _ = fuzzy_ahp_method(to_pydecision_tfn(judgments))

        assert ours == pytest.approx(np.asarray(theirs), abs=TOLERANCE)

    def test_consistency_ratio_differs_for_documented_reasons(self):
        """The weights agree exactly; the consistency ratios do not, and the
        reason is two deliberate choices rather than an error.

        pyDecision defuzzifies with the graded mean (l + 4m + u)/6 and takes
        lambda_max as the mean of the ratios (Aw)_i / w_i, the standard AHP
        approximation. pyFAP defuzzifies with the centroid (l + m + u)/3 and
        takes lambda_max as the true principal eigenvalue. Both are
        defensible; the test pins the discrepancy so that it stays a known
        difference rather than becoming a surprise.
        """
        from pyDecision.algorithm import fuzzy_ahp_method

        rng = np.random.default_rng(7)
        judgments = from_saaty(random_saaty(rng, 4), spread=1.0)

        model = FAHP(method="geometric_mean", consistency_check=True)
        model.derive(judgments)
        *_, their_cr = fuzzy_ahp_method(to_pydecision_tfn(judgments))

        assert model.consistency_ratio_ is not None
        assert their_cr >= 0.0
        # Same order of magnitude, not the same number.
        assert abs(model.consistency_ratio_ - their_cr) < 0.5


# (pyFAP preference, q, p, s, pyDecision code, pymcdm name)
PREFERENCE_MAP = [
    ("usual", 0.0, 1.0, 1.0, "t1", "usual"),
    ("u-shape", 0.10, 1.0, 1.0, "t2", "ushape"),
    ("v-shape", 0.0, 0.50, 1.0, "t3", "vshape"),
    ("level", 0.10, 0.50, 1.0, "t4", "level"),
    ("v-shape", 0.10, 0.50, 1.0, "t5", "vshape_2"),
    ("gaussian", 0.0, 1.0, 0.30, "t6", None),
]


def flows_from_pyfap(matrix, weights, preference, q, p, s):
    return Promethee(
        version="II", preference=preference, q=q, p=p, s=s
    ).rank(matrix, weights).net_flow


class TestPrometheeAgainstPyDecision:
    @pytest.mark.parametrize(
        "preference,q,p,s,code,_pymcdm", PREFERENCE_MAP
    )
    @pytest.mark.parametrize("seed", range(3))
    def test_net_flows_agree(self, preference, q, p, s, code, _pymcdm, seed):
        from pyDecision.algorithm import promethee_ii

        rng = np.random.default_rng(2000 + seed)
        n_alt, n_crit = 6, 4
        matrix = rng.random((n_alt, n_crit))
        weights = rng.random(n_crit) + 0.1

        ours = flows_from_pyfap(matrix, weights, preference, q, p, s)
        theirs = promethee_ii(
            matrix,
            W=weights.tolist(),
            Q=[q] * n_crit,
            S=[s] * n_crit,
            P=[p] * n_crit,
            F=[code] * n_crit,
            sort=False,
            graph=False,
            verbose=False,
        )
        # Column 0 is a 1-based index; column 1 is the net flow.
        assert ours == pytest.approx(np.asarray(theirs)[:, 1], abs=TOLERANCE)

    @pytest.mark.parametrize("seed", range(3))
    def test_mixed_preference_functions_agree(self, seed):
        """Each criterion gets a different preference function -- the case a
        per-criterion broadcasting error would show up in."""
        from pyDecision.algorithm import promethee_ii

        rng = np.random.default_rng(3000 + seed)
        n_alt = 7
        chosen = [PREFERENCE_MAP[i] for i in (0, 2, 3, 4)]
        n_crit = len(chosen)

        matrix = rng.random((n_alt, n_crit))
        weights = rng.random(n_crit) + 0.1

        ours = Promethee(
            version="II",
            preference=[c[0] for c in chosen],
            q=[c[1] for c in chosen],
            p=[c[2] for c in chosen],
            s=[c[3] for c in chosen],
        ).rank(matrix, weights).net_flow

        theirs = promethee_ii(
            matrix,
            W=weights.tolist(),
            Q=[c[1] for c in chosen],
            S=[c[3] for c in chosen],
            P=[c[2] for c in chosen],
            F=[c[4] for c in chosen],
            sort=False,
            graph=False,
            verbose=False,
        )
        assert ours == pytest.approx(np.asarray(theirs)[:, 1], abs=TOLERANCE)


class TestPrometheeAgainstPymcdm:
    @pytest.mark.parametrize(
        "preference,q,p,s,_code,pymcdm_name",
        [row for row in PREFERENCE_MAP if row[5] is not None],
    )
    @pytest.mark.parametrize("seed", range(3))
    def test_net_flows_agree(
        self, preference, q, p, s, _code, pymcdm_name, seed
    ):
        from pymcdm.methods import PROMETHEE_II

        rng = np.random.default_rng(4000 + seed)
        n_alt, n_crit = 6, 4
        matrix = rng.random((n_alt, n_crit))
        weights = rng.random(n_crit) + 0.1
        weights = weights / weights.sum()

        ours = flows_from_pyfap(matrix, weights, preference, q, p, s)
        method = PROMETHEE_II(pymcdm_name, p=[p] * n_crit, q=[q] * n_crit)
        theirs = method(matrix, weights, np.ones(n_crit))

        assert ours == pytest.approx(np.asarray(theirs), abs=TOLERANCE)


def test_all_three_implementations_agree():
    """The headline: one problem, three independent implementations."""
    from pyDecision.algorithm import promethee_ii
    from pymcdm.methods import PROMETHEE_II

    matrix = np.array(
        [
            [0.80, 0.30, 0.55],
            [0.55, 0.75, 0.40],
            [0.35, 0.60, 0.90],
            [0.65, 0.45, 0.25],
            [0.20, 0.85, 0.70],
        ]
    )
    weights = np.array([0.55, 0.30, 0.15])
    q, p = 0.10, 0.50

    ours = flows_from_pyfap(matrix, weights, "v-shape", q, p, 1.0)

    pydecision = np.asarray(
        promethee_ii(
            matrix, W=weights.tolist(), Q=[q] * 3, S=[0.0] * 3, P=[p] * 3,
            F=["t5"] * 3, sort=False, graph=False, verbose=False,
        )
    )[:, 1]

    pymcdm = np.asarray(
        PROMETHEE_II("vshape_2", p=[p] * 3, q=[q] * 3)(
            matrix, weights, np.ones(3)
        )
    )

    assert ours == pytest.approx(pydecision, abs=TOLERANCE)
    assert ours == pytest.approx(pymcdm, abs=TOLERANCE)
    assert ours.sum() == pytest.approx(0.0, abs=1e-12)

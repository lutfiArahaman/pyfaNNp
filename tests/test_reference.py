"""Validation against independent reference implementations and closed forms.

Two kinds of check live here.

*Differential tests* compare the vectorised implementations in ``pyfap``
against deliberately naive versions written directly below, transcribing the
published formulas with explicit loops and no array tricks. The two are
written differently on purpose: the vectorised code crosses indices when it
inverts a triangular fuzzy number and builds deviation matrices by
broadcasting, and those are precisely the steps where a silent transposition
or a reversed sign would survive every invariant test in the rest of the
suite. Agreement across random inputs is meaningful evidence; agreement is
not proof that the shared understanding of the formula is right, which is
what ``test_published.py`` is for.

*Closed-form tests* pin values that can be derived analytically, so they
depend on no implementation at all.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyfap import FAHP, Promethee, from_saaty

PREFERENCE_KINDS = (
    "usual",
    "u-shape",
    "v-shape",
    "level",
    "linear",
    "gaussian",
)

# ---------------------------------------------------------------------------
# Naive reference implementations. Slow, loop-based, formula-literal.
# ---------------------------------------------------------------------------


def scalar_preference(d, kind, q, p, s):
    """Brans' preference functions, one scalar deviation at a time."""
    if kind == "usual":
        return 1.0 if d > 0 else 0.0
    if kind == "u-shape":
        return 1.0 if d > q else 0.0
    if kind in ("v-shape", "linear"):
        if d <= q:
            return 0.0
        if d > p:
            return 1.0
        return (d - q) / (p - q)
    if kind == "level":
        if d <= q:
            return 0.0
        if d > p:
            return 1.0
        return 0.5
    if kind == "gaussian":
        if d <= 0:
            return 0.0
        return 1.0 - math.exp(-(d * d) / (2.0 * s * s))
    raise ValueError(kind)


def naive_promethee(X, weights, kinds, qs, ps, ss, types):
    """PROMETHEE flows by explicit triple loop."""
    X = np.asarray(X, dtype=float)
    n, m = X.shape
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    pi = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            total = 0.0
            for k in range(m):
                d = (X[a, k] - X[b, k]) * types[k]
                total += w[k] * scalar_preference(d, kinds[k], qs[k], ps[k], ss[k])
            pi[a, b] = total

    phi_plus = np.array(
        [sum(pi[a, b] for b in range(n) if b != a) / (n - 1) for a in range(n)]
    )
    phi_minus = np.array(
        [sum(pi[b, a] for b in range(n) if b != a) / (n - 1) for a in range(n)]
    )
    return phi_plus, phi_minus, phi_plus - phi_minus


def scalar_degree_of_possibility(si, sj):
    """V(S_i >= S_j), written out term by term."""
    _li, mi, ui = si
    lj, mj, _uj = sj
    if mi >= mj:
        return 1.0
    if lj >= ui:
        return 0.0
    return (lj - ui) / ((mi - ui) - (mj - lj))


def naive_extent_analysis(J):
    """Chang's extent analysis by explicit loop."""
    J = np.asarray(J, dtype=float)
    n = J.shape[0]

    rows = []
    for i in range(n):
        rows.append(
            (
                sum(J[i, j, 0] for j in range(n)),
                sum(J[i, j, 1] for j in range(n)),
                sum(J[i, j, 2] for j in range(n)),
            )
        )
    total_l = sum(r[0] for r in rows)
    total_m = sum(r[1] for r in rows)
    total_u = sum(r[2] for r in rows)

    # The inverse of a triangular fuzzy number swaps its bounds, so the
    # synthetic extent divides the lower bound by the total upper bound.
    S = [(r[0] / total_u, r[1] / total_m, r[2] / total_l) for r in rows]

    d = []
    for i in range(n):
        v = 1.0
        for j in range(n):
            if i != j:
                v = min(v, scalar_degree_of_possibility(S[i], S[j]))
        d.append(v)

    total = sum(d)
    return np.array([x / total for x in d])


def naive_geometric_mean(J):
    """Buckley's geometric mean method by explicit loop."""
    J = np.asarray(J, dtype=float)
    n = J.shape[0]

    r = []
    for i in range(n):
        r.append(
            (
                math.prod(J[i, j, 0] for j in range(n)) ** (1.0 / n),
                math.prod(J[i, j, 1] for j in range(n)) ** (1.0 / n),
                math.prod(J[i, j, 2] for j in range(n)) ** (1.0 / n),
            )
        )
    sum_l = sum(x[0] for x in r)
    sum_m = sum(x[1] for x in r)
    sum_u = sum(x[2] for x in r)

    crisp = []
    for lo, mid, hi in r:
        fuzzy_weight = (lo / sum_u, mid / sum_m, hi / sum_l)
        crisp.append(sum(fuzzy_weight) / 3.0)

    total = sum(crisp)
    return np.array([x / total for x in crisp])


# ---------------------------------------------------------------------------
# Random problem generation
# ---------------------------------------------------------------------------


def random_fuzzy_matrix(rng, n):
    """A random reciprocal fuzzy comparison matrix."""
    saaty = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            saaty[i, j] = v
            saaty[j, i] = 1.0 / v
    return from_saaty(saaty, spread=rng.choice([0.5, 1.0, 1.5]))


# ---------------------------------------------------------------------------
# Differential tests: PROMETHEE
# ---------------------------------------------------------------------------


class TestPrometheeAgainstReference:
    @pytest.mark.parametrize("kind", PREFERENCE_KINDS)
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_naive_single_preference(self, kind, seed):
        rng = np.random.default_rng(seed)
        n_alt, n_crit = 7, 4
        X = rng.random((n_alt, n_crit)) * 10.0
        w = rng.random(n_crit)
        q, p, s = 0.4, 2.5, 1.5
        types = np.ones(n_crit)

        flows = Promethee(preference=kind, q=q, p=p, s=s).rank(X, w)
        ref_plus, ref_minus, ref_net = naive_promethee(
            X, w, [kind] * n_crit, [q] * n_crit, [p] * n_crit,
            [s] * n_crit, types,
        )

        assert flows.positive_flow == pytest.approx(ref_plus, abs=1e-12)
        assert flows.negative_flow == pytest.approx(ref_minus, abs=1e-12)
        assert flows.net_flow == pytest.approx(ref_net, abs=1e-12)

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_naive_mixed_preferences(self, seed):
        """Different preference function, threshold and direction per
        criterion -- the configuration most likely to expose a broadcasting
        error."""
        rng = np.random.default_rng(100 + seed)
        n_alt, n_crit = 8, 5

        X = rng.random((n_alt, n_crit)) * 20.0 - 5.0
        w = rng.random(n_crit) + 0.05
        kinds = [
            PREFERENCE_KINDS[rng.integers(len(PREFERENCE_KINDS))]
            for _ in range(n_crit)
        ]
        qs = list(rng.random(n_crit) * 1.5)
        ps = [q + 0.5 + rng.random() * 3.0 for q in qs]
        ss = list(rng.random(n_crit) * 2.0 + 0.5)
        type_labels = [("min" if rng.random() < 0.5 else "max") for _ in range(n_crit)]
        type_signs = np.array([-1.0 if t == "min" else 1.0 for t in type_labels])

        flows = Promethee(
            preference=kinds, q=qs, p=ps, s=ss, criteria_types=type_labels
        ).rank(X, w)
        ref_plus, ref_minus, ref_net = naive_promethee(
            X, w, kinds, qs, ps, ss, type_signs
        )

        assert flows.positive_flow == pytest.approx(ref_plus, abs=1e-12)
        assert flows.negative_flow == pytest.approx(ref_minus, abs=1e-12)
        assert flows.net_flow == pytest.approx(ref_net, abs=1e-12)

    @pytest.mark.parametrize("seed", range(4))
    def test_minimising_equals_negating_the_column(self, seed):
        rng = np.random.default_rng(200 + seed)
        X = rng.random((6, 3)) * 10.0
        w = rng.random(3)

        flipped = X.copy()
        flipped[:, 1] *= -1.0

        a = Promethee(
            preference="v-shape", q=0.2, p=2.0,
            criteria_types=["max", "min", "max"],
        ).rank(X, w)
        b = Promethee(preference="v-shape", q=0.2, p=2.0).rank(flipped, w)

        assert a.net_flow == pytest.approx(b.net_flow, abs=1e-12)


# ---------------------------------------------------------------------------
# Differential tests: FAHP
# ---------------------------------------------------------------------------


class TestFahpAgainstReference:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_extent_analysis_matches_naive(self, seed, n):
        rng = np.random.default_rng(300 + seed)
        J = random_fuzzy_matrix(rng, n)

        # Zero weights are expected here often enough that warning on them
        # would just be noise; the point of this test is agreement.
        got = FAHP(method="extent_analysis", consistency_check=False).derive(J)
        expected = naive_extent_analysis(J)
        assert got == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_geometric_mean_matches_naive(self, seed, n):
        rng = np.random.default_rng(400 + seed)
        J = random_fuzzy_matrix(rng, n)

        got = FAHP(method="geometric_mean", consistency_check=False).derive(J)
        expected = naive_geometric_mean(J)
        assert got == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("seed", range(8))
    def test_degree_of_possibility_matches_naive(self, seed):
        from pyfap.fahp import _degree_of_possibility

        rng = np.random.default_rng(500 + seed)
        for _ in range(50):
            si = np.sort(rng.random(3) * 5.0)
            sj = np.sort(rng.random(3) * 5.0)
            assert _degree_of_possibility(si, sj) == pytest.approx(
                scalar_degree_of_possibility(si, sj), abs=1e-12
            )


# ---------------------------------------------------------------------------
# Closed forms: values derivable by hand, independent of any implementation
# ---------------------------------------------------------------------------


class TestClosedForms:
    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_usual_criterion_net_flow_is_linear_in_rank(self, n):
        """With one criterion, the usual preference function and distinct
        values, the alternative with k others strictly below it has

            phi = (2k - (n - 1)) / (n - 1)

        because phi+ = k/(n-1) and phi- = (n-1-k)/(n-1).
        """
        X = np.arange(n, dtype=float).reshape(-1, 1)
        flows = Promethee(preference="usual").rank(X, [1.0])
        expected = np.array([(2 * k - (n - 1)) / (n - 1) for k in range(n)])
        assert flows.net_flow == pytest.approx(expected)

    def test_best_and_worst_reach_the_extremes(self):
        X = np.arange(5, dtype=float).reshape(-1, 1)
        flows = Promethee(preference="usual").rank(X, [1.0])
        assert flows.net_flow[-1] == pytest.approx(1.0)
        assert flows.net_flow[0] == pytest.approx(-1.0)

    def test_identical_alternatives_have_zero_flow(self):
        X = np.ones((4, 3))
        flows = Promethee(preference="v-shape", q=0.0, p=1.0).rank(
            X, [1.0, 1.0, 1.0]
        )
        assert flows.net_flow == pytest.approx(np.zeros(4))
        assert flows.positive_flow == pytest.approx(np.zeros(4))

    def test_gaussian_matches_its_formula(self):
        s, d = 1.7, 2.3
        expected = 1.0 - math.exp(-(d**2) / (2 * s**2))
        flows = Promethee(preference="gaussian", s=s).rank([[d], [0.0]], [1.0])
        assert flows.net_flow[0] == pytest.approx(expected)

    def test_level_takes_exactly_three_values(self):
        X = np.array([[0.0], [0.05], [0.3], [0.9]])
        flows = Promethee(preference="level", q=0.1, p=0.5).rank(X, [1.0])
        distinct = np.unique(np.round(flows.preference_index, 12))
        assert set(distinct.tolist()) <= {0.0, 0.5, 1.0}

    def test_geometric_mean_recovers_generating_vector(self):
        """For a perfectly consistent matrix A[i, j] = v_i / v_j, Buckley's
        method must return v normalised."""
        v = np.array([0.42, 0.27, 0.19, 0.12])
        A = v[:, None] / v[None, :]
        w = FAHP(method="geometric_mean", consistency_check=False).derive(
            from_saaty(A, spread=0.0)
        )
        assert w == pytest.approx(v / v.sum(), abs=1e-10)

    def test_extent_analysis_is_winner_take_all_without_fuzziness(self):
        """A zero-spread matrix makes every synthetic extent degenerate, so
        V(S_i >= S_j) is 1 or 0 with nothing in between and the whole weight
        collapses onto the top criterion.

        This is a property of Chang's method on crisp input, not a defect in
        this implementation, and it is the mechanism behind the zero weights
        the method is criticised for. It is pinned here so that any change to
        the degree-of-possibility calculation is caught immediately.
        """
        v = np.array([0.5, 0.25, 0.15, 0.10])
        A = v[:, None] / v[None, :]
        with pytest.warns(RuntimeWarning, match="weight of zero"):
            w = FAHP(
                method="extent_analysis", consistency_check=False
            ).derive(from_saaty(A, spread=0.0))
        assert w == pytest.approx([1.0, 0.0, 0.0, 0.0])

    def test_consistent_matrix_has_lambda_max_equal_n(self):
        v = np.array([0.4, 0.3, 0.2, 0.1])
        A = v[:, None] / v[None, :]
        model = FAHP(method="geometric_mean", consistency_check=True)
        model.derive(from_saaty(A, spread=0.0))
        assert model.lambda_max_ == pytest.approx(4.0, abs=1e-9)
        assert model.consistency_index_ == pytest.approx(0.0, abs=1e-9)
        assert model.consistency_ratio_ == pytest.approx(0.0, abs=1e-9)

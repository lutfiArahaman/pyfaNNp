import numpy as np
import pytest

from pyfap import FAHP, from_saaty
from pyfap._fuzzy import as_tfn_matrix, defuzzify, is_reciprocal


def crisp_consistent():
    """A perfectly consistent Saaty matrix: A[i, j] = v[i] / v[j]."""
    v = np.array([0.5, 0.25, 0.15, 0.10])
    return v[:, None] / v[None, :], v


class TestFuzzyHelpers:
    def test_from_saaty_is_reciprocal(self):
        A, _ = crisp_consistent()
        assert is_reciprocal(from_saaty(A))

    def test_from_saaty_diagonal_is_unit(self):
        A, _ = crisp_consistent()
        J = from_saaty(A)
        assert np.allclose(J[np.diag_indices(4)], 1.0)

    def test_defuzzify_centroid(self):
        assert defuzzify(np.array([1.0, 2.0, 6.0])) == pytest.approx(3.0)

    def test_defuzzify_graded_mean(self):
        got = defuzzify(np.array([1.0, 2.0, 6.0]), method="graded_mean")
        assert got == pytest.approx(15.0 / 6.0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            as_tfn_matrix(np.ones((4, 4)))

    def test_rejects_unordered_tfn(self):
        bad = np.ones((2, 2, 3))
        bad[0, 1] = (3.0, 2.0, 1.0)
        with pytest.raises(ValueError, match="l <= m <= u"):
            as_tfn_matrix(bad)


class TestConsistency:
    def test_consistent_matrix_has_near_zero_cr(self):
        A, _ = crisp_consistent()
        model = FAHP(consistency_check=True)
        model.derive(from_saaty(A, spread=0.0))
        assert model.consistency_ratio_ == pytest.approx(0.0, abs=1e-6)

    def test_lambda_max_at_least_n(self):
        A, _ = crisp_consistent()
        model = FAHP(consistency_check=True)
        model.derive(from_saaty(A))
        assert model.lambda_max_ >= 4 - 1e-9

    def test_inconsistent_matrix_warns(self):
        A = np.array(
            [
                [1.0, 9.0, 1 / 9.0],
                [1 / 9.0, 1.0, 9.0],
                [9.0, 1 / 9.0, 1.0],
            ]
        )
        with pytest.warns(RuntimeWarning, match="consistency ratio"):
            FAHP(consistency_check=True).derive(from_saaty(A, spread=0.0))


class TestWeights:
    @pytest.mark.parametrize("method", ["extent_analysis", "geometric_mean"])
    def test_weights_sum_to_one(self, method):
        A, _ = crisp_consistent()
        w = FAHP(method=method).derive(from_saaty(A))
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= 0)

    def test_geometric_mean_recovers_the_priority_vector(self):
        """On a perfectly consistent crisp matrix, Buckley's method should
        return the vector that generated it."""
        A, v = crisp_consistent()
        w = FAHP(method="geometric_mean").derive(from_saaty(A, spread=0.0))
        assert w == pytest.approx(v / v.sum(), abs=1e-6)

    def test_weights_respect_judgement_order(self):
        A, _ = crisp_consistent()
        w = FAHP(method="geometric_mean").derive(from_saaty(A))
        assert np.all(np.diff(w) < 0)  # criteria entered in decreasing priority

    def test_equal_judgements_give_equal_weights(self):
        J = np.ones((4, 4, 3))
        w = FAHP(method="geometric_mean").derive(J)
        assert w == pytest.approx(np.full(4, 0.25))

    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            FAHP(method="nope")

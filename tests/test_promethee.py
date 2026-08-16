import numpy as np
import pytest

from pyfap import Promethee


class TestFlowInvariants:
    """Properties that must hold for any decision matrix."""

    @pytest.mark.parametrize("seed", range(5))
    def test_net_flows_sum_to_zero(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.random((7, 4))
        w = rng.random(4)
        flows = Promethee(preference="v-shape", q=0.05, p=0.4).rank(X, w)
        assert flows.net_flow.sum() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("seed", range(5))
    def test_flows_are_bounded(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.random((6, 3))
        w = rng.random(3)
        flows = Promethee(preference="usual").rank(X, w)
        assert np.all(flows.positive_flow >= -1e-12)
        assert np.all(flows.positive_flow <= 1 + 1e-12)
        assert np.all(np.abs(flows.net_flow) <= 1 + 1e-12)

    def test_preference_index_diagonal_is_zero(self):
        X = np.array([[1.0], [2.0], [3.0]])
        flows = Promethee(preference="usual").rank(X, [1.0])
        assert np.allclose(np.diag(flows.preference_index), 0.0)

    def test_weights_are_normalised_internally(self):
        X = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 1.0]])
        a = Promethee(preference="usual").rank(X, [1.0, 1.0])
        b = Promethee(preference="usual").rank(X, [50.0, 50.0])
        assert a.net_flow == pytest.approx(b.net_flow)


class TestPreferenceFunctions:
    """Hand-computable two-alternative cases."""

    def test_usual(self):
        flows = Promethee(preference="usual").rank([[1.0], [0.0]], [1.0])
        assert flows.net_flow == pytest.approx([1.0, -1.0])

    def test_u_shape_below_threshold_is_indifferent(self):
        flows = Promethee(preference="u-shape", q=0.5).rank(
            [[0.3], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([0.0, 0.0])

    def test_v_shape_interpolates(self):
        # d = 0.3, q = 0.1, p = 0.5  ->  (0.3 - 0.1) / (0.5 - 0.1) = 0.5
        flows = Promethee(preference="v-shape", q=0.1, p=0.5).rank(
            [[0.3], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([0.5, -0.5])

    def test_v_shape_with_zero_q_is_brans_type_three(self):
        # d = 0.25, p = 0.5  ->  0.5
        flows = Promethee(preference="v-shape", q=0.0, p=0.5).rank(
            [[0.25], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([0.5, -0.5])

    def test_linear_is_an_alias_of_v_shape(self):
        X, w = [[0.3], [0.0]], [1.0]
        a = Promethee(preference="v-shape", q=0.1, p=0.5).rank(X, w)
        b = Promethee(preference="linear", q=0.1, p=0.5).rank(X, w)
        assert a.net_flow == pytest.approx(b.net_flow)

    def test_level_gives_one_half_between_thresholds(self):
        flows = Promethee(preference="level", q=0.1, p=0.5).rank(
            [[0.3], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([0.5, -0.5])

    def test_level_saturates_above_p(self):
        flows = Promethee(preference="level", q=0.1, p=0.5).rank(
            [[0.9], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([1.0, -1.0])

    def test_gaussian(self):
        expected = 1.0 - np.exp(-0.5)
        flows = Promethee(preference="gaussian", s=1.0).rank(
            [[1.0], [0.0]], [1.0]
        )
        assert flows.net_flow == pytest.approx([expected, -expected])

    def test_gaussian_requires_s(self):
        with pytest.raises(ValueError, match="requires s > 0"):
            Promethee(preference="gaussian").rank([[1.0], [0.0]], [1.0])

    def test_v_shape_requires_p_above_q(self):
        with pytest.raises(ValueError, match="strictly greater"):
            Promethee(preference="v-shape", q=0.5, p=0.5).rank(
                [[1.0], [0.0]], [1.0]
            )


class TestCriteriaDirection:
    def test_minimised_criterion_reverses_preference(self):
        X = [[1.0], [0.0]]
        maxed = Promethee(preference="usual").rank(X, [1.0])
        mined = Promethee(preference="usual", criteria_types=["min"]).rank(
            X, [1.0]
        )
        assert mined.net_flow == pytest.approx(-maxed.net_flow)

    def test_numeric_criteria_types(self):
        X = [[1.0], [0.0]]
        a = Promethee(preference="usual", criteria_types=[-1]).rank(X, [1.0])
        b = Promethee(preference="usual", criteria_types=["min"]).rank(X, [1.0])
        assert a.net_flow == pytest.approx(b.net_flow)

    def test_unknown_criterion_type_rejected(self):
        with pytest.raises(ValueError, match="unknown criterion type"):
            Promethee(criteria_types=["sideways"]).rank([[1.0], [0.0]], [1.0])


class TestPerCriterionParameters:
    def test_thresholds_broadcast_per_criterion(self):
        X = np.array([[0.3, 0.3], [0.0, 0.0]])
        # First criterion saturates (p = 0.1), second interpolates to 0.5.
        flows = Promethee(
            preference="v-shape", q=[0.0, 0.1], p=[0.1, 0.5]
        ).rank(X, [0.5, 0.5])
        assert flows.net_flow[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)

    def test_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="length"):
            Promethee(preference="v-shape", p=[0.1, 0.2, 0.3]).rank(
                np.ones((3, 2)), [0.5, 0.5]
            )


class TestOrderAndPartialOrder:
    def test_order_is_best_first(self):
        X = np.array([[3.0], [1.0], [2.0]])
        flows = Promethee(preference="usual").rank(X, [1.0])
        assert flows.order.tolist() == [0, 2, 1]

    def test_partial_order_shape_and_diagonal(self):
        X = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        flows = Promethee(preference="usual").rank(X, [0.5, 0.5])
        relation = Promethee.partial_order(flows)
        assert relation.shape == (3, 3)
        assert all(relation[i, i] == "" for i in range(3))

    def test_dominant_alternative_outranks(self):
        X = np.array([[3.0, 3.0], [1.0, 1.0]])
        flows = Promethee(preference="usual").rank(X, [0.5, 0.5])
        assert Promethee.partial_order(flows)[0, 1] == "P"


class TestValidation:
    def test_requires_two_alternatives(self):
        with pytest.raises(ValueError, match="two alternatives"):
            Promethee().rank([[1.0, 2.0]], [0.5, 0.5])

    def test_weight_length_must_match(self):
        with pytest.raises(ValueError, match="criteria"):
            Promethee().rank(np.ones((3, 4)), [0.5, 0.5])

    def test_negative_weights_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Promethee().rank(np.ones((3, 2)), [-0.5, 1.5])

    def test_unknown_preference_rejected(self):
        with pytest.raises(ValueError, match="unknown preference"):
            Promethee(preference="parabolic").rank([[1.0], [0.0]], [1.0])

    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            Promethee(version="III")

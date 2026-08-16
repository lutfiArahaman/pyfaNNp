import numpy as np
import pytest

from pyfap import (
    FAHP,
    ANNSurrogate,
    DecisionPipeline,
    Promethee,
    minmax_normalize,
    rank_stability,
)
from pyfap.datasets import load_demo


@pytest.fixture
def problem():
    return load_demo()


@pytest.fixture
def normalized(problem):
    return minmax_normalize(problem.decision_matrix)


def build(problem, surrogate=None):
    return DecisionPipeline(
        weights=FAHP(method="geometric_mean"),
        surrogate=surrogate,
        ranker=Promethee(
            preference="v-shape",
            q=0.1,
            p=0.5,
            criteria_types=problem.criteria_types,
        ),
    )


class TestDataset:
    def test_shapes_agree(self, problem):
        n_crit = len(problem.criteria)
        assert problem.judgments.shape == (n_crit, n_crit, 3)
        assert problem.decision_matrix.shape == (
            len(problem.alternatives),
            n_crit,
        )

    def test_normalization_maps_to_unit_range(self, normalized):
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_normalization_preserves_column_order(self, problem, normalized):
        for k in range(problem.decision_matrix.shape[1]):
            raw_order = np.argsort(problem.decision_matrix[:, k])
            norm_order = np.argsort(normalized[:, k])
            assert raw_order.tolist() == norm_order.tolist()


class TestPipelineWithoutSurrogate:
    def test_end_to_end(self, problem, normalized):
        result = build(problem).fit_rank(
            judgments=problem.judgments,
            decision_matrix=normalized,
            alternatives=problem.alternatives,
            criteria=problem.criteria,
        )
        assert result.weights.shape == (4,)
        assert result.weights.sum() == pytest.approx(1.0)
        assert result.net_flow.shape == (6,)
        assert result.net_flow.sum() == pytest.approx(0.0, abs=1e-12)
        assert sorted(result.ranking) == sorted(problem.alternatives)

    def test_ranking_matches_net_flow_order(self, problem, normalized):
        result = build(problem).fit_rank(
            problem.judgments, normalized, alternatives=problem.alternatives
        )
        flows = [
            result.net_flow[result.alternatives.index(a)]
            for a in result.ranking
        ]
        assert flows == sorted(flows, reverse=True)

    def test_consistency_ratio_is_reported(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        assert result.consistency_ratio is not None
        assert result.consistency_ratio < 0.10

    def test_partial_order_shape(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        assert result.partial_order.shape == (6, 6)

    def test_predict_without_surrogate_raises(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        with pytest.raises(RuntimeError, match="without a surrogate"):
            result.predict(normalized)

    def test_rank_before_fit_raises(self, problem):
        with pytest.raises(RuntimeError, match="call fit"):
            build(problem).rank()

    def test_criteria_count_mismatch_raises(self, problem):
        with pytest.raises(ValueError, match="criteria"):
            build(problem).fit(problem.judgments, np.ones((6, 3)))

    def test_label_count_mismatch_raises(self, problem, normalized):
        with pytest.raises(ValueError, match="labels"):
            build(problem).fit(
                problem.judgments, normalized, alternatives=["only", "two"]
            )


class TestStability:
    def test_report_shapes(self, problem, normalized):
        result = build(problem).fit_rank(
            problem.judgments, normalized, alternatives=problem.alternatives
        )
        report = result.stability(n=200, random_state=0)
        assert report.ranks.shape == (200, 6)
        assert report.rank_counts.sum() == 200 * 6
        assert report.top1_frequency.sum() == pytest.approx(1.0)
        assert 0.0 <= report.rank_reversal_rate <= 1.0

    def test_every_simulation_is_a_permutation(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        report = result.stability(n=50, random_state=1)
        for row in report.ranks:
            assert sorted(row.tolist()) == list(range(1, 7))

    def test_zero_perturbation_limit_has_no_reversals(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        report = result.stability(n=100, scale=1e-9, random_state=0)
        assert report.rank_reversal_rate == pytest.approx(0.0)

    def test_reproducible_under_seed(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        a = result.stability(n=100, random_state=7)
        b = result.stability(n=100, random_state=7)
        assert np.array_equal(a.ranks, b.ranks)

    def test_dirichlet_method_runs(self, problem, normalized):
        result = build(problem).fit_rank(problem.judgments, normalized)
        report = result.stability(n=50, method="dirichlet", random_state=0)
        assert report.ranks.shape == (50, 6)

    def test_summary_is_printable(self, problem, normalized):
        result = build(problem).fit_rank(
            problem.judgments, normalized, alternatives=problem.alternatives
        )
        text = result.stability(n=20, random_state=0).summary()
        assert "rank reversal rate" in text
        assert problem.alternatives[0] in text

    def test_unknown_method_rejected(self, problem, normalized):
        with pytest.raises(ValueError, match="unknown perturbation method"):
            rank_stability(
                normalized,
                np.full(4, 0.25),
                Promethee(preference="usual"),
                n=5,
                method="uniform",
            )


class TestSurrogate:
    """The surrogate is the coupling under test; skipped without sklearn."""

    def test_trains_and_predicts(self, problem, normalized):
        pytest.importorskip("sklearn")
        surrogate = ANNSurrogate(hidden=(32, 16), epochs=2000, random_state=0)
        result = build(problem, surrogate).fit_rank(
            problem.judgments, normalized, alternatives=problem.alternatives
        )
        predicted = result.predict(normalized)
        assert predicted.shape == (6,)
        assert result.surrogate_score is not None

    def test_tracks_the_flows_it_was_trained_on(self, problem, normalized):
        """With only six training points this is a memorisation check, not a
        generalisation claim.

        Replace it with a held-out split once a larger alternative set is
        available -- that is the experiment the paper's example section needs,
        and this assertion is deliberately weak so that it does not stand in
        for it.
        """
        pytest.importorskip("sklearn")
        surrogate = ANNSurrogate(hidden=(64, 32), epochs=5000, random_state=0)
        result = build(problem, surrogate).fit_rank(
            problem.judgments, normalized, alternatives=problem.alternatives
        )
        predicted = result.predict(normalized)

        # Correlation between predicted and exact flows, not exact rank
        # agreement: a six-point fit is too small to demand the latter.
        correlation = np.corrcoef(predicted, result.net_flow)[0, 1]
        assert correlation > 0.9
        assert np.argmax(predicted) == np.argmax(result.net_flow)

    def test_predict_before_fit_raises(self):
        pytest.importorskip("sklearn")
        with pytest.raises(RuntimeError, match="not been fitted"):
            ANNSurrogate().predict(np.ones((2, 4)))

    def test_torch_backend_not_implemented(self):
        with pytest.raises(NotImplementedError, match="torch"):
            ANNSurrogate(backend="torch").fit(np.ones((4, 2)), np.zeros(4))

    def test_constant_criterion_does_not_divide_by_zero(self):
        pytest.importorskip("sklearn")
        X = np.column_stack([np.ones(8), np.linspace(0, 1, 8)])
        y = np.linspace(-1, 1, 8)
        model = ANNSurrogate(epochs=200, random_state=0).fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="rows"):
            ANNSurrogate().fit(np.ones((4, 2)), np.zeros(3))

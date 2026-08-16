"""Validation against the worked examples in the source papers.

These tests are SKIPPED until the expected values are transcribed from the
papers themselves. They are deliberately left unfilled rather than populated
with plausible-looking numbers: a validation test whose "published" values
were reconstructed from memory is worse than no test at all, because it
looks like evidence and is not. Nothing here should be cited in the paper
until the constants below come from the printed tables.

To activate a test
------------------
1. Open the paper named in the class docstring and find the worked example.
2. Fill in the module-level constant with the input matrix and the published
   result, to whatever precision the paper prints.
3. Record the page, table or equation number in ``SOURCE`` so the value can
   be audited later.
4. Set ``TOLERANCE`` to match the printed precision -- if the paper gives
   three decimals, do not assert agreement to ten.
5. Delete nothing else; the skip lifts automatically once the constant is
   no longer ``None``.

If a published example disagrees with this implementation, that is a
finding, not a test to relax. Record it in the README rather than widening
the tolerance to make it pass.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyfap import FAHP, Promethee

# ===========================================================================
# Chang (1996), extent analysis
# European Journal of Operational Research 95(3), 649-655
# ===========================================================================

CHANG_SOURCE = ""  # e.g. "Chang (1996), Table 2, p. 653"

# Fuzzy pairwise comparison matrix from the paper, shape (n, n, 3).
CHANG_JUDGMENTS: np.ndarray | None = None

# Criterion weights the paper reports for that matrix, shape (n,).
CHANG_EXPECTED_WEIGHTS: np.ndarray | None = None

CHANG_TOLERANCE = 1e-3


@pytest.mark.skipif(
    CHANG_JUDGMENTS is None,
    reason="expected values not yet transcribed from Chang (1996)",
)
class TestChang1996:
    """Chang's extent analysis, against the paper's own worked example."""

    def test_weights_match_the_paper(self):
        got = FAHP(
            method="extent_analysis", consistency_check=False
        ).derive(CHANG_JUDGMENTS)
        assert got == pytest.approx(
            CHANG_EXPECTED_WEIGHTS, abs=CHANG_TOLERANCE
        ), f"disagrees with {CHANG_SOURCE}"

    def test_source_is_recorded(self):
        assert CHANG_SOURCE, "record where the values came from"


# ===========================================================================
# Buckley (1985), geometric mean method
# Fuzzy Sets and Systems 17(3), 233-247
# ===========================================================================

BUCKLEY_SOURCE = ""

BUCKLEY_JUDGMENTS: np.ndarray | None = None
BUCKLEY_EXPECTED_WEIGHTS: np.ndarray | None = None
BUCKLEY_TOLERANCE = 1e-3


@pytest.mark.skipif(
    BUCKLEY_JUDGMENTS is None,
    reason="expected values not yet transcribed from Buckley (1985)",
)
class TestBuckley1985:
    """Buckley's fuzzy hierarchical analysis, against the paper's example."""

    def test_weights_match_the_paper(self):
        got = FAHP(
            method="geometric_mean", consistency_check=False
        ).derive(BUCKLEY_JUDGMENTS)
        assert got == pytest.approx(
            BUCKLEY_EXPECTED_WEIGHTS, abs=BUCKLEY_TOLERANCE
        ), f"disagrees with {BUCKLEY_SOURCE}"

    def test_source_is_recorded(self):
        assert BUCKLEY_SOURCE, "record where the values came from"


# ===========================================================================
# Brans & Vincke (1985) / Brans, Vincke & Mareschal (1986), PROMETHEE
# Management Science 31(6), 647-656 / EJOR 24(2), 228-238
# ===========================================================================

BRANS_SOURCE = ""

# Decision matrix from the paper, shape (n_alternatives, n_criteria).
BRANS_DECISION_MATRIX: np.ndarray | None = None

# Criterion weights the paper uses.
BRANS_WEIGHTS: np.ndarray | None = None

# Preference function name per criterion, and its thresholds. The paper's
# type numbers map to this package's names as:
#   type 1 -> "usual"      type 2 -> "u-shape"    type 3 -> "v-shape" (q = 0)
#   type 4 -> "level"      type 5 -> "v-shape"    type 6 -> "gaussian"
BRANS_PREFERENCE: list | None = None
BRANS_Q: list | None = None
BRANS_P: list | None = None
BRANS_S: list | None = None
BRANS_CRITERIA_TYPES: list | None = None

# Published flows, shape (n_alternatives,).
BRANS_EXPECTED_PHI_PLUS: np.ndarray | None = None
BRANS_EXPECTED_PHI_MINUS: np.ndarray | None = None
BRANS_EXPECTED_NET_FLOW: np.ndarray | None = None

# Published PROMETHEE II ranking, best to worst, as zero-based indices.
BRANS_EXPECTED_RANKING: list | None = None

BRANS_TOLERANCE = 1e-3


@pytest.mark.skipif(
    BRANS_DECISION_MATRIX is None,
    reason="expected values not yet transcribed from Brans & Vincke (1985)",
)
class TestBransVincke1985:
    """PROMETHEE flows and ranking, against the founding paper's example."""

    def _flows(self):
        return Promethee(
            version="II",
            preference=BRANS_PREFERENCE,
            q=BRANS_Q,
            p=BRANS_P,
            s=BRANS_S,
            criteria_types=BRANS_CRITERIA_TYPES,
        ).rank(BRANS_DECISION_MATRIX, BRANS_WEIGHTS)

    def test_positive_flow_matches_the_paper(self):
        assert self._flows().positive_flow == pytest.approx(
            BRANS_EXPECTED_PHI_PLUS, abs=BRANS_TOLERANCE
        ), f"disagrees with {BRANS_SOURCE}"

    def test_negative_flow_matches_the_paper(self):
        assert self._flows().negative_flow == pytest.approx(
            BRANS_EXPECTED_PHI_MINUS, abs=BRANS_TOLERANCE
        ), f"disagrees with {BRANS_SOURCE}"

    def test_net_flow_matches_the_paper(self):
        assert self._flows().net_flow == pytest.approx(
            BRANS_EXPECTED_NET_FLOW, abs=BRANS_TOLERANCE
        ), f"disagrees with {BRANS_SOURCE}"

    def test_ranking_matches_the_paper(self):
        assert self._flows().order.tolist() == BRANS_EXPECTED_RANKING

    def test_source_is_recorded(self):
        assert BRANS_SOURCE, "record where the values came from"


# ===========================================================================
# Guard: make the unfilled state visible rather than silently green.
# ===========================================================================


def test_validation_status_is_reported():
    """Always runs. Fails nothing, but warns about what is still unvalidated.

    A warning rather than a print, so the notice survives into pytest's
    warnings summary and CI logs instead of being swallowed with the captured
    stdout of a passing test. Remove this test once all three sources are
    transcribed.
    """
    pending = [
        name
        for name, value in [
            ("Chang (1996) extent analysis", CHANG_JUDGMENTS),
            ("Buckley (1985) geometric mean", BUCKLEY_JUDGMENTS),
            ("Brans & Vincke (1985) PROMETHEE", BRANS_DECISION_MATRIX),
        ]
        if value is None
    ]
    if pending:
        warnings.warn(
            "NOT YET VALIDATED against published worked examples: "
            + "; ".join(pending)
            + ". See tests/test_published.py for how to fill these in.",
            UserWarning,
            stacklevel=1,
        )

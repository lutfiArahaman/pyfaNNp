"""Cross-check pyFAP against independent implementations.

    python tools/crosscheck.py

Runs the same decision problem through pyFAP, pyDecision and pymcdm and
compares the results. This is external validation: agreement with a
separately written implementation is evidence that pyFAP computes what the
published methods specify, which is the gap `tests/test_published.py` leaves
open until the source papers are transcribed.

Disagreement is a finding either way. These methods have well-known
variants -- Chang's extent analysis versus Buckley's geometric mean, the
six Brans preference functions versus subsets of them, normalised versus raw
criteria -- so two implementations differing does not by itself mean one is
wrong. Read what each is doing before concluding anything.

Run from the `Related work survey` workflow, which installs the comparison
packages.
"""

from __future__ import annotations

import inspect
import sys
import traceback

import numpy as np

from pyfap import FAHP, Promethee, from_saaty

RULE = "=" * 78


def describe(label, value, limit=700):
    """Print the shape of a returned object without assuming what it is."""
    print(f"  {label}:")
    print(f"    type   {type(value).__name__}")
    if isinstance(value, np.ndarray):
        print(f"    shape  {value.shape}   dtype {value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"    len    {len(value)}")
        if value and isinstance(value[0], (list, tuple, np.ndarray)):
            print(f"    first element: type {type(value[0]).__name__}, "
                  f"len {len(value[0])}")
    text = repr(value)
    print(f"    value  {text[:limit]}{' ...' if len(text) > limit else ''}")


def show_source(func, name, lines=70):
    print(f"\n--- source of {name} (first {lines} lines) ---")
    try:
        source = inspect.getsource(func)
    except Exception as exc:
        print(f"  unavailable: {exc!r}")
        return
    for i, line in enumerate(source.splitlines()[:lines]):
        print(f"  {i + 1:>3} | {line}")


# ---------------------------------------------------------------------------
# The shared problem. Deliberately small, all criteria maximised, so that
# implementations differing on criterion direction cannot confound the
# comparison.
# ---------------------------------------------------------------------------

SAATY = np.array(
    [
        [1.0,     3.0,     5.0],
        [1 / 3.0, 1.0,     2.0],
        [1 / 5.0, 1 / 2.0, 1.0],
    ]
)

DECISION = np.array(
    [
        [0.80, 0.30, 0.55],
        [0.55, 0.75, 0.40],
        [0.35, 0.60, 0.90],
        [0.65, 0.45, 0.25],
        [0.20, 0.85, 0.70],
    ]
)

Q, P = 0.10, 0.50


def section_fuzzy_ahp():
    print(f"\n{RULE}\nSECTION 1 -- fuzzy AHP weights\n{RULE}")

    judgments = from_saaty(SAATY, spread=1.0)

    geometric = FAHP(method="geometric_mean", consistency_check=True)
    w_geometric = geometric.derive(judgments)
    extent = FAHP(method="extent_analysis", consistency_check=False)
    w_extent = extent.derive(judgments)

    print(f"\n  pyfap geometric_mean   {np.round(w_geometric, 6)}")
    print(f"  pyfap consistency ratio {geometric.consistency_ratio_:.6f}")
    print(f"  pyfap extent_analysis  {np.round(w_extent, 6)}")

    try:
        from pyDecision.algorithm import fuzzy_ahp_method
    except Exception as exc:
        print(f"\n  pyDecision unavailable: {exc!r}")
        return

    show_source(fuzzy_ahp_method, "fuzzy_ahp_method")

    # pyDecision takes a list of rows of (l, m, u) tuples.
    dataset = [
        [tuple(float(x) for x in judgments[i, j]) for j in range(3)]
        for i in range(3)
    ]
    print("\n  input passed to pyDecision:")
    for row in dataset:
        print(f"    {row}")

    try:
        result = fuzzy_ahp_method(dataset)
    except Exception:
        print("\n  fuzzy_ahp_method raised:")
        traceback.print_exc(limit=3)
        return

    print("\n  pyDecision returned:")
    if isinstance(result, tuple):
        print(f"    a {len(result)}-tuple")
        for i, item in enumerate(result):
            describe(f"  [{i}]", item)
    else:
        describe("  result", result)


def section_promethee():
    print(f"\n{RULE}\nSECTION 2 -- PROMETHEE II\n{RULE}")

    weights = np.array([0.55, 0.30, 0.15])

    flows = Promethee(
        version="II", preference="v-shape", q=Q, p=P
    ).rank(DECISION, weights)
    print(f"\n  pyfap phi+     {np.round(flows.positive_flow, 6)}")
    print(f"  pyfap phi-     {np.round(flows.negative_flow, 6)}")
    print(f"  pyfap phi      {np.round(flows.net_flow, 6)}")
    print(f"  pyfap ranking  {(flows.order + 1).tolist()}  (1-based)")

    # -- pyDecision ---------------------------------------------------------
    try:
        from pyDecision.algorithm import promethee_ii
    except Exception as exc:
        print(f"\n  pyDecision unavailable: {exc!r}")
    else:
        show_source(promethee_ii, "promethee_ii", lines=25)
        try:
            from pyDecision.algorithm.promethee import preference_degree
            show_source(preference_degree, "preference_degree", lines=60)
        except Exception:
            for candidate in ("promethee_i", "promethee_ii"):
                try:
                    module = sys.modules[
                        getattr(
                            __import__("pyDecision.algorithm", fromlist=[candidate]),
                            candidate,
                        ).__module__
                    ]
                    show_source(module.preference_degree, "preference_degree", 60)
                    break
                except Exception:
                    continue

        # The matrix must be an ndarray: promethee_ii reads .shape off it.
        for codes in (["t5"] * 3, ["t3"] * 3, ["t1"] * 3):
            print(f"\n  attempting F={codes}")
            try:
                result = promethee_ii(
                    DECISION,
                    W=weights.tolist(),
                    Q=[Q] * 3,
                    S=[0.0] * 3,
                    P=[P] * 3,
                    F=codes,
                    sort=False,
                    graph=False,
                    verbose=False,
                )
            except Exception as exc:
                print(f"    raised {type(exc).__name__}: {exc}")
                continue
            describe("    result", result)
            print(f"    column 1 (phi): {np.round(np.asarray(result)[:, 1], 6)}")
            break

    # -- pymcdm -------------------------------------------------------------
    try:
        from pymcdm.methods import PROMETHEE_II as PymcdmPromethee
    except Exception as exc:
        print(f"\n  pymcdm unavailable: {exc!r}")
        return

    types = np.ones(3)
    for name in ("vshape_2", "vshape", "usual"):
        print(f"\n  attempting pymcdm preference_function={name!r}")
        try:
            method = PymcdmPromethee(name, p=[P] * 3, q=[Q] * 3)
            result = method(DECISION, weights, types)
        except Exception as exc:
            print(f"    raised {type(exc).__name__}: {exc}")
            continue
        describe("    result", result)
        break


def main():
    print(f"python {sys.version.split()[0]}")
    for module in ("pyDecision", "pymcdm"):
        try:
            import importlib.metadata as meta
            print(f"{module} {meta.version(module)}")
        except Exception:
            print(f"{module} version unknown")

    section_fuzzy_ahp()
    section_promethee()

    print(f"\n{RULE}")
    print("This run is for discovering the foreign APIs. Once the return")
    print("shapes above are known, the comparison becomes an assertion and")
    print("moves into the test suite.")


if __name__ == "__main__":
    main()

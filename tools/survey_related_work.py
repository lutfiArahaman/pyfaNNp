"""Survey the packages cited in the Related Work section.

    python tools/survey_related_work.py

Walks each installed package, collects every public name it exposes, and
reports which of them match the capabilities pyFAP claims to combine. The
point is to settle, with evidence rather than recollection, whether any
existing package already chains a fuzzy AHP weight derivation to a PROMETHEE
outranking -- which is the claim in Related Work a reviewer is most likely
to check.

This is a one-off investigation kept in the repository so the comparison in
the paper is reproducible and dated. It is run from a manually dispatched
workflow, not on every push.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pkgutil
import sys
import traceback

# (distribution name on PyPI, top-level import name)
PACKAGES = [
    ("pymcdm", "pymcdm"),
    ("scikit-criteria", "skcriteria"),
    ("pyDecision", "pyDecision"),
    ("scikit-fuzzy", "skfuzzy"),
    ("simpful", "simpful"),
]

# Capability -> substrings that would appear in a name providing it.
CAPABILITIES = {
    "fuzzy AHP": ("fuzzy_ahp", "fahp", "fuzzy_analytic", "fuzzyahp"),
    "AHP (crisp)": ("ahp", "analytic_hierarchy"),
    "PROMETHEE": ("promethee",),
    "fuzzy numbers": ("triangular", "trapezoid", "tfn", "defuzz"),
    "neural / ML": ("neural", "mlp", "network", "regressor", "classifier"),
    "sensitivity": ("sensitivity", "robust", "perturb", "monte_carlo"),
}


def collect_names(import_name):
    """Every public attribute reachable from the package and its submodules."""
    names = set()
    modules_seen = 0
    failures = []

    try:
        root = importlib.import_module(import_name)
    except Exception as exc:
        return None, 0, [f"import {import_name}: {exc!r}"]

    modules_seen += 1
    names.update(n for n in dir(root) if not n.startswith("_"))

    paths = getattr(root, "__path__", None)
    if paths:
        for info in pkgutil.walk_packages(paths, prefix=root.__name__ + "."):
            try:
                module = importlib.import_module(info.name)
            except Exception as exc:  # optional deps, py-version guards, etc.
                failures.append(f"{info.name}: {type(exc).__name__}")
                continue
            modules_seen += 1
            names.add(info.name.rsplit(".", 1)[-1])
            names.update(n for n in dir(module) if not n.startswith("_"))

    return names, modules_seen, failures


def match(names, needles):
    lowered = {n: n.lower() for n in names}
    hits = sorted(
        original
        for original, low in lowered.items()
        if any(needle in low for needle in needles)
    )
    return hits


def metadata(dist_name):
    try:
        meta = importlib.metadata.metadata(dist_name)
    except Exception:
        return {}
    fields = {}
    for key in ("Version", "Summary", "Home-page", "Author", "License"):
        value = meta.get(key)
        if value:
            fields[key] = value
    urls = meta.get_all("Project-URL") or []
    if urls:
        fields["Project-URL"] = "; ".join(urls)
    return fields


def main():
    print(f"python {sys.version.split()[0]}")
    print("=" * 78)

    summary = {}

    for dist_name, import_name in PACKAGES:
        print(f"\n### {dist_name}  (import {import_name})")
        print("-" * 78)

        for key, value in metadata(dist_name).items():
            print(f"  {key:<12} {value}")

        names, modules_seen, failures = collect_names(import_name)
        if names is None:
            print(f"  NOT IMPORTABLE: {failures[0]}")
            summary[dist_name] = {}
            continue

        print(f"  modules walked: {modules_seen}, public names: {len(names)}")
        if failures:
            print(f"  submodules that failed to import: {len(failures)}")
            for line in failures[:8]:
                print(f"    - {line}")

        found = {}
        for capability, needles in CAPABILITIES.items():
            hits = match(names, needles)
            found[capability] = hits
            marker = "YES" if hits else " no"
            shown = ", ".join(hits[:12])
            if len(hits) > 12:
                shown += f", ... (+{len(hits) - 12} more)"
            print(f"  [{marker}] {capability:<14} {shown}")
        summary[dist_name] = found

    # The claim under test -------------------------------------------------
    print("\n" + "=" * 78)
    print("THE CLAIM UNDER TEST")
    print("Does any package expose BOTH a fuzzy AHP weight derivation AND")
    print("a PROMETHEE outranking?")
    print("-" * 78)
    for dist_name, found in summary.items():
        if not found:
            print(f"  {dist_name:<18} could not be inspected")
            continue
        fuzzy = bool(found.get("fuzzy AHP"))
        promethee = bool(found.get("PROMETHEE"))
        verdict = "BOTH" if (fuzzy and promethee) else (
            "PROMETHEE only" if promethee else
            "fuzzy AHP only" if fuzzy else "neither"
        )
        print(f"  {dist_name:<18} {verdict}")

    print("\nA 'BOTH' above means the Related Work paragraph must concede that")
    print("package explicitly. Name matching is a coarse instrument: read the")
    print("hit lists, then read that package's docs before writing the prose.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

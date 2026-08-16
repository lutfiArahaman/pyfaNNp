"""Scaling experiment: what the neural surrogate is for.

    python examples/scaling.py [--eval-size 2000] [--repeats 5]

Produces ``figures/scaling.{png,pdf}`` and ``figures/scaling_data.csv``.

The question this answers
------------------------
A reader can reasonably ask why a neural network belongs between fuzzy AHP
and PROMETHEE at all: on a small alternative set the surrogate reproduces an
outranking that has already been computed exactly, which is no use to
anyone. The answer is that the exact outranking is quadratic in the number
of alternatives and the surrogate is linear, so past some set size the
surrogate is the only affordable way to score alternatives at all.

Why a surrogate can transfer between alternative sets
-----------------------------------------------------
The obvious objection is that PROMETHEE flows are defined relative to the
set being ranked, so a network trained on one set should not generalise to
another. The positive flow, however, is an average:

    phi+(a) = 1/(n-1) * sum_{b != a} pi(a, b)

If the other alternatives are drawn independently from some population,
this is an unbiased estimator of the population quantity E_b[pi(a, b)],
whatever n is. The surrogate is therefore learning a well-defined function
of a criterion vector -- its expected preference against the population --
not memorising one particular set. Panel (a) checks this empirically: the
spread of the estimate falls as n^(-1/2), which is what an average of
independent draws does.

The three panels
----------------
(a) Convergence of the flow estimate, validating the argument above.
(b) Rank fidelity of the surrogate against the exact outranking, as a
    function of how many alternatives it was trained on.
(c) Wall-clock cost of the exact outranking versus train-plus-predict,
    with the crossover marked.

All synthetic: the population is sampled, not observed. That is appropriate
here because the claim being tested is about computational scaling, which
does not depend on the data being real. The claims in the Example section
of the paper do, and need a published benchmark.
"""

from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pyfap import FAHP, ANNSurrogate, Promethee
from pyfap.datasets import load_demo

from _style import BLUE, GREEN, GRID, INK, MUTED, ORANGE, legend, style_axis, title

N_CRITERIA = 4


# ---------------------------------------------------------------------------
# Synthetic alternative population
# ---------------------------------------------------------------------------


def make_loadings(rng, n_criteria=N_CRITERIA, n_factors=2):
    """A fixed factor structure, so criteria correlate as they do in practice
    rather than being independent noise."""
    return rng.normal(size=(n_criteria, n_factors))


def sample_population(rng, n, loadings, noise=0.6):
    """Draw ``n`` alternatives with correlated criterion values in (0, 1)."""
    factors = rng.standard_normal((n, loadings.shape[1]))
    latent = factors @ loadings.T
    latent += rng.standard_normal((n, loadings.shape[0])) * noise
    latent /= latent.std(axis=0, keepdims=True) + 1e-12
    return 1.0 / (1.0 + np.exp(-latent))  # logistic squash into (0, 1)


def make_ranker():
    return Promethee(version="II", preference="v-shape", q=0.05, p=0.30)


def derive_weights():
    """Weights from the demo judgements; the geometric mean is used because
    extent analysis zeroes a criterion on this matrix."""
    problem = load_demo()
    return FAHP(method="geometric_mean", consistency_check=False).derive(
        problem.judgments
    )


# ---------------------------------------------------------------------------
# Rank agreement, without a scipy dependency
# ---------------------------------------------------------------------------


def spearman(a, b):
    """Spearman rho = Pearson correlation of the ranks."""
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def top_k_overlap(predicted, exact, k):
    """Proportion of the exact top-k that the surrogate also puts in its
    top-k. This is what a decision maker actually reads off a ranking."""
    k = min(k, predicted.size)
    best_predicted = set(np.argsort(-predicted)[:k].tolist())
    best_exact = set(np.argsort(-exact)[:k].tolist())
    return len(best_predicted & best_exact) / float(k)


# ---------------------------------------------------------------------------
# (a) Does the flow estimate converge?
# ---------------------------------------------------------------------------


def experiment_convergence(rng, weights, ranker, loadings, sizes, repeats):
    """Spread of phi(a0) across independent draws of the other alternatives.

    If phi is an average of independent terms, its standard deviation should
    fall like n^(-1/2). If it does, a surrogate trained on one sample is
    estimating a population quantity rather than memorising a set.
    """
    probe = sample_population(rng, 1, loadings)
    rows = []
    for n in sizes:
        values = []
        for _ in range(repeats):
            others = sample_population(rng, n - 1, loadings)
            X = np.vstack([probe, others])
            values.append(ranker.rank(X, weights).net_flow[0])
        values = np.asarray(values)
        rows.append(
            {"n": n, "mean": float(values.mean()), "std": float(values.std(ddof=1))}
        )
        print(f"  n={n:<6d} phi={values.mean():+.5f}  sd={values.std(ddof=1):.5f}")
    return rows


# ---------------------------------------------------------------------------
# (b) How faithful is the surrogate, as a function of training size?
# ---------------------------------------------------------------------------


def experiment_fidelity(
    rng, weights, ranker, loadings, train_sizes, eval_size, repeats, hidden, epochs
):
    rows = []
    for n_train in train_sizes:
        rhos, overlaps, scores = [], [], []
        for repeat in range(repeats):
            X_train = sample_population(rng, n_train, loadings)
            phi_train = ranker.rank(X_train, weights).net_flow

            surrogate = ANNSurrogate(
                hidden=hidden, epochs=epochs, random_state=repeat
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn convergence chatter
                surrogate.fit(X_train, phi_train)

            X_eval = sample_population(rng, eval_size, loadings)
            phi_exact = ranker.rank(X_eval, weights).net_flow
            phi_hat = surrogate.predict(X_eval)

            rhos.append(spearman(phi_hat, phi_exact))
            overlaps.append(top_k_overlap(phi_hat, phi_exact, k=50))
            scores.append(surrogate.train_score_)

        row = {
            "n_train": n_train,
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1)) if repeats > 1 else 0.0,
            "top50_mean": float(np.mean(overlaps)),
            "top50_std": float(np.std(overlaps, ddof=1)) if repeats > 1 else 0.0,
            "train_r2": float(np.mean(scores)),
        }
        rows.append(row)
        print(
            f"  n_train={n_train:<6d} rho={row['rho_mean']:.4f}"
            f" +/-{row['rho_std']:.4f}   top-50 overlap={row['top50_mean']:.1%}"
        )
    return rows


# ---------------------------------------------------------------------------
# (c) What does each approach cost?
# ---------------------------------------------------------------------------


def experiment_cost(
    rng, weights, ranker, loadings, exact_sizes, surrogate_sizes,
    n_train, hidden, epochs,
):
    """Exact outranking is O(N^2); train-plus-predict is a fixed cost plus
    O(N). The fixed cost is charged in full to the surrogate: it includes the
    exact outranking of the training subsample."""
    exact_rows = []
    for n in exact_sizes:
        X = sample_population(rng, n, loadings)
        start = time.perf_counter()
        ranker.rank(X, weights)
        elapsed = time.perf_counter() - start
        exact_rows.append({"n": n, "seconds": elapsed})
        print(f"  exact      N={n:<7d} {elapsed:8.4f} s")

    # The fixed cost, paid once regardless of how many alternatives follow.
    X_train = sample_population(rng, n_train, loadings)
    start = time.perf_counter()
    phi_train = ranker.rank(X_train, weights).net_flow
    surrogate = ANNSurrogate(hidden=hidden, epochs=epochs, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate.fit(X_train, phi_train)
    setup = time.perf_counter() - start
    print(f"  surrogate setup (n_train={n_train}): {setup:.4f} s")

    surrogate_rows = []
    for n in surrogate_sizes:
        X = sample_population(rng, n, loadings)
        start = time.perf_counter()
        surrogate.predict(X)
        elapsed = time.perf_counter() - start
        surrogate_rows.append({"n": n, "seconds": setup + elapsed, "predict": elapsed})
        print(f"  surrogate  N={n:<7d} {setup + elapsed:8.4f} s "
              f"(predict {elapsed:.4f} s)")

    return exact_rows, surrogate_rows, setup


def find_crossover(exact_rows, surrogate_rows):
    """Smallest N at which the surrogate becomes the cheaper option.

    Uses the quadratic fit through the exact timings so the answer is not an
    artefact of which N values happened to be measured.
    """
    ns = np.array([r["n"] for r in exact_rows], dtype=float)
    ts = np.array([r["seconds"] for r in exact_rows], dtype=float)
    if ns.size < 2:
        return None
    # t = c * N^2, fitted through the origin on the largest points.
    c = float(np.sum(ts * ns**2) / np.sum(ns**4))

    setup = surrogate_rows[0]["seconds"] - surrogate_rows[0]["predict"]
    per_item = float(
        np.mean([r["predict"] / r["n"] for r in surrogate_rows])
    )
    # c*N^2 = setup + per_item*N
    disc = per_item**2 + 4 * c * setup
    return float((per_item + np.sqrt(disc)) / (2 * c))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw(convergence, fidelity, exact_rows, surrogate_rows, crossover, path_stem):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # (a) convergence -------------------------------------------------------
    ax = axes[0]
    ns = np.array([r["n"] for r in convergence], dtype=float)
    sds = np.array([r["std"] for r in convergence], dtype=float)
    ax.loglog(ns, sds, "o-", color=BLUE, linewidth=2.0, markersize=6,
              label="observed spread")
    reference = sds[0] * (ns / ns[0]) ** -0.5
    ax.loglog(ns, reference, "--", color=MUTED, linewidth=1.4,
              label=r"$n^{-1/2}$")
    ax.set_xlabel("Alternatives in the set, $n$", fontsize=10, color=INK)
    ax.set_ylabel(r"SD of $\phi$ estimate", fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)
    legend(ax, loc="upper right")
    title(ax, "(a) The flow estimate converges")

    # (b) fidelity ----------------------------------------------------------
    ax = axes[1]
    xs = np.array([r["n_train"] for r in fidelity], dtype=float)
    rho = np.array([r["rho_mean"] for r in fidelity])
    rho_sd = np.array([r["rho_std"] for r in fidelity])
    top = np.array([r["top50_mean"] for r in fidelity])
    top_sd = np.array([r["top50_std"] for r in fidelity])

    ax.errorbar(xs, rho, yerr=rho_sd, fmt="o-", color=BLUE, linewidth=2.0,
                markersize=6, capsize=3, label=r"Spearman $\rho$")
    ax.errorbar(xs, top, yerr=top_sd, fmt="s--", color=ORANGE, linewidth=2.0,
                markersize=6, capsize=3, label="top-50 overlap")
    ax.set_xscale("log")
    ax.set_xlabel("Alternatives used for training", fontsize=10, color=INK)
    ax.set_ylabel("Agreement with exact ranking", fontsize=10, color=INK)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)
    legend(ax, loc="lower right")
    title(ax, "(b) Fidelity against the exact outranking")

    # (c) cost --------------------------------------------------------------
    ax = axes[2]
    ex_n = [r["n"] for r in exact_rows]
    ex_t = [r["seconds"] for r in exact_rows]
    su_n = [r["n"] for r in surrogate_rows]
    su_t = [r["seconds"] for r in surrogate_rows]

    ax.loglog(ex_n, ex_t, "o-", color=BLUE, linewidth=2.0, markersize=6,
              label=r"exact outranking, $O(N^2)$")
    ax.loglog(su_n, su_t, "s--", color=GREEN, linewidth=2.0, markersize=6,
              label=r"train + predict, $O(N)$")

    if crossover is not None and min(su_n) < crossover < max(su_n) * 4:
        ax.axvline(crossover, color=MUTED, linewidth=1.2, linestyle=":")
        ax.annotate(
            f"crossover\n$N \\approx {crossover:,.0f}$",
            xy=(crossover, min(ex_t)), xytext=(6, 4),
            textcoords="offset points", fontsize=8, color=INK,
        )

    ax.set_xlabel("Alternatives ranked, $N$", fontsize=10, color=INK)
    ax.set_ylabel("Wall-clock seconds", fontsize=10, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    style_axis(ax)
    legend(ax, loc="upper left")
    title(ax, "(c) Cost of ranking $N$ alternatives")

    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = f"{path_stem}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


def write_table(path, convergence, fidelity, exact_rows, surrogate_rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "x", "quantity", "value"])
        for row in convergence:
            writer.writerow(["a", "phi estimate", row["n"], "sd", f"{row['std']:.8f}"])
            writer.writerow(["a", "phi estimate", row["n"], "mean", f"{row['mean']:.8f}"])
        for row in fidelity:
            writer.writerow(
                ["b", "spearman", row["n_train"], "mean", f"{row['rho_mean']:.6f}"]
            )
            writer.writerow(
                ["b", "spearman", row["n_train"], "sd", f"{row['rho_std']:.6f}"]
            )
            writer.writerow(
                ["b", "top50", row["n_train"], "mean", f"{row['top50_mean']:.6f}"]
            )
        for row in exact_rows:
            writer.writerow(["c", "exact", row["n"], "seconds", f"{row['seconds']:.6f}"])
        for row in surrogate_rows:
            writer.writerow(
                ["c", "surrogate", row["n"], "seconds", f"{row['seconds']:.6f}"]
            )
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Surrogate scaling experiment")
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--train-subsample", type=int, default=250)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    weights = derive_weights()
    ranker = make_ranker()
    loadings = make_loadings(rng)
    hidden = tuple(args.hidden)

    print("criterion weights:", np.round(weights, 4))

    print("\n(a) convergence of the flow estimate")
    convergence = experiment_convergence(
        rng, weights, ranker, loadings,
        sizes=[10, 25, 50, 100, 250, 500, 1000], repeats=40,
    )

    print("\n(b) surrogate fidelity vs training size")
    fidelity = experiment_fidelity(
        rng, weights, ranker, loadings,
        train_sizes=[25, 50, 100, 250, 500, 1000],
        eval_size=args.eval_size, repeats=args.repeats,
        hidden=hidden, epochs=args.epochs,
    )

    print("\n(c) wall-clock cost")
    exact_rows, surrogate_rows, setup = experiment_cost(
        rng, weights, ranker, loadings,
        exact_sizes=[100, 250, 500, 1000, 1500, 2000, 3000],
        surrogate_sizes=[100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000],
        n_train=args.train_subsample, hidden=hidden, epochs=args.epochs,
    )
    crossover = find_crossover(exact_rows, surrogate_rows)

    draw(
        convergence, fidelity, exact_rows, surrogate_rows, crossover,
        str(args.outdir / "scaling"),
    )
    write_table(
        args.outdir / "scaling_data.csv",
        convergence, fidelity, exact_rows, surrogate_rows,
    )

    print("\n--- numbers for the paper ---")

    # Report the fitted log-log slope rather than an endpoint ratio: the
    # smallest set sizes are noisy and boundary-affected, so anchoring on
    # them misstates the rate. The prediction is a slope of -1/2.
    ns = np.array([r["n"] for r in convergence], dtype=float)
    sds = np.array([r["std"] for r in convergence], dtype=float)
    slope, _ = np.polyfit(np.log(ns), np.log(sds), 1)
    print(f"convergence: fitted slope {slope:+.3f} (n^(-1/2) predicts -0.500)")
    tail = convergence[1:]
    ratio = tail[0]["std"] / tail[-1]["std"]
    expected = np.sqrt(tail[-1]["n"] / tail[0]["n"])
    print(
        f"  excluding the smallest set: spread falls {ratio:.1f}x from "
        f"n={tail[0]['n']} to n={tail[-1]['n']}; n^(-1/2) predicts "
        f"{expected:.1f}x"
    )
    for row in fidelity:
        print(
            f"n_train={row['n_train']:<5d} rho={row['rho_mean']:.4f}  "
            f"top-50 overlap={row['top50_mean']:.1%}"
        )
    print(f"surrogate fixed cost: {setup:.3f} s (n_train={args.train_subsample})")
    if crossover is not None:
        print(f"crossover at N ~ {crossover:,.0f} alternatives")
    biggest = exact_rows[-1]
    match = [r for r in surrogate_rows if r["n"] == biggest["n"]]
    if match:
        print(
            f"at N={biggest['n']}: exact {biggest['seconds']:.3f} s vs "
            f"surrogate {match[0]['seconds']:.3f} s "
            f"({biggest['seconds'] / match[0]['seconds']:.1f}x)"
        )

    # Memory is the harder limit than time: the exact outranking materialises
    # an N-by-N preference matrix, so it stops being possible before it stops
    # being fast.
    largest = surrogate_rows[-1]["n"]
    exact_bytes = 8.0 * largest**2
    print(
        f"at N={largest:,}: the exact preference matrix alone would need "
        f"{exact_bytes / 2**30:,.1f} GiB; the surrogate needs O(N)."
    )


if __name__ == "__main__":
    main()

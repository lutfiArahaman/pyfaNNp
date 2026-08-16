---
title: 'pyFAP: A Python package for coupled fuzzy AHP, neural network and PROMETHEE decision analysis'
tags:
  - Python
  - multi-criteria decision making
  - fuzzy AHP
  - PROMETHEE
  - artificial neural networks
  - outranking
  - sensitivity analysis
authors:
  - name: <FIRST AUTHOR>
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
  - name: <SECOND AUTHOR>
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: <Department, Institution, City, Country>
    index: 1
  - name: <Department, Institution, City, Country>
    index: 2
date: <DD Month YYYY>
bibliography: paper.bib
---

<!--
DRAFTING NOTES (delete before submission)

Target length: JOSS asks for 250-1000 words of body text. This skeleton runs
long on purpose so you can cut rather than pad. If you submit to SoftwareX
instead, keep the same prose and re-label the sections:
  Summary            -> Motivation and significance (merge with Statement of Need)
  Software Description -> Software description
  Example            -> Illustrative examples
  add                -> Impact, Conclusions

PACKAGE NAME: "pyFAP" (Fuzzy AHP - ANN - PROMETHEE) is a placeholder.
Search PyPI before committing to it.

ANN COUPLING: this draft assumes coupling (c) - the network is trained as a
surrogate for the PROMETHEE net flow, so that large alternative sets can be
ranked without rerunning the full pairwise outranking. If you instead use the
network to impute missing criterion values (a) or to learn a non-linear
correction to the linear weighting assumption (b), the passages marked
[COUPLING] must be rewritten. Do not leave this vague; an unmotivated hybrid
pipeline is the most common reason this class of paper is rejected.
-->

# Summary

Multi-criteria decision analysis is routinely used to rank alternatives in
site selection, supplier evaluation, risk prioritisation and infrastructure
assessment. Three families of methods dominate this practice, and each is
incomplete on its own. The fuzzy analytic hierarchy process (FAHP) derives
criterion weights from expert pairwise judgements while tolerating the
linguistic vagueness of those judgements, but it yields a single static,
linear weight vector [@Chang1996; @Buckley1985]. Artificial neural networks
capture non-linear relationships between criteria and observed outcomes, but
they are opaque and produce no defensible preference ordering. PROMETHEE
provides a non-compensatory outranking that respects decision-maker
preference functions, but it depends entirely on weights supplied from
outside the method [@Brans1985; @Brans1986]. In applied work the three are
therefore used together — and, almost always, in three disconnected tools,
with results transcribed by hand between them.

pyFAP (Python for Fuzzy AHP-ANN-PROMETHEE) is an open-source Python package
that expresses this three-stage analysis as a single scripted pipeline. It
derives crisp weights and a consistency ratio from triangular fuzzy pairwise
comparison matrices, trains a neural surrogate on the resulting decision
problem, and produces PROMETHEE I and II rankings, with the handoffs between
stages defined by an explicit data contract rather than by manual
transcription. Every stage is addressable on its own, so the package can also
be used as three independent components.

With pyFAP, users can specify the fuzzy aggregation method, the preference
function and its indifference and preference thresholds, and the surrogate
architecture, and obtain the net flows, the full outranking relation and a
stability report as ordinary Python objects. The package builds on the
scientific Python ecosystem through NumPy [@Harris2020] and scikit-learn
[@Pedregosa2011], with optional pandas [@McKinney2010] output, and is
intended for researchers
and practitioners who need decision analyses that are reproducible,
version-controlled and testable rather than assembled once in a spreadsheet.

<!-- ~300 words. Cut the second half of paragraph one first if you need room. -->

# Statement of Need

The need for coupled decision-analytic tooling arises from the way hybrid
multi-criteria studies are actually carried out. A typical workflow computes
fuzzy weights in a spreadsheet, trains a predictive model in a separate
Python or MATLAB script, and performs the outranking in dedicated desktop
software. Each transition is a manual copy. The consequence is that the
analysis cannot be re-executed from a single command, intermediate results
are not retained, and the sensitivity of the final ranking to the expert
judgements that produced the weights is rarely examined — even though
rank reversal under small weight perturbations is a well-documented property
of outranking methods [@ADD_RANK_REVERSAL_REF].

pyFAP addresses this gap by providing a tool in which the three stages share
one data structure and one call. Because the pipeline is a single object,
the entire analysis can be re-run under perturbed judgements, which makes
Monte Carlo weight sensitivity a default output rather than a separate
undertaking. [COUPLING] Because the network is trained against the outranking
it approximates, alternative sets far larger than those tractable for
pairwise outranking can be scored at a fraction of the cost. This makes
pyFAP suitable for studies in which the number of alternatives, the number of
expert panels, or the need for auditable re-execution places the analysis
beyond what a manual workflow can support.

# Related Work

Several Python packages implement multi-criteria decision methods, but their
focus is on providing methods as independent routines rather than on
composing them. `pymcdm` and `scikit-criteria` offer broad collections of
normalisations, weighting schemes and ranking methods, and `pyDecision`
implements a wide range of AHP and PROMETHEE variants including fuzzy
formulations [@ADD_PYMCDM; @ADD_SCIKIT_CRITERIA; @ADD_PYDECISION]. These are
mature and well-tested libraries, and pyFAP does not attempt to replace them.
They do, however, leave the analyst to wire the stages together: there is no
shared representation carried from weight derivation through to outranking,
no propagation of the uncertainty expressed in the fuzzy judgements into the
final ranking, and no machine-learning component.

<!--
VERIFY THIS PARAGRAPH BEFORE SUBMITTING. Install pymcdm, scikit-criteria and
pyDecision and confirm exactly what each provides. If one of them already
chains fuzzy AHP to PROMETHEE, this paragraph must concede it explicitly and
the contribution must rest on the surrogate and the pipeline abstraction.
A reviewer will run this check.
-->

Fuzzy logic libraries such as `scikit-fuzzy` and `simpful` provide general
machinery for membership functions and fuzzy inference
[@ADD_SKFUZZY; @ADD_SIMPFUL]. They are not decision-analytic tools: they
offer no weight derivation from pairwise comparisons and no consistency
diagnostics, both of which are required before fuzzy judgements can be used
as criterion weights.

General-purpose machine learning libraries, including scikit-learn
[@Pedregosa2011] and PyTorch [@Paszke2019], will fit any model to a
criteria-by-alternatives matrix. They have, however, no notion of criteria,
preference functions or outranking relations, and cannot consume expert
judgement in its elicited form. A model fitted directly to a decision matrix
returns a prediction, not a justifiable ordering.

By contrast, pyFAP treats the composition itself as the object of interest.
It carries fuzzy judgements, derived weights, the decision matrix and the
outranking flows in one structure, exposes the pipeline as a single
fittable estimator, and makes the sensitivity of the ranking to its inputs a
first-class output. Thus pyFAP fills a gap between libraries that implement
decision methods individually and libraries that fit predictive models
without decision semantics.

# Software Description

pyFAP is organised as five modules and one pipeline object.

`fahp` constructs triangular fuzzy pairwise comparison matrices from
linguistic scales, derives weights by extent analysis [@Chang1996] or by the
geometric mean method [@Buckley1985], and reports the consistency ratio of
the defuzzified matrix.

`ann` wraps a feed-forward network behind a small interface with a
scikit-learn compatible backend and an optional PyTorch backend, exposing
only the arguments relevant to the decision context.

`promethee` implements the six standard preference functions and returns
positive, negative and net flows together with the PROMETHEE I partial and
PROMETHEE II complete orderings [@Brans1985].

`sensitivity` perturbs the derived weights under a specified distribution and
reports the resulting rank distribution per alternative.

`preprocessing` provides criterion normalisation, since PROMETHEE thresholds
are stated in each criterion's own units and a threshold shared across
criteria presupposes a shared scale.

`DecisionPipeline` composes the above. A complete analysis takes the
following form:

```python
from pyfap import FAHP, ANNSurrogate, Promethee, DecisionPipeline

pipe = DecisionPipeline(
    weights=FAHP(method="extent_analysis", consistency_check=True),
    surrogate=ANNSurrogate(hidden=(32, 16), epochs=300, random_state=0),
    ranker=Promethee(version="II", preference="v-shape", q=0.1, p=0.5),
)

result = pipe.fit(judgments=J, decision_matrix=X).rank()

result.weights            # crisp criterion weights
result.consistency_ratio  # CR of the defuzzified judgement matrix
result.net_flow           # PROMETHEE II net flow per alternative
result.ranking            # complete ordering
result.stability(n=1000)  # rank distribution under perturbed weights
```

<!--
Add a second, shorter listing showing the components used independently -
this is what demonstrates the package is a library and not one script.
Add Figure 2: the architecture diagram (boxes and arrows, judgements ->
weights -> surrogate -> flows -> stability). Keep it schematic; it carries
the "the pipeline is the contribution" argument better than prose can.
-->

# Example

Figure 1 illustrates what the pipeline makes visible that a manual workflow
does not. It is produced by `examples/figure1.py`, which CI re-runs on every
change.

<!--
DATASET: the figure currently uses pyfap.datasets.load_demo(), which is
SYNTHETIC data invented for the scaffold. The numbers quoted below are real
outputs of that data, recorded so the argument's shape is fixed, but they
must be regenerated on a published benchmark before submission and the
sentence below must name and cite it. Do not present the synthetic problem
as a benchmark.
-->

The top panel reports the PROMETHEE II net flows for six alternatives under
the two weight-derivation methods the package offers. The methods disagree
on the leading alternative: extent analysis ranks S1 first, the geometric
mean ranks S4 first, and the two orderings agree on only the last place.
The mechanism is visible in the weights. Extent analysis assigns the fourth
criterion a weight of exactly zero — a documented property of the method
rather than a statement that the criterion is irrelevant — so the ranking it
produces is effectively a three-criterion ranking, while the geometric mean
retains all four.

The bottom panels plot the distribution of each alternative's rank across
1,000 Monte Carlo perturbations of the derived weights, drawn at a relative
standard deviation of 10%, one panel per method. The complete ordering
changes in 71% of runs under extent analysis and 91% under the geometric
mean. Some positions are robust — one alternative holds last place in 94% of
runs — but the leading position is not: under the geometric mean it is taken
by three different alternatives across the simulations, none of them in a
majority. A workflow that reports a single ordering would present the
point estimate with no indication that it is this fragile, and the
perturbation study that reveals it is only affordable because the analysis
is one scripted object.

![PROMETHEE II net flows per alternative under Chang's extent analysis and
Buckley's geometric mean (top), and the distribution of each alternative's
rank across 1,000 Monte Carlo perturbations of the derived weights, one
panel per method (bottom). The two methods disagree on the leading
alternative, and neither ordering survives perturbation of the weights
within the imprecision the fuzzy judgements already
express.\label{fig:example}](figures/figure1.png)

Correctness of the individual components is verified against the worked
examples published in the original method papers; the test suite reproduces
the weights of <SOURCE> and the net flows of <SOURCE> to <TOLERANCE>.

For applications of coupled fuzzy multi-criteria and machine learning
analysis to substantive problems, see <ADD 2-3 APPLICATION CITATIONS>.

# Acknowledgements

The methods implemented in pyFAP follow the formulations of
<Chang / Buckley for FAHP> and <Brans and Vincke for PROMETHEE>, with
modifications including <state them: vectorised flow computation, consistency
handling for incomplete matrices, and so on>. We thank the authors for their
contributions to the field.

<FUNDING STATEMENT, GRANT NUMBER, AND ANY REQUIRED DISCLAIMER>

# References

<!--
Bibliography keys used above that you must add to paper.bib:

  Chang1996            Chang, D.-Y. (1996) Applications of the extent analysis
                       method on fuzzy AHP. EJOR 95(3), 649-655.
  Buckley1985          Buckley, J.J. (1985) Fuzzy hierarchical analysis.
                       Fuzzy Sets and Systems 17(3), 233-247.
  Brans1985            Brans, J.P. & Vincke, P. (1985) A preference ranking
                       organisation method. Management Science 31(6), 647-656.
  Brans1986            Brans, J.P., Vincke, P. & Mareschal, B. (1986) How to
                       select and how to rank projects: the PROMETHEE method.
                       EJOR 24(2), 228-238.
  Harris2020           NumPy. Nature 585, 357-362.
  Pedregosa2011        scikit-learn. JMLR 12, 2825-2830.
  McKinney2010         pandas. Proc. SciPy 2010.
  Paszke2019           PyTorch. NeurIPS 2019.
  ADD_PYMCDM           pymcdm - locate the SoftwareX paper.
  ADD_SCIKIT_CRITERIA  scikit-criteria - locate the citation.
  ADD_PYDECISION       pyDecision - locate the citation.
  ADD_SKFUZZY          scikit-fuzzy.
  ADD_SIMPFUL          simpful.
  ADD_RANK_REVERSAL_REF  a rank-reversal reference for outranking methods.

Verify every DOI. JOSS checks them.
-->

# Magnetic Fields from Measurements

[Read the draft](arxiv-draft.pdf) · [LaTeX source](arxiv-draft.tex)

This draft follows the route from Maxwell's field equations to a Gaussian map learned from robot readings, a textbook dipole field fitted to that map, and an exact identity linking the assumed sources to the measured readings. A controlled wire experiment asks how much of the field's mathematical form can be recovered from data within a library of candidate expressions.

The [3D trajectory](trial4_trajectory.pdf) uses all 501 aligned samples from 85 to 95 seconds of Aalto Invensense trial 4. The prediction curves evaluate the fitted models from the [magnetic reconstruction experiment](../../experiments/magnetic_mapping/dipole_field/RESULTS.md) at those positions. The second figure is a copy of the [measurement influence map](../04-measurement-influence/influence_map.pdf).

To reproduce the direct dipole weight in the same example, use the query, source grid, ridge penalty, and selected reading from the [influence calculation](../04-measurement-influence/build_assets.py). Pass the identity matrix as the smoother to `dipole_weights` and take the selected reading's vertical-to-vertical entry. This gives −0.02496049 in the direct fit. Using the fitted Gaussian smoother gives −0.08492318 in the two-stage fit.

From this directory, run the following command twice.

~~~powershell
pdflatex -interaction=nonstopmode -halt-on-error arxiv-draft.tex
~~~

The submission source consists of [arxiv-draft.tex](arxiv-draft.tex), [trial4_trajectory.pdf](trial4_trajectory.pdf), and [influence_map.pdf](influence_map.pdf).

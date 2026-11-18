# The magnetic field as a learned function

[Read the paper](measurement_influence.pdf) · [LaTeX source](measurement_influence.tex) · [Influence maps](influence_map.pdf)

This companion to [the equivalent-dipole experiment](../02-kernel-to-magnetostatics/README.md) derives the matrix weight of each measured field vector at a query. The two maps compare the vertical-to-vertical weights at the retained Invensense trial-4 position inside the spatial gap nearest its center. The selection uses only coordinates.

From this directory, run the following commands with the repository dependencies and pdfLaTeX installed.

~~~powershell
python build_assets.py
pdflatex -halt-on-error measurement_influence.tex
pdflatex -halt-on-error measurement_influence.tex
~~~

The [builder](build_assets.py) uses the archived geometry, readings and settings, verifies both weighted sums against the saved predictions, checks constant-vector reproduction, and refits after a one-unit perturbation to the vertical reading with the largest difference between the two vertical weights. The [manifest](asset_manifest.json) records the hashes of the numerical inputs and source files.

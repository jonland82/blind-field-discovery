# Maxwell's Perspective

[Read the four-page PDF](maxwells_perspective.pdf) · [LaTeX source](maxwells_perspective.tex).

**Maxwell's Perspective: Observations, learned kernels and the analytic functions we infer**, J. R. Landers, November 2026.

An expository companion to [note 1](../01-kernel-to-equation/kernel_to_equation.pdf) and [note 2](../02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf), with the same fonts, margins, author treatment and numbered narrative style. It connects all three experiments, places the theoretical and recovered wire fields side by side, and asks when different observational and mathematical processes could converge on the same function.

The note distinguishes a physical system, an observation operator and the conditional prediction target. A missing-current example illustrates how changing the supplied variables changes the inferable function. It treats agreement on tested observations, common mathematical structure and identification of a unique field as separate claims. Historical context is supported by Maxwell's primary writings; the wire simulation is a controlled analogy to an inference problem.

The [experiment](../../experiments/hidden_wire/README.md), [fixed protocol](../../experiments/hidden_wire/PROTOCOL.md) and [all results](../../experiments/hidden_wire/results/RESULTS.md) contain the complete controls and sensor-stage comparison.

From the repository root:

```powershell
python scripts/build_papers.py
```

To build this note alone:

```powershell
python papers/03-maxwells-perspective/build_assets.py
cd papers/03-maxwells-perspective
pdflatex -interaction=nonstopmode -halt-on-error maxwells_perspective.tex
pdflatex -interaction=nonstopmode -halt-on-error maxwells_perspective.tex
```

The builder verifies experiment implementation/protocol/input hashes, reproduces every saved analytic prediction and its test score, and generates the displayed coefficients, numerical results and figure from retained arrays. It records its inputs in `asset_manifest.json`. No raw-data download or new fitting is needed. Keep `values.tex`, `wire_table.tex` and `perspective.pdf` beside the main source. The [figure PNG](perspective.png) is also available for inspection.

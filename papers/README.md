# Research notes

**J. R. Landers · November 2026**

1. **From samples to an equation: A magnetic-like kernel benchmark.** [PDF](01-kernel-to-equation/kernel_to_equation.pdf) · [Source and build instructions](01-kernel-to-equation/README.md). A controlled synthetic demonstration from samples to a kernel, a differential equation and the original analytic function family.
2. **From a learned kernel to a magnetic field: Real-data reconstruction with equivalent dipoles.** [PDF](02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf) · [Source and build instructions](02-kernel-to-magnetostatics/README.md). A standalone follow-up that projects a real-data vector kernel onto a magnetostatic source representation and evaluates measured predictions.
3. **Maxwell's Perspective: Observations, learned kernels and the analytic functions we infer.** [PDF](03-maxwells-perspective/maxwells_perspective.pdf) · [Source and build instructions](03-maxwells-perspective/README.md). An expository companion connecting both notes with hidden-wire discovery, side-by-side field equations, and the question of whether different observational processes converge on a common function.

4. **The magnetic field as a learned function** [PDF](04-measurement-influence/measurement_influence.pdf) · [Source and build instructions](04-measurement-influence/README.md). A classical dipole superposition becomes an exact sum over training measurements, connecting physical field representations with kernel learning.

**Unified arXiv draft.** [Magnetic Fields from Measurements](arxiv-draft-maxwell/arxiv-draft.pdf) · [Source and build instructions](arxiv-draft-maxwell/README.md). The paper follows a path from Maxwell's field equations through the measured room to a controlled recovery of the field around a wire.

The first two papers are three pages each. The third and fourth are four pages each. From the repository root, run `python scripts/build_papers.py` to rebuild all figures, tables and PDFs from retained experiment outputs.

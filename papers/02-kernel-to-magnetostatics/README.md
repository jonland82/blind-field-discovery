# Follow-up LaTeX note

[Read the three-page PDF](kernel_to_magnetostatics.pdf) · [LaTeX source](kernel_to_magnetostatics.tex).

**From a learned kernel to a magnetic field: Real-data reconstruction with equivalent dipoles**, J. R. Landers, November 2026.

This standalone follow-up to [From samples to an equation](../01-kernel-to-equation/kernel_to_equation.pdf) uses the same document class, margins, fonts, author treatment and numbered narrative style. It derives the magnetostatic dipole response and regularized projection from the learned vector field, reports real-data validation and held-out results, and distinguishes the supplied physics from the inferred equivalent sources. The figure and tables are built from the saved experiment rather than manually transcribed.

From the workspace root:

```powershell
python papers/02-kernel-to-magnetostatics/build_assets.py
cd papers/02-kernel-to-magnetostatics
pdflatex -interaction=nonstopmode -halt-on-error kernel_to_magnetostatics.tex
pdflatex -interaction=nonstopmode -halt-on-error kernel_to_magnetostatics.tex
```

The builder checks implementation hashes and saved prediction errors, and verifies the pinned raw archive when it is cached. The retained result arrays suffice to rebuild the paper without downloading raw data. Keep `field_reconstruction.pdf`, `stable_table.tex`, `stress_table.tex` and `values.tex` beside the main source. `asset_manifest.json` records input hashes. Requirements: Python with NumPy, SciPy, Matplotlib and threadpoolctl; LaTeX with the same packages as the earlier note.

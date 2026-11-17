# Three-page LaTeX note

Follow-up: [From a learned kernel to a magnetic field](../02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf), a standalone three-page note in the same style, with the real-data magnetostatic derivation and measured results. [Source and build instructions](../02-kernel-to-magnetostatics/README.md).

[Read the PDF](kernel_to_equation.pdf) · [LaTeX source](kernel_to_equation.tex).

The note poses the starting question, derives the kernel/differential/integral representations, and completes the symbolic-recovery step: discover a field-only polynomial differential equation, solve it, and recover the original tanh/sech-squared family. It reports fitted parameters, prediction errors, assumptions, and extrapolation results. Both vector figures and numerical tables are built directly from saved benchmark outputs.

From the workspace root, build the assets:

```powershell
python papers/01-kernel-to-equation/build_assets.py
```

Then compile twice from this directory:

```powershell
cd papers/01-kernel-to-equation
pdflatex -interaction=nonstopmode -halt-on-error kernel_to_equation.tex
pdflatex -interaction=nonstopmode -halt-on-error kernel_to_equation.tex
```

Keep `reconstruction.pdf`, `symbolic_recovery.pdf`, `results_table.tex`, `symbolic_table.tex`, and `values.tex` alongside the main TeX source. `build_assets.py` requires NumPy and Matplotlib and checks that the saved benchmark matches its implementation hash. Compilation uses a standard LaTeX installation with AMS math, graphicx, booktabs, geometry, Latin Modern and hyperref.

# Equivalent Descriptions of Light

**Jonathan R. Landers**

[Read the paper](equivalent_descriptions.pdf) · [LaTeX source](equivalent_descriptions.tex)

This four-page mathematical companion extends the research arc from learning and tracing a fitted field to understanding equivalent descriptions of a physical theory. It introduces notation as it derives changes of variables, the complex Maxwell field with sources, vacuum wave equations, plane-wave constraints, and Fourier representations. Square-loop and continuous-forest constructions encode a chosen waveform. An ideal circuit chain illustrates how local dynamics can instead yield a wave speed and amplitude ratio, with a finite-spacing dispersion relation showing the limits of the continuum description.

The paper uses 11-point type, one-inch margins, and natural page flow. It has no generated figures or external data dependencies. From this directory, run the following command twice with a LaTeX installation providing `pdflatex`.

```powershell
pdflatex -interaction=nonstopmode -halt-on-error equivalent_descriptions.tex
```

## Decision stumps as exact Maxwell fronts

[Read the expanded companion](stumps_as_maxwell_fronts.pdf) · [LaTeX source](stumps_as_maxwell_fronts.tex) · [Original working note](stumps_as_maxwell_fronts.md)

An exact physical representation need not be the best way to learn a field. The paper introduces that takeaway first, then develops section 5's threshold ensemble into a representation by exact weak vacuum solutions. It derives the jump conditions, bounded-variation representation, and uniform profile-sampling bounds before reporting two exploratory experiments. Spatial forests and boosted stumps trail the existing Gaussian and dipole models on measured magnetic maps. On a simulated noisy traveling pulse, phase-based stumps attain 6.28% relative error, while a forest using the same physical phase attains 3.75%; both propagate their profile error unchanged. The discussion explains why this negative prediction finding leaves the equivalence result intact, and why the better phase forest is itself expressible as a stump sum.

The expanded paper uses 11-point type, one-inch margins, and natural page flow, with two vector figures and two result tables generated from retained experiment outputs. The [asset builder](build_stump_assets.py) verifies recorded source and input hashes and requires NumPy and Matplotlib; it does not refit models or download data. From the repository root:

```powershell
python papers/05-equivalent-descriptions/build_stump_assets.py
```

The figures compare [sensor prediction errors](stump_sensor_comparison.pdf) and [traveling-wave predictions and propagation errors](stump_wave_comparison.pdf). [Experiment protocol and reproduction](../../experiments/magnetic_mapping/README.md#tree-and-traveling-wave-comparison). [Asset input hashes](asset_manifest.json).

Compile twice from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error stumps_as_maxwell_fronts.tex
```

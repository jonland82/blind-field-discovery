# From learned kernels to magnetic-field equations

**[Live site →](https://jonland82.github.io/blind-field-discovery/)**

Research notes by **Jonathan R. Landers** on the path from sampled observations to learned kernels, explicit equations and equivalent mathematical descriptions of fields.

The five notes follow observations into fitted equations, trace those equations back to the individual readings that shape their predictions, and examine what survives a change of mathematical representation. The arc moves from learning a field to comparing descriptions of that field, with Maxwell’s equations and light as a worked example. Visit [the live research site](https://jonland82.github.io/blind-field-discovery/) or open [index.html](index.html) locally. GitHub Pages serves the buildless site from the root of `master`. [Design and asset notes](assets/site/README.md).

## The papers

**Unified paper.** *Magnetic Fields from Measurements* starts with Maxwell and follows 500 robot readings into a Gaussian map and a fitted dipole field. It shows how that same field is exactly a weighted sum of the readings, then asks whether a learner can choose the equation for a simulated wire. [Draft PDF](papers/arxiv-draft-maxwell/arxiv-draft.pdf) · [LaTeX and build instructions](papers/arxiv-draft-maxwell/README.md) · [The paper in the research story](https://jonland82.github.io/blind-field-discovery/#unified-paper)

| | Paper | Read | Source and reproduction |
| --- | --- | --- | --- |
| **1** | **From samples to an equation** — A magnetic-like kernel benchmark | [PDF](papers/01-kernel-to-equation/kernel_to_equation.pdf) | [LaTeX and assets](papers/01-kernel-to-equation/README.md) · [Experiment](experiments/kernel_benchmark/README.md) |
| **2** | **From a learned kernel to a magnetic field** — Real-data reconstruction with equivalent dipoles | [PDF](papers/02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf) | [LaTeX and assets](papers/02-kernel-to-magnetostatics/README.md) · [Theory and experiment](experiments/magnetic_mapping/dipole_field/THEORY.md) |
| **3** | **Maxwell's Perspective** — Observations, learned kernels and the analytic functions we infer | [PDF](papers/03-maxwells-perspective/maxwells_perspective.pdf) | [LaTeX and assets](papers/03-maxwells-perspective/README.md) · [Hidden-wire experiment](experiments/hidden_wire/README.md) |
| **4** | **The magnetic field as a learned function** | [PDF](papers/04-measurement-influence/measurement_influence.pdf) | [Derivation, influence maps and reproduction](papers/04-measurement-influence/README.md) |
| **5** | **Equivalent Descriptions of Light** — Mathematical representation, Maxwell’s equations, and physical interpretation | [PDF](papers/05-equivalent-descriptions/equivalent_descriptions.pdf) | [LaTeX and build instructions](papers/05-equivalent-descriptions/README.md) |

The first note verifies the workflow on a hidden synthetic function: samples → Gaussian kernel → differential equation → recovered tanh/sech-squared functions. The second uses public measured magnetic data: a learned vector kernel → equivalent magnetic dipoles → held-out measurement predictions. The dipole reconstruction reduces reference-instrument vector RMSE from 8.09 to 5.22 in this exploratory study. The physical family is supplied; the equivalent moments are inferred. It does not uniquely identify the room's actual sources or independently discover Maxwell's equations.

The third note connects the experiments through "Maxwell's Perspective": how observations and mathematical choices shape an inferred analytic function, and when different discovery processes could converge. It compares a hidden wire's theoretical field with the recovered expression, including noise, coverage and direct-fit controls.

The fourth note returns to the measured field from note II. With the fitting settings fixed, the reconstructed field is exactly both a sum of fitted classical dipole responses and a matrix-weighted sum of the original robot readings. Fitting equivalent sources to magnetic data is an established method. This note derives the per-reading weights for the Gaussian fit and its dipole reconstruction and compares them on the same data. At one held-out query, the reading with the largest difference in vertical weights raises the Gaussian prediction by 0.579 units and lowers the dipole prediction by 0.085 units when increased by one unit. The paper verifies that reversal by refitting both models. [Influence maps](papers/04-measurement-influence/influence_map.pdf) · [Emilia’s 1973 equivalent-source study](https://pubs.geoscienceworld.org/seg/geophysics/article-abstract/38/2/339/71420/Equivalent-Sources-Used-As-An-Analytic-Base-For)

The fifth note develops mathematical equivalence from an invertible change of variables, derives a complex-field reformulation of Maxwell’s equations with sources, and obtains light’s propagation speed and polarization in vacuum. Time and frequency descriptions, square-loop clocks, and continuous threshold forests illustrate how the same waveform can take different mathematical forms. An ideal circuit chain then derives wave behavior from local laws. The paper distinguishes exact reformulation of a theory from encoding one solution or deriving an approximation in a continuum limit. [Read the equivalence chapter on the site](https://jonland82.github.io/blind-field-discovery/#equivalent-descriptions).

## Repository map

```text
papers/
  01-kernel-to-equation/       First paper: PDF, LaTeX, figures, tables, builder
  02-kernel-to-magnetostatics/ Follow-up: PDF, LaTeX, figures, tables, builder
  03-maxwells-perspective/    Expository synthesis, wire equations and convergence
  04-measurement-influence/   Measurement weights for the Gaussian and dipole fields
  05-equivalent-descriptions/ Maxwell reformulations, light waves and model equivalence
  arxiv-draft-maxwell/       Unified paper, LaTeX source and figures
experiments/
  kernel_benchmark/           Controlled synthetic workflow behind paper 1
  hidden_wire/               Hidden wire: kernel to discovered analytic field
  magnetic_mapping/          Aalto/FIU studies and dipole reconstruction for paper 2
  radio_response/             Supporting measured radio-response investigation
  maxwell_control/            Earlier synthetic equation discovery and pose diagnostic
docs/                        Broader mathematics, provenance and project history
data/                        Local raw-data caches and tracked provenance manifests
scripts/                     Paper builds and repository checks
```

## Rebuild the papers

Python 3.10+ and a LaTeX installation with `pdflatex` are required. The checked-in result arrays are sufficient; no raw-data download is needed to build the papers.

```powershell
python -m pip install -r requirements.txt
python scripts/build_papers.py
```

This rebuilds the first four notes’ figures and tables from saved results, verifies their implementation hashes, and compiles each PDF twice. The first two papers are three pages each. The third, fourth, and fifth are four pages each. The PDFs and source assets are included in the repository for direct reading.

The fifth note is a standalone mathematical treatment with no generated assets. Build it from its directory with `pdflatex -interaction=nonstopmode -halt-on-error equivalent_descriptions.tex`, twice. The unified draft has its own [build instructions](papers/arxiv-draft-maxwell/README.md).

## Reproduce the central experiments

```powershell
python experiments/kernel_benchmark/run_demo.py
python experiments/magnetic_mapping/fetch_data.py
python experiments/magnetic_mapping/run_experiment.py
python experiments/magnetic_mapping/run_dipole_field.py
python experiments/hidden_wire/run_experiment.py
python scripts/build_papers.py
```

The magnetic fetcher downloads the pinned public Aalto archive into `data/aalto_magnetic/`. The analyzer trains on Invensense trials 1–2, selects on trial 3, and evaluates stable trials 4–5 plus changed-environment trials 6–9. Other instruments retain the trial-1-only affine alignment. These acquisitions were already examined during project development; results are exploratory.

## Check and explore

```powershell
python -m pytest experiments -q
python scripts/check_repository.py
```

The tests cover kernel calculus, analytic solutions, data leakage, magnetic identities and saved prediction scores. The optional radio artifact check skips when its unversioned 88 MB prediction archive has not been regenerated.

- [Magnetic experiment overview](experiments/magnetic_mapping/README.md), including the independent FIU study and failed scalar equation candidates.
- [Hidden-wire discovery and matched sensor comparison](experiments/hidden_wire/README.md): general algebra recovers the wire-field shape; includes noise, coverage and direct-fit controls.
- [Radio-response theory and measurements](experiments/radio_response/README.md).
- [Earlier Maxwell control](experiments/maxwell_control/README.md) and [mathematical walkthrough](docs/MATHEMATICS.md).
- [Dataset provenance and attribution](data/README.md), [repository reorganization](docs/REORGANIZATION.md), and [rights and third-party material](NOTICE.md).

The active work uses public recordings. The older hardware collection protocol is retained as an inactive design reference.

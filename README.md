# From learned kernels to magnetic-field equations

Research notes by **J. R. Landers** on the path from sampled observations to learned kernels, explicit equations and theoretical field representations.

## The papers

| | Paper | Read | Source and reproduction |
| --- | --- | --- | --- |
| **1** | **From samples to an equation** — A magnetic-like kernel benchmark | [PDF](papers/01-kernel-to-equation/kernel_to_equation.pdf) | [LaTeX and assets](papers/01-kernel-to-equation/README.md) · [Experiment](experiments/kernel_benchmark/README.md) |
| **2** | **From a learned kernel to a magnetic field** — Real-data reconstruction with equivalent dipoles | [PDF](papers/02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf) | [LaTeX and assets](papers/02-kernel-to-magnetostatics/README.md) · [Theory and experiment](experiments/magnetic_mapping/dipole_field/THEORY.md) |

The first note verifies the workflow on a hidden synthetic function: samples → Gaussian kernel → differential equation → recovered tanh/sech-squared functions. The second uses public measured magnetic data: a learned vector kernel → equivalent magnetic dipoles → held-out measurement predictions. The dipole reconstruction reduces reference-instrument vector RMSE from 8.09 to 5.22 in this exploratory study. The physical family is supplied; the equivalent moments are inferred. It does not uniquely identify the room's actual sources or independently discover Maxwell's equations.

## Repository map

```text
papers/
  01-kernel-to-equation/       First paper: PDF, LaTeX, figures, tables, builder
  02-kernel-to-magnetostatics/ Follow-up: PDF, LaTeX, figures, tables, builder
experiments/
  kernel_benchmark/           Controlled synthetic workflow behind paper 1
  magnetic_mapping/          Aalto/FIU studies and dipole reconstruction for paper 2
  radio_response/             Supporting measured radio-response investigation
  maxwell_control/            Earlier synthetic equation discovery and pose diagnostic
docs/                        Broader mathematics, provenance and project history
data/                        Local raw-data caches and tracked provenance manifests
scripts/                     Paper builds and repository checks
```

## Rebuild the papers

Python 3.10+ and a LaTeX installation with `pdflatex` are required. The checked-in result arrays are sufficient; no raw-data download is needed to build either paper.

```powershell
python -m pip install -r requirements.txt
python scripts/build_papers.py
```

This rebuilds figures and tables from saved results, verifies their implementation hashes, and compiles each three-page PDF twice. The PDFs and source assets are included in the repository for direct reading.

## Reproduce the central experiments

```powershell
python experiments/kernel_benchmark/run_demo.py
python experiments/magnetic_mapping/fetch_data.py
python experiments/magnetic_mapping/run_experiment.py
python experiments/magnetic_mapping/run_dipole_field.py
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
- [Radio-response theory and measurements](experiments/radio_response/README.md).
- [Earlier Maxwell control](experiments/maxwell_control/README.md) and [mathematical walkthrough](docs/MATHEMATICS.md).
- [Dataset provenance and attribution](data/README.md), [repository reorganization](docs/REORGANIZATION.md), and [rights and third-party material](NOTICE.md).

The active work uses public recordings. The older hardware collection protocol is retained as an inactive design reference.

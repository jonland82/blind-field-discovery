# Repository organization

The two papers are the primary entry points. This reorganization moves their complete sources, checked figures, tables and PDFs into `papers/`; groups supporting implementations in `experiments/`; and centralizes local public-data caches under `data/`.

| Previous location | Current location |
| --- | --- |
| `kernel_demo/note/` | `papers/01-kernel-to-equation/` |
| `real_experiment/dipole_field/note/` | `papers/02-kernel-to-magnetostatics/` |
| `kernel_demo/` | `experiments/kernel_benchmark/` |
| `real_experiment/` | `experiments/magnetic_mapping/` |
| `radio_experiment/` | `experiments/radio_response/` |
| `blind_discovery/` | `experiments/maxwell_control/` |
| `blind_discovery.py` | `experiments/maxwell_control/run_discovery.py` |
| `blind_discovery/MATHEMATICS.md` | `docs/MATHEMATICS.md` |
| Raw archives under the experiments | `data/aalto_magnetic/`, `data/fiu_marg/`, `data/dichasus/` |
| `advio-12/` | `data/advio-12/` |

Commands, relative links, default data locations, paper asset builders and reproducibility footers follow the new paths. Numerical results are regenerated with the relocated implementations so their recorded source hashes remain valid. Repository line-ending rules preserve those hashes across checkouts.

Validation after relocation: all 11 saved numerical archives remained byte-identical, and all eight scientific result reports retained the same content after excluding updated implementation/input hashes. The full suite passed 38 tests. Both papers rebuilt to three pages, and the local documentation links and recorded source hashes passed checks.

Raw downloads, build intermediates and Python caches remain local. The supporting radio study's 88 MB derived prediction archive is regenerable and excluded from Git; its reports, figures, provenance and small impulse-response examples remain tracked. Both papers' supporting numerical arrays are retained.

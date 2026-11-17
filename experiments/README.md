# Supporting experiments

The [two papers](../papers/README.md) are the main entry point. These directories retain their supporting code, saved results and earlier investigations.

| Directory | Role | Entry point |
| --- | --- | --- |
| [kernel_benchmark](kernel_benchmark/README.md) | Synthetic verification behind paper 1 | `python experiments/kernel_benchmark/run_demo.py` |
| [magnetic_mapping](magnetic_mapping/README.md) | Measured magnetic studies; dipole reconstruction behind paper 2 | [Dipole theory and experiment](magnetic_mapping/dipole_field/THEORY.md) |
| [radio_response](radio_response/README.md) | Supporting public radio-response experiment | `python experiments/radio_response/run_experiment.py` |
| [maxwell_control](maxwell_control/README.md) | Earlier synthetic equation discovery and ADVIO route-coverage diagnostic | `python experiments/maxwell_control/run_discovery.py` |

Each experiment keeps code next to its results and tests. Raw input archives are centralized under `data/` and ignored by Git. Results distinguish generated test fixtures, synthetic physical benchmarks and actual measured observations.

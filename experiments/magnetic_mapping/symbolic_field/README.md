# Real-data kernel-to-equation experiment

[Read the results, formalism, actual extracted functions and limitations](RESULTS.md) · [Plots of every region](comparison.png).

This is the measured-data follow-up to the [synthetic symbolic-recovery note](../../../papers/01-kernel-to-equation/kernel_to_equation.pdf). It uses the already cached public Aalto magnetic recordings and requires no new hardware or generated physical observations.

From the workspace root:

```powershell
python experiments/magnetic_mapping/run_symbolic_field.py
python -m pytest experiments/magnetic_mapping/tests/test_symbolic_field.py -q
```

The script verifies the pinned source hashes, fixes geometry using training positions, selects kernels and derivative equations using trials 1–3, then freezes all functions before evaluating later trials and other instruments. The real observable is field magnitude in a finite-width spatial band. Reusing previously scored acquisitions makes this exploratory rather than a fresh confirmatory test.

`results.json` records every candidate, selection failure, fitted equation and analytic parameters, source hashes, calibration, raw row selections, implementation hashes, and per-recording scores. `predictions.npz` retains measured values, selected original rows, kernel weights, and all predictions. Synthetic arrays in software tests are temporary fixtures and do not enter these results.

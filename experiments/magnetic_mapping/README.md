# From measurements to a transferable description

Latest experiment: [derive an equivalent magnetic dipole field from the learned vector kernel](dipole_field/THEORY.md). The reconstructed theoretical field follows the measured shapes and reduces stable source-test vector RMSE from 8.09 to 5.22. [Results and plot](dipole_field/RESULTS.md). This supplies magnetostatic structure and estimates equivalent sources; it does not identify the actual room sources or discover Maxwell independently.

This experiment asks whether a description learned from one instrument predicts other measurements of the same environment, and how it compares with an explicitly stated Maxwell-based description. Agreement with Maxwell is not the acceptance criterion; held-out predictive performance is.

**The active research path uses publicly available measurements. No phone purchases or new recordings are required.** There are two completed public-data studies:

- **Completed public-data pilot:** measured recordings from a Nexus phone and two robot-mounted magnetometers. The source instrument's first two trials train a field map. Trial 3 selects settings. Later trials and other instruments test transfer. [Results and limitations](results/RESULTS.md), [comparison figure](results/comparison.png).
- **Completed optical-pose study:** 30 public magnetic recordings with independent optical position and orientation. Models train on recordings 01–10, select settings on 11–15, and predict 16–30. It compares a generic pose-to-reading learner, a shared-vector field description, and a static-potential description. [Results](public_transfer/RESULTS.md), [comparison figure](public_transfer/comparison.png).

The previously prepared [two-phone collection protocol](COLLECTION_PROTOCOL.md) is retained as an optional design reference. It is inactive under the public-data-only constraint, and its empty collection is not a blocker for this project. No new physical measurements have been collected. See [the public-data research plan](PUBLIC_DATA_PLAN.md) for the active scope and remaining questions.

**Response-kernel follow-up:** [the completed radio experiment](../radio_response/README.md) now learns a relative antenna-response operator from measured DICHASUS data and compares it with geometric propagation. Its [theory and protocol](../radio_response/THEORY_AND_EXPERIMENT.md) connect the statistical similarity kernel to Maxwell's Green-function formulation; its [results](../radio_response/results/RESULTS.md) preserve the clock and environmental-transfer limits.

**Kernel-to-analytic-function follow-up:** [the real magnetic-profile experiment](symbolic_field/RESULTS.md) now tries the complete workflow from the [synthetic note](../../papers/01-kernel-to-equation/kernel_to_equation.pdf): learn a kernel, infer a derivative equation, solve it, and predict actual held-out measurements. It tests one narrow spatial band and all four geometry-defined local intervals. The kernel predicts well on the reference instrument; local analytic candidates retain much of that value-prediction performance, but **none passes the joint derivative/prediction acceptance criteria**. The global Riccati equation fails to represent the multi-peaked profile. [All-region plot](symbolic_field/comparison.png).

## What was observed in the pilot

The unrestricted kernel learner predicts held-out recordings substantially better than constant and shuffled-label controls. Its error is also lower than the planar-potential learner's on the two stable-room test trials for every instrument in this run. Errors inside the withheld spatial rectangle are mixed: transfer to new positions is not uniformly successful, particularly for the phone's fifth trial. The detailed report preserves all these results.

This supports useful spatial prediction and some transfer after calibration. It does not establish a unique representation of electromagnetism or independently recover Maxwell. The Maxwell-inspired comparator imposes one planar static relationship, not the full electromagnetic system. The source data is already calibrated and orientation-corrected, which supplies part of the representation.

In the independent optical-pose study, the generic learner reduced pooled test error by 56.6% relative to a constant prediction, the shared-vector model by 55.1%, and the static-potential model by 41.1%. The shared-vector model performed similarly to the unconstrained learner, although its pooled error was higher. One difficult recording strongly affects the pooled errors; the report also shows median and every per-recording score. This tests explicit candidate descriptions across recordings, not spontaneous discovery of vector variables or Maxwell's equations. The second dataset uses a MARG module, not a phone.

## Run the public-data pilot

From the workspace root:

```powershell
python -m pip install -r experiments/magnetic_mapping/requirements.txt
python experiments/magnetic_mapping/fetch_data.py
python experiments/magnetic_mapping/run_experiment.py
python experiments/magnetic_mapping/fetch_fiu.py
python experiments/magnetic_mapping/run_fiu.py
python experiments/magnetic_mapping/run_symbolic_field.py
python -m pytest experiments/magnetic_mapping/tests -q
```

The data download is pinned to commit `b45da3bc7ab623fdef6e01247021daebf7983f29` of [AaltoML/magnetic-data](https://github.com/AaltoML/magnetic-data). The downloader records SHA-256 hashes and retains the upstream README. The analyzer verifies those hashes and makes no network requests. Downloaded numeric files are unchanged; one incomplete row in Trivisio trial 8 is excluded during loading and listed in the quality-control report.

Dependencies are NumPy, SciPy, Matplotlib, and threadpoolctl; pytest runs the checks. Outputs are written to `experiments/magnetic_mapping/results` by default, separately from the original synthetic experiment. Use `--output` to select another folder. Sample caps and selected row indices are recorded in the JSON results.

The downstream report contains:

- Per-instrument, per-trial prediction errors, including a spatial region removed from training and validation.
- A linear spatial baseline, a constant baseline, and a label-shuffled control.
- A comparison of an unconstrained learned map with a planar scalar-potential map.
- A later changed-environment stress test with the original maps frozen.
- Explicit calibration coefficients and trial membership, provenance, predictions, and a summary figure.

## Tree and traveling-wave comparison

`run_tree_comparison.py` compares random forests and boosted depth-one trees with the saved Gaussian and dipole predictions using exactly the same training, validation, calibration, and test arrays. Settings are selected on trial 3 across three fixed seeds. It separately fits a controlled noisy traveling wave with generic trees, Maxwell-constrained stumps, and a random forest given the same traveling phase. The last control distinguishes knowing the propagation coordinate from choosing a particular estimator. These are exploratory comparisons; the sensor tests were examined in earlier studies.

```powershell
python -m pip install -r experiments/magnetic_mapping/tree_requirements.txt
python experiments/magnetic_mapping/run_tree_comparison.py
python -m pytest experiments/magnetic_mapping/tests/test_tree_comparison.py -q
```

Outputs: [comparison figure](tree_comparison/comparison.png), [sensor scores and all validation candidates](tree_comparison/sensor_results.json), [wave scores, fitted fronts, and jump checks](tree_comparison/wave_results.json), and [environment and input hashes](tree_comparison/manifest.json). Prediction arrays are retained alongside the reports. The wave experiment uses dimensionless light speed one and checks distributional jump coefficients rather than finite differences across discontinuities. It supplies the propagation law and does not discover Maxwell's equations from the room data.

## The comparison being made mathematically

The unrestricted learner models the three observed channels as separate smooth functions of supplied position:

$$
\widehat{\mathbf b}(x,y)=\overline{\mathbf b}
+\sum_i k_\ell((x,y),\mathbf x_i)\mathbf w_i,
\qquad
k_\ell(\mathbf x,\mathbf x')=\exp\left(-\frac{\|\mathbf x-\mathbf x'\|^2}{2\ell^2}\right).
$$

The second learner couples the horizontal components through a planar potential:

$$
(\widehat B_x,\widehat B_y)=-\nabla_{xy}\psi,
\qquad
\partial_x\widehat B_y-\partial_y\widehat B_x=0.
$$

For a scalar potential covariance $\ell^2 k_\ell$, the horizontal vector covariance is

$$
K_{ab}(\mathbf x,\mathbf x')
=\left(\delta_{ab}-\frac{(x_a-x'_a)(x_b-x'_b)}{\ell^2}\right)k_\ell(\mathbf x,\mathbf x'),
\qquad a,b\in\{x,y\}.
$$

Both maps use regularized kernel fitting, training-only channel means, and the same candidate length scales and ridge values. Validation uses only the source instrument's third trial, outside the spatial exclusion. Their effective model classes differ by design; equal grids do not guarantee equal statistical flexibility.

The potential assumption corresponds to the observed component of a current-free, quasistatic magnetic-curl condition in the sampled air region. It does not imply full $\nabla\cdot\mathbf B=0$ from planar data; $\partial_z B_z$ is unmeasured. The vertical channel is fitted independently. The interpretation assumes that the upstream alignment has placed the recorded field axes in a consistent frame relative to the position axes. An unknown frame mismatch is an alternative explanation for a poorer constrained fit.

The unrestricted map has no Maxwell penalty or derivative features. Its inputs are still supplied physical coordinates, and its smooth kernel is an inductive bias. This pilot is an operational comparison of descriptions, not discovery of a complete latent physical state.

Cross-instrument predictions use a fixed empirical transformation

$$
\mathbf b_{\rm reference}\approx\mathbf b_{\rm target}A+\mathbf c.
$$

It is fitted only to nearby paired observations in trial 1, outside the spatial exclusion, with at most 500 pairs. It can absorb scale, offset, and axis mixing. It cannot establish that the channels were independently discovered as vector components. The target instruments supply calibration data but no labels for training or selecting the spatial maps. Frozen calibration may fail under later drift; all such errors remain in the report.

## How this connects ML to theory

The first measurable bridge is transfer: a learned relation earns scope by predicting new observations. An explicit physical description is evaluated on the same observations. A prediction advantage for either model concerns the entire description, including source, sensor, and stationarity assumptions.

Establishing a stronger representation-level bridge would require learning variables from raw observations, showing that those variables transform consistently between experiments, and extracting relationships that continue to predict new interventions. Neither the original synthetic run nor this public pilot establishes that stronger result.

The [optical-pose study](public_transfer/RESULTS.md) tests how a field description explains measurements under independently observed changes in pose. The [controlled protocol](COLLECTION_PROTOCOL.md) remains an inactive design reference for independently changed source and observer configurations; executing it is outside the current public-data-only scope.

## Attribution

Solin, Kok, Wahlström, Schön, and Särkkä (2018), [Modeling and interpolation of the ambient magnetic field by Gaussian processes](https://doi.org/10.1109/TRO.2018.2830326), with [the authors' data documentation](https://github.com/AaltoML/magnetic-data). The dataset authors previously studied Maxwell-informed magnetic mapping; this pilot builds a separate, small comparison rather than claiming that comparison is novel.

The pinned data repository has no explicit licence file. Its README requests citation. The local download and derived artifacts retain provenance; no new redistribution licence is asserted. The original ADVIO data has separate terms documented in [the original README](../maxwell_control/README.md).

The optical-pose study uses [FIUMARGDB](https://github.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations), published with Sonchan and colleagues (2023), [Benchmarking Dataset of Signals from a Commercial MEMS MARG Sensor](https://doi.org/10.3390/s23083786). The pinned repository's LICENSE is retained with its download. See that study's report for provenance and numerical results.

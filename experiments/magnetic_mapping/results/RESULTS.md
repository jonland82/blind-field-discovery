# Real-data pilot: learning across magnetometers

**This run uses measured magnetic recordings, not generated fields.** It compares prediction across later trials and instruments. It does not discover the full Maxwell system.

The public recording has one Nexus phone and two robot-mounted magnetometers. It is a proxy for the proposed two-phone experiment, not a new two-phone collection.

## Observed result

On the held-out phone trials, pooled vector RMSE is **10.3424** for the unrestricted learner, **16.3157** for the constant baseline, **16.1673** for linear regression, and **11.5569** for the potential-constrained map. The unrestricted learner reduces error by **36.6%** relative to the constant baseline. All numbers use the reference recording's numerical units after frozen device alignment.

In this run, the unrestricted map has lower error than the potential map on both stable-room test trials for every instrument. Spatial-hole performance is mixed: for phone trial 5, neither learned map improves on the constant baseline inside the excluded rectangle. The evidence supports useful predictive structure, with limits to spatial and device transfer. The imposed static constraint does not improve this particular benchmark. This is an exploratory comparison in one environment, not a statistically established ranking of physical theories.

## Fixed protocol

- Spatial models train on Invensense trials 1–2, select hyperparameters on trial 3, and test on held-out trials 4–5. Trials 6–9 are a separate environment-change stress test.
- The rectangle $1.5\leq x\leq2.5$, $-2\leq y\leq-1$ metres is withheld from both training and validation, with a 0.15 m buffer. Test observations in it assess spatial transfer.
- Nexus and Trivisio observations never train or select the spatial maps. Each receives a separate affine alignment learned from at most 500 paired observations in trial 1, outside the exclusion.
- Calibration, orientation correction, and position/alignment correction were also supplied upstream. This is transfer after preprocessing and limited device adaptation, not discovery of raw sensor semantics.
- Errors below use the reference Invensense recording's numerical units. The repository README does not explicitly specify a common unit for every sensor; no unverified unit conversion is assumed.

## Models

The unrestricted learner fits three independent smooth kernel maps from position to measured channels. The planar-potential learner ties its first two channels through a shared scalar potential, enforcing $\partial_x B_y-\partial_y B_x=0$. Both learn from real labels, use the same samples and hyperparameter grid, and select configurations using only source validation data. This is a quasistatic, locally current-free Maxwell-inspired assumption; the vertical component is unconstrained. It also assumes that the upstream field-axis alignment matches the position-coordinate frame. Unknown misalignment is an alternative explanation for a poorer constrained fit.

The constant training mean, a linear position map, and a tuned learner trained on position-shuffled labels provide baselines. No scalar-potential constraint is supplied to the unrestricted learner.

## Held-out stable-room trials

Vector RMSE is $\sqrt{\operatorname{mean}\|\widehat{\mathbf b}-\mathbf b\|^2}$. Each table row is a separate held-out trial; no sample-wise significance claim is made.

| Instrument | Trial | Model | All test positions | Withheld rectangle | Improvement over mean |
| --- | ---: | --- | ---: | ---: | ---: |
| invensense | 4 | unrestricted | 8.6599 | 7.4498 (n=105) | 51.8% |
| invensense | 4 | planar_potential | 9.3013 | 10.3505 (n=105) | 48.2% |
| invensense | 4 | shuffled_labels | 18.0922 | 12.4449 (n=105) | -0.7% |
| invensense | 4 | linear_position | 17.3697 | 11.8791 (n=105) | 3.3% |
| invensense | 4 | training_mean | 17.9633 | 13.4961 (n=105) | 0.0% |
| invensense | 5 | unrestricted | 7.4763 | 8.1509 (n=249) | 49.7% |
| invensense | 5 | planar_potential | 7.5542 | 8.3836 (n=249) | 49.1% |
| invensense | 5 | shuffled_labels | 14.7297 | 10.5711 (n=249) | 0.8% |
| invensense | 5 | linear_position | 14.4824 | 10.1103 (n=249) | 2.5% |
| invensense | 5 | training_mean | 14.8534 | 11.2227 (n=249) | 0.0% |
| nexus | 4 | unrestricted | 8.4651 | 6.1509 (n=106) | 50.4% |
| nexus | 4 | planar_potential | 9.4942 | 11.4916 (n=106) | 44.4% |
| nexus | 4 | shuffled_labels | 17.2713 | 10.9386 (n=106) | -1.1% |
| nexus | 4 | linear_position | 16.7533 | 10.5032 (n=106) | 1.9% |
| nexus | 4 | training_mean | 17.0832 | 12.0291 (n=106) | 0.0% |
| nexus | 5 | unrestricted | 11.9279 | 8.9177 (n=246) | 23.1% |
| nexus | 5 | planar_potential | 13.3036 | 16.6006 (n=246) | 14.2% |
| nexus | 5 | shuffled_labels | 15.9579 | 6.9742 (n=246) | -2.9% |
| nexus | 5 | linear_position | 15.5592 | 8.6755 (n=246) | -0.3% |
| nexus | 5 | training_mean | 15.5103 | 8.0101 (n=246) | 0.0% |
| trivisio | 4 | unrestricted | 8.4086 | 8.3376 (n=108) | 52.4% |
| trivisio | 4 | planar_potential | 8.6874 | 8.7243 (n=108) | 50.9% |
| trivisio | 4 | shuffled_labels | 17.7181 | 13.7331 (n=108) | -0.2% |
| trivisio | 4 | linear_position | 17.1664 | 12.9686 (n=108) | 2.9% |
| trivisio | 4 | training_mean | 17.6771 | 14.8726 (n=108) | 0.0% |
| trivisio | 5 | unrestricted | 7.6254 | 8.2423 (n=244) | 49.4% |
| trivisio | 5 | planar_potential | 7.9834 | 8.9129 (n=244) | 47.0% |
| trivisio | 5 | shuffled_labels | 14.9238 | 11.4258 (n=244) | 0.9% |
| trivisio | 5 | linear_position | 14.6060 | 10.4472 (n=244) | 3.0% |
| trivisio | 5 | training_mean | 15.0581 | 11.7748 (n=244) | 0.0% |

## Source validation choices

| Model | Length (m) | Ridge | Validation vector RMSE |
| --- | ---: | ---: | ---: |
| unrestricted | 0.35 | 0.1 | 8.6255 |
| planar_potential | 0.7 | 0.01 | 9.5228 |
| shuffled_labels | 2.8 | 1.0 | 20.8020 |

## Changed-environment stress test

Later trials contain changes to the magnetic surroundings according to the dataset authors. The models remain frozen. Their position-only inputs cannot distinguish two different source arrangements at the same position; this tests the limits of that representation, not a violation of Maxwell.

| Instrument | Trial | Mean baseline | Linear position | Unrestricted | Planar potential |
| --- | ---: | ---: | ---: | ---: | ---: |
| invensense | 6 | 15.0344 | 14.0531 | 8.8247 | 9.3077 |
| invensense | 7 | 21.8136 | 22.0961 | 7.0100 | 10.1437 |
| invensense | 8 | 14.7629 | 14.3726 | 8.7703 | 10.9707 |
| invensense | 9 | 17.9563 | 17.9159 | 9.7305 | 10.7337 |
| nexus | 6 | 17.4079 | 17.7275 | 14.0004 | 14.8673 |
| nexus | 7 | 22.7094 | 24.6292 | 14.4339 | 17.7213 |
| nexus | 8 | 43.4768 | 45.0967 | 42.8267 | 45.0244 |
| nexus | 9 | 17.9313 | 17.6957 | 9.7901 | 10.9602 |
| trivisio | 6 | 15.8647 | 14.5105 | 9.6332 | 9.7202 |
| trivisio | 7 | 21.4789 | 21.2763 | 7.6630 | 10.2891 |
| trivisio | 8 | 14.3448 | 13.4672 | 8.5394 | 10.1978 |
| trivisio | 9 | 16.4354 | 16.3570 | 8.7997 | 9.8253 |

## What this can establish

Lower held-out error than the constant and shuffled-label controls supports reproducible spatial structure. Transfer to the phone or another instrument after frozen calibration supports a description shared across these measurement systems. The potential comparison tests whether one explicit Maxwell-inspired restriction helps this prediction task.

A win for the potential map would not establish Maxwell uniquely; a loss would not refute Maxwell. Sensor drift, calibration, sampling, static-field assumptions, kernel family, finite hyperparameter grid, and source conditions all affect this comparison. Kernel coordinates are supplied; no electromagnetic ontology or representation equivalence has been independently discovered.

Planar observations cannot evaluate full 3D divergence without unmeasured normal derivatives. Electric fields, charge/current channels, wave propagation, and controlled unseen source arrangements are not tested here. The potential model's near-zero interpolated curl is enforced, not an independent empirical recovery.

See [the collection protocol](../COLLECTION_PROTOCOL.md) for the proposed raw two-phone experiment. No new physical measurements have been collected by this run.

## Provenance and reproduction

Data: [AaltoML/magnetic-data](https://github.com/AaltoML/magnetic-data), commit `b45da3bc7ab623fdef6e01247021daebf7983f29`. Attribution: Solin, Kok, Wahlström, Schön, and Särkkä (2018), [Modeling and interpolation of the ambient magnetic field by Gaussian processes](https://doi.org/10.1109/TRO.2018.2830326). The authors describe the recording and preprocessing in the repository README. There is no explicit licence file in the pinned tree; attribution does not itself establish redistribution permissions.

From the workspace root:

```powershell
python experiments/magnetic_mapping/fetch_data.py
python experiments/magnetic_mapping/run_experiment.py
```

[results.json](results.json) includes every validation configuration, per-trial/component errors, calibration coefficients, selected row indices, spatial-coverage diagnostics, and numerical curl checks. `predictions.npz` stores sampled test observations and predictions. The test observations are uniformly spaced in time to cap each trial at the recorded sample limit, not randomly mixed into training.

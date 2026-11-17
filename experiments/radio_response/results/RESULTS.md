# Measured radio response kernels: held-out experiment

**This experiment uses measured DICHASUS CSI. No simulated propagation data are used as labels.**

The target is a relative response between receiver antennas, not the transmitter-to-field Green function. The transmitter's absolute time and phase reference is unavailable; the common ambiguity cancels in the ratio. This is the experimentally identifiable operator we can test with these data.

Read [the theory and protocol](../THEORY_AND_EXPERIMENT.md) for definitions, assumptions, and the connection to Maxwell.

## Observed takeaway

In the north array, the propagation-plus-learning model has test NMSE **0.757**, compared with **0.813** for the generic learner and **0.850** for a constant filter. In the south array, the geometric model has the lowest error, **0.833**, compared with **0.876** for the generic learner and **0.881** for the constant filter.

These are modest improvements of 10.9% and 5.4% in squared error over the constant-filter baseline. Substantial residual error remains. The added geometric propagation structure helps this task; it does not make the predictions complete or identify a fundamental law. In the changed environment, every nonzero fitted model is worse than the zero-output baseline. Spatial-hole scores have 4 and 4 accepted snapshots in the two arrays, too few here for a strong spatial-transfer conclusion.

## Main result: new recording, unchanged hallway

Models train on sampled snapshots from dc01, select settings on dc02, and test on dc04. A two-metre-radius disk around the coordinate origin is excluded from training and validation, with an extra 0.5 m buffer. The disk is a spatial holdout, not a different room.

The primary score is squared complex prediction error divided by measured response energy. Zero is perfect; the zero-output filter scores one. The input at test time is position and the measured reference-antenna response. The other antennas' responses are held-out targets.

| Array | Model | Test snapshots | Transfer NMSE | Spatial-hole snapshots | Spatial-hole NMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | zero_filter | 234 | 1.000000 | 4 | 1.000000 |
| 0 | constant_filter | 234 | 0.850017 | 4 | 0.943059 |
| 0 | learned_filter | 234 | 0.813016 | 4 | 0.985472 |
| 0 | geometric_green | 234 | 0.779979 | 4 | 1.001430 |
| 0 | green_envelope | 234 | 0.757130 | 4 | 0.954705 |
| 0 | shuffled_geometry | 234 | 0.928857 | 4 | 1.047245 |
| 1 | zero_filter | 239 | 1.000000 | 4 | 1.000000 |
| 1 | constant_filter | 239 | 0.880704 | 4 | 0.822094 |
| 1 | learned_filter | 239 | 0.876028 | 4 | 0.823016 |
| 1 | geometric_green | 239 | 0.833335 | 4 | 0.735700 |
| 1 | green_envelope | 239 | 0.846590 | 4 | 0.785003 |
| 1 | shuffled_geometry | 239 | 1.044637 | 4 | 0.978903 |

## Different-environment stress test

The frozen models also predict dc13, where a metal obstruction changes the source environment. No fitting or convention selection uses its labels. Position alone cannot specify that changed environment, so failure here is a limit of the supplied representation.

| Array | Model | Snapshots | Transfer NMSE |
| --- | --- | ---: | ---: |
| 0 | zero_filter | 86 | 1.000000 |
| 0 | constant_filter | 86 | 1.119644 |
| 0 | learned_filter | 86 | 1.307022 |
| 0 | geometric_green | 86 | 1.187624 |
| 0 | green_envelope | 86 | 1.255754 |
| 0 | shuffled_geometry | 86 | 1.205622 |
| 1 | zero_filter | 104 | 1.000000 |
| 1 | constant_filter | 104 | 1.157257 |
| 1 | learned_filter | 104 | 1.286849 |
| 1 | geometric_green | 104 | 1.258504 |
| 1 | green_envelope | 104 | 1.291058 |
| 1 | shuffled_geometry | 104 | 1.486438 |

## Models compared

- **zero_filter:** predicts no received output, providing an NMSE reference of one.
- **constant_filter:** a measured least-squares complex coupling per receiver and frequency, independent of position.
- **shuffled_geometry:** the same learner and selection grid as learned_filter, with measured quotient vectors shuffled across training positions within each array task.
- **learned_filter:** smooth kernel regression from position to the complex relative response; it contains no propagation-speed or Maxwell constraint.
- **geometric_green:** a direct-path free-space Green-function ratio, with constant receiver/frequency gains fitted to training observations. It supplies the known propagation speed and uses published antenna geometry.
- **green_envelope:** the same learned regression applied after factoring out geometric propagation, testing a combined physics/learning representation.

Every model is evaluated on the same accepted snapshots for its array. Discrete antenna row/column axis signs and complex phase sign are selected on validation data only. They calibrate a coordinate/IQ convention, not a newly measured physical law. Models predict the same receiver ports used during training; this experiment does not claim unseen-instrument transfer.

## Data audit

The experiment downloads 768 whole TFRecords from four public files using fixed evenly spaced record indices and bounded HTTP byte ranges. Each selected record has its framing, CRC32C and local SHA-256 checked. The full multi-gigabyte files are not downloaded, so their published full-file MD5 values are recorded but not claimed as verified.

Only 64 equally spaced frequency bins from each 1024-bin measured spectrum are modeled. The native bin ordering follows the published measurement-processing code: applying fftshift yields the centered frequency ordering. The nominal carrier is 3.438 GHz and bandwidth 50 MHz.

Reference antennas are selected by median training SNR within each array. Fixed quality rules require finite relevant data, a reference SNR statistic above zero, position interpolation age within 0.1 s, and non-negligible reference amplitude in every selected bin. The amplitude floor is 2% of the per-bin training reference RMS; no validation/test labels set it. Exclusions and retained original indices are listed in results.json. No snapshot is removed because a model predicts it poorly.

Per-port, per-frequency gain fits use only training responses. The spec warns that offsets may be accurate within arrays but not between arrays, so each relative-response task stays within one physical receive array. No cross-array phase coherence is assumed.

## What the result means

Lower transfer error demonstrates prediction of a measured radio response operator at held-out positions/recordings. A benefit from green_envelope would support the usefulness of geometric propagation as a representation for this task. A good learned_filter alone demonstrates empirical regularity; it does not uniquely identify its physical explanation.

The direct-path comparator is one limited Maxwell-based approximation. It omits full reflections, diffraction, scattering, antenna patterns, coupling and a complete boundary-value solution. Its success or failure does not settle Maxwell versus ML in general. The geometric propagation speed is supplied, not discovered. Frequency-dependent constant electronic responses are learned as nuisance parameters.

The relative operator can have negative delay because the reference receiver may see a wave later than another receiver. Its finite-band IFFT is periodic and only a representation of measured relative response. Therefore it cannot independently establish absolute causality, a light cone, physical travel time, or light speed. No new transmitted waveforms or source currents are measured here. The experiment evaluates the kernel through the actual common sounding response, not invented test waveforms. See the theory document for the distinction between a causal field Green function and this relative antenna filter.

## Reproduce and inspect

```powershell
python experiments/radio_response/fetch_data.py
python experiments/radio_response/run_experiment.py
python -m pytest experiments/radio_response/tests -q
```

[results.json](results.json) contains the split audit, quality counts, training-selected reference ports and amplitude floors, all validation searches, per-receiver errors, spatial coverage and source hashes. `predictions.npz` contains measured reference/target responses and the predicted complex relative filters. `comparison.png` shows the test results. No earlier phone or optical study's outputs are overwritten.

[Relative impulse-response examples](relative_impulse_responses.png) show the fixed first accepted test snapshot and first target port for each array. `impulse_examples.npz` stores those curves. They illustrate the modeled relative filter; they are not additional independent validation.

Data attribution: Euchner, Stephan, and Gauger (2023), [CSI Dataset dichasus-dcxx: Institute hallway with known 3D model, distributed antennas](https://doi.org/10.18419/DARUS-3831), DaRUS version 1.0, CC BY 4.0. The retained `spec.json` provides the source conventions and warnings. Frequency-order interpretation is also supported by [the authors' measurement-processing code](https://github.com/NVlabs/diff-rt-calibration/blob/main/code/utils.py).

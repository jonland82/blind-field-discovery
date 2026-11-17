# Active research plan: public measurements only

The question is how a learned description of observations can acquire the transfer, structure, and scope of a physical theory. Maxwell is a candidate description and a comparison point. Agreement with Maxwell is not built into the unconstrained learners' objective or used as the criterion for accepting their predictions.

**Constraint:** use publicly available measured data. New phone recordings and purchases are not required. The controlled collection plan is an inactive reference, not an unfinished prerequisite.

## Completed experiments

| Public source | What is tested | What the data cannot establish here |
| --- | --- | --- |
| [Aalto magnetic data](https://github.com/AaltoML/magnetic-data) | Transfer from one magnetometer to a phone and another magnetometer, across held-out trials and a spatial exclusion | Raw variable discovery, observer rotations before preprocessing, or complete Maxwell dynamics |
| [FIU MARG data](https://github.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations) | Prediction across held-out recordings under independently optically measured position and orientation | Transfer to a new instrument, a unique latent ontology, or dense full-system electromagnetic identification |
| [DICHASUS radio responses](https://doi.org/10.18419/DARUS-3831) | Learned and geometric relative antenna-response kernels across acquisitions, spatial exclusion, and a changed environment | Absolute transmitter timing/causality, light-speed identification, or a complete hallway Maxwell solution |

In the [Aalto pilot](results/RESULTS.md), a learned map transfers usefully after frozen instrument calibration. In the [FIU study](public_transfer/RESULTS.md), a generic pose learner and an explicit shared-vector field model both predict new recordings substantially better than a constant or shuffled-data control. Adding a current-free static-potential restriction increases pooled errors in both studies. These are finite model comparisons with source, sensor, coordinate, and stationarity assumptions; they do not establish that Maxwell is wrong or independently recovered.

In the [radio-response follow-up](../radio_response/results/RESULTS.md), geometric propagation is useful: the combined representation wins in one array and the geometric model wins in the other. The gains over constant filters are modest, and all fitted models fail to beat zero output after the environment changes. The [theory document](../radio_response/THEORY_AND_EXPERIMENT.md) identifies precisely which kernel is learned and why the data do not support an absolute light-cone test.

The [real magnetic-profile extraction](symbolic_field/RESULTS.md) now applies the toy benchmark's kernel-to-equation-to-solved-function workflow to actual Aalto readings. A training-defined 0.3 m spatial band and four fixed local intervals provide a simplified scalar field-strength problem. Whole-band kernel RMSE on the reference instrument is 2.879 versus 8.100 for a constant, while a single global Riccati equation gives 16.992. Local analytic candidates give pooled RMSE 2.936 versus 2.636 for local kernels, but no region passes all derivative and prediction requirements. This is exploratory reuse of previously scored acquisitions. The next issue is reliable derivative identification and representation scope, rather than equating value-fit accuracy with a recovered field law.

## Next analyses that remain possible without collecting data

1. **Diagnose the observation model.** Use training and validation recordings to examine optical/sensor frame alignment, fixed bias versus more general sensor response, marker-to-sensor offsets, and stationarity. Any revised model needs a separately designated new evaluation protocol; the current test recordings have already been scored.
2. **Test a representation-level bridge.** Assess whether the unconstrained predictor's outputs, transformed through independent poses, admit a consistent common-field description at repeated locations. Use new withheld recordings or an independent dataset to distinguish a prediction fit from a transferable observation rule. Explicitly record which transformations were learned and which were supplied.
3. **Replicate across public phone models.** The [Tyrex smartphone attitude benchmark](https://github.com/tyrex-team/benchmarks-attitude-smartphones) provides phone sensor streams and motion-capture references. Its documentation has been inspected, but it has not been downloaded or analyzed in this project. This is a candidate independent replication; synchronization, device-frame conventions, calibration, and acquisition grouping need to be handled before testing.
4. **Approach more of Maxwell with richer observables.** [MMS public datasets](https://lasp.colorado.edu/mms/sdc/public/datasets/) include independently instrumented electric and magnetic fields, particle distributions, and spacecraft geometry across four spacecraft. This is a candidate for studying spatial/temporal relations unavailable in phone recordings. It has not yet been downloaded or analyzed here. Any derivative estimate would require checking tetrahedron geometry, synchronization, scales, measurement uncertainty, and product quality. Source quantities derived from particle measurements must be distinguished from quantities defined using Maxwell itself.

MMS is a possible later change of experimental regime, not a claim that all of Maxwell can automatically be identified from its available products. The phone and optical studies ask whether a common observation description transfers; the richer-source study would ask which coupled field relationships can be identified from independently measured channels.

## Evidence rules

- Use only actual observations as experimental labels. Artificial arrays used in software tests are temporary fixtures, not evidence.
- Hold out whole acquisition groups. Fit preprocessing, calibration, nuisance parameters, representations, and model choices without those held-out labels.
- Report failed transfer, difficult recordings, shuffled controls, and per-recording errors alongside pooled results. Do not remove a recording because a model predicts it poorly.
- Distinguish an assumed vector transformation or physics constraint from an empirically recovered relationship.
- Preserve downloaded versions, upstream documentation, file hashes, selected rows, predictions, and the scope of every conclusion.

These rules make the shared experimental evidence the link between ML and physical theory. They do not require treating either the learned representation or Maxwell as a privileged ground truth.

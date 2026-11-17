# Public-data experiment: does a shared field explain changing observations?

**Completed using 30 public measured recordings. No new hardware or synthetic observations are required.**

The FIU MARG dataset records a magnetic sensor's readings, positions, and orientations obtained from an independent optical system. We compare descriptions that make different assumptions about how a common field appears through a rotating instrument. This extends the earlier [multi-instrument phone pilot](../results/RESULTS.md); it uses a different dataset and a single MARG sensor, not two phones.

## Result

| Description | Pooled held-out vector RMSE (gauss) | Improvement over constant |
| --- | ---: | ---: |
| constant | 0.638139 | 0.0% |
| shuffled_pose | 0.605167 | 5.2% |
| pose_learner | 0.277243 | 56.6% |
| shared_vector | 0.286452 | 55.1% |
| static_potential | 0.376013 | 41.1% |

The lowest pooled test error among the three substantive descriptions is **pose_learner**. This is a prediction result under the stated observation, calibration, stationarity, and kernel assumptions. It is not independent discovery or rejection of Maxwell's equations.

Median per-recording RMSE is 0.113081 gauss for the pose learner, 0.122116 for the shared-vector model, and 0.200570 for the static-potential model. Recording 20 has much larger errors for every description and strongly affects the pooled metric. It is retained in every comparison; the table below shows the complete per-recording results.

## What each description assumes

- **pose_learner:** learns three separate functions from supplied optical position and orientation to sensor readings. It is not told that its three outputs form a rotating vector.
- **shared_vector:** learns a three-component field over optical position. The model explicitly assumes a vector transformation between that field and the sensor's readings, plus a fixed sensor bias.
- **static_potential:** uses the same observation transformation and learns a scalar-potential field, adding a curl-free constraint consistent with a locally current-free quasistatic region.
- **constant:** predicts the training-channel mean.
- **shuffled_pose:** fits and selects the same kind of model as pose_learner after measured vectors are shuffled within each training recording, destroying their alignment with pose.

The shared-vector observation equation is

$$
\mathbf b_i=R_i^{\mathsf T}\mathbf B(\mathbf x_i)+\mathbf c+\boldsymbol\eta_i.
$$

The potential model additionally uses

$$
\mathbf B(\mathbf x)=-\nabla\psi(\mathbf x),\qquad\nabla\times\mathbf B=0.
$$

These are explicit candidate representations. A good result for a shared-vector description would support its usefulness across these recordings; it would not show that the vector transformation was discovered without assumptions. The unconstrained learner's outputs remain in the instrument frame.

## Split and controls

- Recordings 01–10 train the models; 11–15 select settings; 16–30 supply held-out scores.
- At most 50 equally spaced accepted observations per training recording and 100 per validation recording are used. Each test recording contributes at most 400 accepted observations.
- Missing optical tracks and invalid quaternions are excluded by fixed rules. Time gaps are not interpolated. The camera quaternion columns are used; onboard magnetically informed orientation estimates are not.
- Quaternion matrices and their transposes are two declared coordinate-convention candidates for the shared-vector/potential maps. Validation chooses between them. This is convention selection, not discovery of a new physical transformation.
- A common background vector and fixed sensor bias are fitted to training observations only. Each spatial field is learned from the residuals. No test-recording calibration or magnetic labels are used for fitting or selection.
- Whole recordings are held out. These are repeated observations in one laboratory; they are not 15 independent laboratories or proof of new-device transfer.

## Per-recording held-out errors

All errors below are vector RMSE in gauss. Each row is one held-out recording.

| Recording | Constant | Shuffled | Pose learner | Shared vector | Static potential |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.482679 | 0.462443 | 0.091365 | 0.098624 | 0.200570 |
| 17 | 0.735169 | 0.679907 | 0.245765 | 0.294284 | 0.456609 |
| 18 | 0.268456 | 0.315749 | 0.136950 | 0.122116 | 0.156267 |
| 19 | 0.472844 | 0.469110 | 0.097640 | 0.120569 | 0.217167 |
| 20 | 1.757350 | 1.636240 | 0.962494 | 0.977216 | 1.197341 |
| 21 | 0.432647 | 0.412144 | 0.108617 | 0.116626 | 0.148612 |
| 22 | 0.281121 | 0.302273 | 0.122795 | 0.112360 | 0.142073 |
| 23 | 0.556651 | 0.507361 | 0.094146 | 0.105595 | 0.183785 |
| 24 | 0.390396 | 0.385928 | 0.109726 | 0.138165 | 0.200950 |
| 25 | 0.579725 | 0.562566 | 0.119113 | 0.129069 | 0.228311 |
| 26 | 0.347948 | 0.333900 | 0.113081 | 0.101274 | 0.223137 |
| 27 | 0.404663 | 0.386921 | 0.099050 | 0.128916 | 0.241382 |
| 28 | 0.273148 | 0.311655 | 0.120955 | 0.136393 | 0.180178 |
| 29 | 0.419315 | 0.423748 | 0.134808 | 0.116356 | 0.163421 |
| 30 | 0.592548 | 0.546149 | 0.110246 | 0.133970 | 0.176296 |

## Selected settings

| Model | Validation-selected configuration |
| --- | --- |
| pose_learner | `{"length_metres": 0.06, "orientation_length": 0.5, "ridge": 0.001, "validation_vector_rmse_gauss": 0.3110086428913896}` |
| shared_vector | `{"convention": "as_recorded", "length_metres": 0.06, "ridge": 0.01, "validation_vector_rmse_gauss": 0.2926407559528729}` |
| static_potential | `{"convention": "as_recorded", "length_metres": 0.06, "ridge": 0.1, "validation_vector_rmse_gauss": 0.4650455980847015}` |
| shuffled_pose | `{"length_metres": 0.18, "orientation_length": 4.0, "ridge": 0.1, "validation_vector_rmse_gauss": 0.5979273750421746}` |

## What this gets us toward a theory

The bridge tested here is whether one field description can explain how observations change with independently measured pose. The unconstrained model can learn pose-dependent regularities; the shared-vector and potential descriptions compress those regularities using stronger structure. Prediction on new recordings evaluates the usefulness of that structure.

The experiment does not learn a unique latent ontology, establish electromagnetic constants, observe electric fields or charge/current channels, or recover the complete Maxwell system. No divergence constraint is imposed. Although positions have three coordinates, the observations follow a restricted manipulation protocol rather than a dense 3D field survey, so full spatial-law identification should not be inferred from the potential-model score.

Failures may reflect sensor response, optical/sensor alignment, marker-to-sensor displacement, time synchronization, nonstationarity, or the finite feature/hyperparameter choices. We do not select away those failures based on held-out magnetic errors. Optical coordinates and a smooth kernel are supplied assumptions in every substantive description. Agreement among models is not itself proof of a unique Platonic representation.

## Evidence and reproduction

Data: [FIUMARGDB](https://github.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations), commit `998c9ebbd243b3e13ba2dc8bbf94e958b6be1b8b`. Retained source documentation and LICENSE accompany the local download. Cite Sonchan and colleagues (2023), [Benchmarking Dataset of Signals from a Commercial MEMS MARG Sensor](https://doi.org/10.3390/s23083786). The paper describes the optical reference and magnetic data; its magnetic readings are in gauss and positions in metres.

From the workspace root:

```powershell
python experiments/magnetic_mapping/fetch_fiu.py
python experiments/magnetic_mapping/run_fiu.py
```

[results.json](results.json) includes every selected setting, grid score, original row index, data-quality count, bias/background fit, software version, and provenance reference. `predictions.npz` preserves held-out observations, optical poses, and every prediction. The original phone pilot is preserved in `experiments/magnetic_mapping/results`.

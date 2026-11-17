# Controlled two-phone experiment

**Inactive design reference:** the project now relies on public recordings. This protocol is preserved for context; collecting its 192 new recordings is not required. See [the completed public-data studies](README.md) and [active research plan](PUBLIC_DATA_PLAN.md).

**Question:** Can relationships learned from phone A predict phone B, new magnet arrangements, new observation configurations, and a later recording session? Does a Maxwell-based source/sensor description predict the same observations better or worse?

The acquisition plan is ready at [collection/plan.csv](collection/plan.csv). [collection/samples.csv](collection/samples.csv) contains only a header. It must be filled with real recordings before the controlled experiment can run. The public-data pilot in [results/RESULTS.md](results/RESULTS.md) is separate evidence.

## Apparatus and conventions

Use two phones, one small permanent magnet with a marked axis, and a nonmagnetic fixture with measured positions and four repeatable phone orientations. Use the **same physical magnet** in all present-source arrangements. Keep the magnet much smaller than its separation from each sensor if testing the point-dipole approximation.

The positions in the plan refer to the magnetometer sensing location, not the center of the phone case. Put one phone at a time into the same fixture. Remove the other phone from the measurement area. This avoids treating readings from displaced sensors as if they were observations at one point. Record actual apparatus dimensions and deviations.

Define one right-handed fixture coordinate system. In `plan.csv`, each quaternion is ordered $(q_w,q_x,q_y,q_z)$, and its rotation matrix maps sensor axes into fixture axes. Establish the fixture rotations mechanically or with an independent optical reference. A magnetometer-derived compass heading would feed magnetic information into the representation we intend to test.

The plan provides desired geometry. Correct its numeric position, orientation, and source-axis fields to the **measured** fixture geometry before analysis, consistently across repeated conditions. Do not alter the condition IDs or split assignments after inspecting prediction errors. Keep a dated copy of the finalized plan before recording.

For each device, record in an acquisition log:

- Model, operating system, recording application/API, sample rate, and channel units.
- Whether the API applies calibration, bias correction, filtering, or automatic recalibration; use the same settings throughout.
- Magnetometer location estimate, fixture-axis convention, source dimensions, and how its marked axis was measured.
- Session timing, nearby objects, changes to the apparatus, and acquisition problems.

The supplied code is an offline CSV analyzer; it does not install a phone app or remotely operate sensors. Use a recorder that exports timestamped three-axis readings. Store original exports alongside the converted CSV.

## Recording schedule

There are 192 planned recordings: two devices, two sessions, three positions, four orientations, and four source states. At five seconds each this is 16 minutes of recorded signal, plus time to change arrangements. The second session should be a separately restarted recording session, preferably later in the day or on another day, with its conditions documented.

The four source states are: magnet absent, two training magnet arrangements, and a third magnet arrangement held out from source training and calibration. The absent-source observations measure how orientation and background affect the instrument without the introduced magnet.

Follow the randomized order in `plan.csv` within each session. It randomizes physical arrangements and the order of the two devices for each arrangement. Keep all arrangements in the plan, including ones that give uninteresting readings. Record a failed acquisition as such, fix the acquisition issue, and repeat it without selecting trials based on prediction performance.

The fixed split is:

| Data | Purpose |
| --- | --- |
| Phone A, session 1, absent/M1/M2, ordinary geometry | Training |
| Phone A, session 1, absent/M1/M2 at P1/O3 | Validation only |
| Phone A, session 1, absent/M1/M2 at P2/O4 | Unseen position/orientation combination |
| Phone A, session 1, M3 | Unseen source arrangement |
| Phone B, session 1, P3 with absent/M1/M2 | Twelve device-calibration arrangements, paired with phone A training conditions |
| Other phone B, session 1, absent/M1/M2 | Instrument transfer |
| Phone B, session 1, M3 | Joint source and instrument transfer |
| Both phones, session 2 | Session transfer with every fitted model frozen |

Phone B is not entirely unseen: it supplies twelve calibration trial means. Its other recordings are held out. Report that distinction rather than calling the result zero-shot device discovery.

## CSV format

Append actual readings to `collection/samples.csv`:

```csv
trial_id,timestamp_s,u0,u1,u2,unit
```

No example numeric readings are provided, to avoid confusing a demonstration with measurements. Copy each `trial_id` from `plan.csv`. `timestamp_s` must increase within that trial. `u0,u1,u2` are the recorder's three magnetic channels in its original sensor frame; do not rotate them or enforce a physical equation in preprocessing. `unit` must be a nonempty, consistent label per device, such as the documented physical unit or a documented raw device-unit label.

Record at least five seconds per trial. The analyzer discards the first second as a settling interval, requires a total span of at least four seconds and at least twenty retained samples, and uses the mean of the remaining samples for the static prediction task. It also saves within-trial standard deviations. Those deviations describe recorded variation; correlated samples are not treated as independent experimental replications.

Malformed timestamps, unknown trial IDs, changed units within a device, invalid quaternions, or nonfinite values are errors. Missing or too-short trials leave the collection awaiting measurements. The analyzer does not silently fill missing trials with predicted or synthetic values.

## Analyze the real collection

From the workspace root:

```powershell
python experiments/magnetic_mapping/controlled_experiment.py plan
python experiments/magnetic_mapping/controlled_experiment.py analyze
```

`plan` creates missing templates and preserves existing recordings. `analyze` currently writes an explicit `awaiting_measurements` status to [controlled_results/collection_status.json](controlled_results/collection_status.json). Once all recordings are complete, it writes `controlled_results/results.json`, `RESULTS.md`, and `predictions.npz`. Existing result files from a previous complete run are not evidence about an incomplete new collection: always check `collection_status.json` first.

The empirical models receive numeric fixture geometry: sensor position, orientation matrix, magnet position, marked magnet-axis direction, and whether the magnet is present. Trial IDs, source IDs, device IDs, and session IDs are used to assign and report splits; they are not predictor features. Thus the task tests transfer across arrangements given measured geometry, not spontaneous discovery of geometry or electromagnetic variables.

The unconstrained learner is smooth kernel regression. A regularized linear geometry model, training mean, and shuffled-label learner are controls. Model choices use phone A validation trials only. A shared affine map converts phone B's channels to phone A's numerical representation using only the twelve paired calibration arrangements. The calibration is frozen for testing.

The physical comparator assumes a uniform ambient field plus a small point magnet:

$$
\mathbf b_A
=R^{\mathsf T}\left[
\mathbf B_{\rm background}
+s\,p\,\frac{3\widehat{\mathbf r}(\widehat{\mathbf r}\cdot\widehat{\mathbf m})-\widehat{\mathbf m}}{\|\mathbf r\|^3}
\right]+\mathbf c_A.
$$

Here $\mathbf r$ is sensor position minus magnet position, $\widehat{\mathbf m}$ is its marked axis, $p$ is zero when the magnet is absent and one otherwise, $R$ maps phone A axes to fixture axes, and $s$ absorbs magnet strength and unit factors. The three background components, one shared strength, and three fixed sensor-bias components are fitted using phone A training trials only. No expected strength or field labels are generated for the learner.

This comparator assumes an ideal vector response plus constant reference-phone bias, a point-dipole source, a stable uniform background, and quasistatic conditions. Reference-phone scale anisotropy, environmental magnetic structure, finite magnet size, fixture errors, or drift can violate those assumptions. Losing to ML would challenge this complete approximation on these experiments; it would not isolate which assumption failed or refute Maxwell.

## What to report

Report vector prediction error separately for each device and held-out split, together with each trial's recorded variability. Compare the learned models, controls, and physical comparator on exactly the same observations. Do not tune models after seeing held-out errors and continue to call those recordings untouched tests; collect a new test session for a revised model.

Success means accurate, reproducible transfer to the specified new experiments. A stronger ML-to-theory result would additionally require identifying a consistent transformation from learned variables to a physical description and validating its relationships under further interventions. This protocol establishes a concrete first comparison; it does not measure electric fields, recover the complete Maxwell system, or test wave dynamics.

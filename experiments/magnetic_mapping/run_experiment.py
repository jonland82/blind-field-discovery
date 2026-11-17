"""Run a real-data, held-out-trial comparison of learned magnetic field maps.

No synthetic observations are generated. The test instruments and recordings
never select the spatial models or their hyperparameters. Optional instrument
alignment uses only paired observations from calibration trial 1.
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from fetch_data import COMMIT, SENSORS
from models import FieldMap, LinearMap, planar_curl

ROOT = Path(__file__).resolve().parent
SOURCE = "invensense"
TRAIN_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
TEST_TRIALS = (4, 5)
SHIFT_TRIALS = (6, 7, 8, 9)
LENGTHS = (0.35, 0.7, 1.4, 2.8)
RIDGES = (0.001, 0.01, 0.1, 1.0)
HOLE = (1.5, 2.5, -2.0, -1.0)  # Fixed in metres before fitting any models.
BUFFER = 0.15


@dataclass
class Recording:
    sensor: str
    trial: int
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    original_rows: np.ndarray
    rejected_rows: np.ndarray


def in_hole(x, buffer=0.0):
    left, right, bottom, top = HOLE
    return ((x[:, 0] >= left - buffer) & (x[:, 0] <= right + buffer)
            & (x[:, 1] >= bottom - buffer) & (x[:, 1] <= top + buffer))


def evenly_spaced(indices, maximum):
    indices = np.asarray(indices)
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def verify_archive(data):
    manifest = json.loads((data / "provenance.json").read_text())
    if manifest.get("commit") != COMMIT:
        raise ValueError("Data commit differs from the pinned protocol")
    expected = {f"data/{sensor}/{trial}-{kind}.csv"
                for sensor in SENSORS for trial in range(1, 10)
                for kind in ("loc", "mag", "time")}
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if not expected <= entries.keys():
        raise ValueError("Incomplete data provenance manifest")
    for name, entry in entries.items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Data hash mismatch: {name}")
    return manifest


def load_recording(data, sensor, trial):
    base = data / "data" / sensor
    x = np.loadtxt(base / f"{trial}-loc.csv", delimiter=",")
    y = np.loadtxt(base / f"{trial}-mag.csv", delimiter=",")
    t = np.loadtxt(base / f"{trial}-time.csv", delimiter=",")
    if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x), 3) or t.shape != (len(x),):
        raise ValueError(f"Malformed recording: {sensor}, trial {trial}")
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError(f"Non-finite or non-increasing time: {sensor}, trial {trial}")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if finite.sum() < 20:
        raise ValueError(f"Insufficient complete observations: {sensor}, trial {trial}")
    return Recording(sensor, trial, x[finite], y[finite], t[finite],
                     np.flatnonzero(finite), np.flatnonzero(~finite))


def spatial_split(recordings, trials, maximum):
    xs, ys, selected = [], [], []
    for trial in trials:
        rec = recordings[SOURCE, trial]
        indices = evenly_spaced(np.flatnonzero(~in_hole(rec.x, BUFFER)), maximum)
        if len(indices) < 10:
            raise ValueError(f"Insufficient observations outside the spatial exclusion in trial {trial}")
        xs.append(rec.x[indices])
        ys.append(rec.y[indices])
        selected.append({"sensor": SOURCE, "trial": trial, "row_indices": indices.tolist()})
    return np.concatenate(xs), np.concatenate(ys), selected


def affine_fit(inputs, targets, minimum_pairs=20):
    design = np.column_stack((inputs, np.ones(len(inputs))))
    if len(inputs) < minimum_pairs or np.linalg.matrix_rank(design) < 4:
        raise ValueError(f"Calibration requires at least {minimum_pairs} pairs spanning an affine 3D map")
    coefficients, _, _, singular = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients, float(singular[0] / singular[-1])


def apply_alignment(values, coefficients):
    return values @ coefficients[:3] + coefficients[3]


def calibrate(reference, target):
    """Learn target -> reference units/frame from trial 1 only, outside the hole.

    The paired source readings are actual measurements, not spatial-model
    predictions. Nearby timestamps alone are insufficient: positions must also
    agree within 0.10 m. No time shift is optimized using target test labels.
    """
    if reference.trial != 1 or target.trial != 1:
        raise ValueError("Device alignment may use only calibration trial 1")
    nearest = np.searchsorted(reference.t, target.t)
    high = np.clip(nearest, 0, len(reference.t) - 1)
    low = np.clip(nearest - 1, 0, len(reference.t) - 1)
    nearest = np.where(abs(reference.t[low] - target.t) <= abs(reference.t[high] - target.t), low, high)
    distance = np.linalg.norm(reference.x[nearest] - target.x, axis=1)
    dt = abs(reference.t[nearest] - target.t)
    eligible = ((distance <= 0.10) & (dt <= 0.03)
                & ~in_hole(reference.x[nearest], BUFFER) & ~in_hole(target.x, BUFFER))
    indices = evenly_spaced(np.flatnonzero(eligible), 500)
    coefficients, condition = affine_fit(target.y[indices], reference.y[nearest[indices]])
    fitted = apply_alignment(target.y[indices], coefficients)
    return coefficients, {
        "trial": 1, "pairs": len(indices), "target_row_indices": indices.tolist(),
        "reference_row_indices": nearest[indices].tolist(),
        "max_time_difference_seconds": float(dt[indices].max()),
        "max_position_difference_metres": float(distance[indices].max()),
        "design_condition_number": condition,
        "calibration_vector_rmse": float(np.sqrt(np.mean(np.sum((fitted - reference.y[nearest[indices]])**2, axis=1)))),
        "target_to_reference_affine": coefficients.tolist(),
    }


def vector_rmse(actual, predicted):
    return float(np.sqrt(np.mean(np.sum((actual - predicted)**2, axis=1))))


def metrics(actual, predicted, mask):
    if not np.any(mask):
        return {"n": 0, "vector_rmse": None, "component_rmse": None, "horizontal_vector_rmse": None}
    error = actual[mask] - predicted[mask]
    return {"n": int(mask.sum()), "vector_rmse": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "component_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "horizontal_vector_rmse": float(np.sqrt(np.mean(np.sum(error[:, :2]**2, axis=1))))}


def tune(kind, x, y, xv, yv):
    trials = []
    best = None
    for length in LENGTHS:
        for ridge in RIDGES:
            model = FieldMap(kind, length, ridge).fit(x, y)
            score = vector_rmse(yv, model.predict(xv))
            trials.append({"length_metres": length, "ridge": ridge, "validation_vector_rmse": score})
            if best is None or score < best[0]:
                best = score, model, {"length_metres": length, "ridge": ridge}
    return best[1], {"selected": best[2], "validation_vector_rmse": best[0], "search": trials}


def write_report(result, output):
    rows = result["evaluations"]
    baseline = {(r["sensor"], r["trial"]): r for r in rows if r["model"] == "training_mean"}
    pooled = {}
    for sensor in SENSORS:
        pooled[sensor] = {}
        for model in ("training_mean", "linear_position", "shuffled_labels", "unrestricted", "planar_potential"):
            group = [r["scores"]["all"] for r in rows if r["sensor"] == sensor
                     and r["trial"] in TEST_TRIALS and r["model"] == model]
            pooled[sensor][model] = float(np.sqrt(sum(g["n"] * g["vector_rmse"]**2 for g in group)
                                                  / sum(g["n"] for g in group)))
    phone = pooled["nexus"]
    improvement = 100 * (1 - phone["unrestricted"] / phone["training_mean"])
    lines = [
        "# Real-data pilot: learning across magnetometers", "",
        "**This run uses measured magnetic recordings, not generated fields.** It compares "
        "prediction across later trials and instruments. It does not discover the full Maxwell system.", "",
        "The public recording has one Nexus phone and two robot-mounted magnetometers. "
        "It is a proxy for the proposed two-phone experiment, not a new two-phone collection.", "",
        "## Observed result", "",
        f"On the held-out phone trials, pooled vector RMSE is **{phone['unrestricted']:.4f}** for the "
        f"unrestricted learner, **{phone['training_mean']:.4f}** for the constant baseline, "
        f"**{phone['linear_position']:.4f}** for linear regression, and **{phone['planar_potential']:.4f}** "
        f"for the potential-constrained map. The unrestricted learner reduces error by **{improvement:.1f}%** "
        "relative to the constant baseline. All numbers use the reference recording's numerical units after "
        "frozen device alignment.", "",
        "In this run, the unrestricted map has lower error than the potential map on both stable-room test "
        "trials for every instrument. Spatial-hole performance is mixed: for phone trial 5, neither learned "
        "map improves on the constant baseline inside the excluded rectangle. The evidence supports useful "
        "predictive structure, with limits to spatial and device transfer. The imposed static constraint does "
        "not improve this particular benchmark. This is an exploratory comparison in one environment, not "
        "a statistically established ranking of physical theories.", "",
        "## Fixed protocol", "",
        "- Spatial models train on Invensense trials 1–2, select hyperparameters on trial 3, "
        "and test on held-out trials 4–5. Trials 6–9 are a separate environment-change stress test.",
        "- The rectangle $1.5\\leq x\\leq2.5$, $-2\\leq y\\leq-1$ metres is withheld from both "
        "training and validation, with a 0.15 m buffer. Test observations in it assess spatial transfer.",
        "- Nexus and Trivisio observations never train or select the spatial maps. Each receives "
        "a separate affine alignment learned from at most 500 paired observations in trial 1, outside the exclusion.",
        "- Calibration, orientation correction, and position/alignment correction were also supplied upstream. "
        "This is transfer after preprocessing and limited device adaptation, not discovery of raw sensor semantics.",
        "- Errors below use the reference Invensense recording's numerical units. The repository README "
        "does not explicitly specify a common unit for every sensor; no unverified unit conversion is assumed.", "",
        "## Models", "",
        "The unrestricted learner fits three independent smooth kernel maps from position to measured channels. "
        "The planar-potential learner ties its first two channels through a shared scalar potential, enforcing "
        "$\\partial_x B_y-\\partial_y B_x=0$. Both learn from real labels, use the same samples and "
        "hyperparameter grid, and select configurations using only source validation data. This is a "
        "quasistatic, locally current-free Maxwell-inspired assumption; the vertical component is unconstrained. "
        "It also assumes that the upstream field-axis alignment matches the position-coordinate frame. "
        "Unknown misalignment is an alternative explanation for a poorer constrained fit.", "",
        "The constant training mean, a linear position map, and a tuned learner trained on position-shuffled labels provide baselines. "
        "No scalar-potential constraint is supplied to the unrestricted learner.", "",
        "## Held-out stable-room trials", "",
        "Vector RMSE is $\\sqrt{\\operatorname{mean}\\|\\widehat{\\mathbf b}-\\mathbf b\\|^2}$. "
        "Each table row is a separate held-out trial; no sample-wise significance claim is made.", "",
        "| Instrument | Trial | Model | All test positions | Withheld rectangle | Improvement over mean |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["trial"] not in TEST_TRIALS:
            continue
        all_score = row["scores"]["all"]["vector_rmse"]
        hole = row["scores"]["spatial_hole"]
        base = baseline[row["sensor"], row["trial"]]["scores"]["all"]["vector_rmse"]
        hole_text = "n/a" if hole["n"] == 0 else f"{hole['vector_rmse']:.4f} (n={hole['n']})"
        lines.append(f"| {row['sensor']} | {row['trial']} | {row['model']} | {all_score:.4f} | "
                     f"{hole_text} | {100 * (1 - all_score / base):.1f}% |")
    lines += ["", "## Source validation choices", "",
              "| Model | Length (m) | Ridge | Validation vector RMSE |",
              "| --- | ---: | ---: | ---: |"]
    for name, selection in result["model_selection"].items():
        choice = selection["selected"]
        lines.append(f"| {name} | {choice['length_metres']} | {choice['ridge']} | "
                     f"{selection['validation_vector_rmse']:.4f} |")
    lines += ["", "## Changed-environment stress test", "",
              "Later trials contain changes to the magnetic surroundings according to the dataset authors. "
              "The models remain frozen. Their position-only inputs cannot distinguish two different source "
              "arrangements at the same position; this tests the limits of that representation, not a violation of Maxwell.", "",
              "| Instrument | Trial | Mean baseline | Linear position | Unrestricted | Planar potential |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {(r["sensor"], r["trial"], r["model"]): r for r in rows}
    for sensor in SENSORS:
        for trial in SHIFT_TRIALS:
            scores = [lookup[sensor, trial, name]["scores"]["all"]["vector_rmse"]
                      for name in ("training_mean", "linear_position", "unrestricted", "planar_potential")]
            lines.append(f"| {sensor} | {trial} | " + " | ".join(f"{v:.4f}" for v in scores) + " |")
    lines += ["", "## What this can establish", "",
              "Lower held-out error than the constant and shuffled-label controls supports reproducible "
              "spatial structure. Transfer to the phone or another instrument after frozen calibration supports "
              "a description shared across these measurement systems. The potential comparison tests whether "
              "one explicit Maxwell-inspired restriction helps this prediction task.", "",
              "A win for the potential map would not establish Maxwell uniquely; a loss would not refute Maxwell. "
              "Sensor drift, calibration, sampling, static-field assumptions, kernel family, finite hyperparameter grid, and source conditions "
              "all affect this comparison. Kernel coordinates are supplied; no electromagnetic ontology or "
              "representation equivalence has been independently discovered.", "",
              "Planar observations cannot evaluate full 3D divergence without unmeasured normal derivatives. "
              "Electric fields, charge/current channels, wave propagation, and controlled unseen source "
              "arrangements are not tested here. The potential model's near-zero interpolated curl is enforced, "
              "not an independent empirical recovery.", "",
              "See [the collection protocol](../COLLECTION_PROTOCOL.md) for the proposed raw two-phone experiment. "
              "No new physical measurements have been collected by this run.", "",
              "## Provenance and reproduction", "",
              f"Data: [AaltoML/magnetic-data]({result['data_provenance']['repository']}), "
              f"commit `{COMMIT}`. Attribution: Solin, Kok, Wahlström, Schön, and Särkkä (2018), "
              "[Modeling and interpolation of the ambient magnetic field by Gaussian processes]"
              "(https://doi.org/10.1109/TRO.2018.2830326). "
              "The authors describe the recording and preprocessing in the repository README. "
              "There is no explicit licence file in the pinned tree; attribution does not itself establish "
              "redistribution permissions.", "",
              "From the workspace root:", "", "```powershell",
              "python experiments/magnetic_mapping/fetch_data.py", "python experiments/magnetic_mapping/run_experiment.py", "```", "",
              "[results.json](results.json) includes every validation configuration, per-trial/component errors, "
              "calibration coefficients, selected row indices, spatial-coverage diagnostics, and numerical curl checks. "
              "`predictions.npz` stores sampled test observations and predictions. The test observations are "
              "uniformly spaced in time to cap each trial at the recorded sample limit, not randomly mixed into training.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for item in result["training_geometry"]:
        points = np.asarray(item["xy"])
        axes[0].scatter(points[:, 0], points[:, 1], s=5, label=f"Training trial {item['trial']}", alpha=.7)
    left, right, bottom, top = HOLE
    axes[0].add_patch(Rectangle((left, bottom), right-left, top-bottom, fill=False, color="#b54d39", lw=2))
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Measured training positions; rectangle withheld")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    names = ("training_mean", "linear_position", "shuffled_labels", "unrestricted", "planar_potential")
    colors = ("#a9afb7", "#7a6e9a", "#d4ab69", "#346fac", "#258c78")
    for k, (name, color) in enumerate(zip(names, colors)):
        scores = []
        for sensor in SENSORS:
            group = [r["scores"]["all"] for r in result["evaluations"]
                     if r["sensor"] == sensor and r["trial"] in TEST_TRIALS and r["model"] == name]
            scores.append(np.sqrt(sum(g["n"] * g["vector_rmse"]**2 for g in group) / sum(g["n"] for g in group)))
        axes[1].bar(np.arange(3) + (k-2)*.16, scores, width=.15, color=color, label=name.replace("_", " "))
    axes[1].set_xticks(np.arange(3), ["Reference IMU", "Nexus phone", "Trivisio IMU"])
    axes[1].set(ylabel="Vector RMSE (reference recording units)", title="Held-out trials 4–5, with frozen device alignment")
    axes[1].legend(fontsize=8)
    figure.suptitle("Real magnetic measurements: predictive transfer, not full Maxwell recovery", fontsize=13)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def run(data, output, train_per_trial=250, eval_per_trial=1500):
    provenance = verify_archive(data)
    recordings = {(sensor, trial): load_recording(data, sensor, trial)
                  for sensor in SENSORS for trial in range(1, 10)}
    x, y, train_audit = spatial_split(recordings, TRAIN_TRIALS, train_per_trial)
    xv, yv, validation_audit = spatial_split(recordings, VALIDATION_TRIALS, 600)
    models, selection = {}, {}
    for kind in ("unrestricted", "planar_potential"):
        print(f"Selecting {kind} on source trial 3 only...", flush=True)
        models[kind], selection[kind] = tune(kind, x, y, xv, yv)
    # Destroy position/observation alignment within each training trial, retaining
    # vector components together and preserving each trial's marginal distribution.
    shuffled = y.copy()
    rng = np.random.default_rng(20261116)
    offset = 0
    for audit in train_audit:
        size = len(audit["row_indices"])
        shuffled[offset:offset + size] = rng.permutation(y[offset:offset + size])
        offset += size
    print("Selecting shuffled-label control on the same validation set...", flush=True)
    models["shuffled_labels"], selection["shuffled_labels"] = tune("unrestricted", x, shuffled, xv, yv)
    models["linear_position"] = LinearMap().fit(x, y)
    alignments = {SOURCE: np.vstack((np.eye(3), np.zeros(3)))}
    calibration = {SOURCE: {"type": "identity_reference", "pairs": 0}}
    for sensor in SENSORS:
        if sensor != SOURCE:
            alignments[sensor], calibration[sensor] = calibrate(recordings[SOURCE, 1], recordings[sensor, 1])
    # Freeze all fitting and hyperparameter choices before opening test labels for scoring.
    mean = np.mean(y, axis=0)
    tree = cKDTree(x)
    rows, saved = [], {}
    for sensor in SENSORS:
        for trial in TEST_TRIALS + SHIFT_TRIALS:
            rec = recordings[sensor, trial]
            indices = evenly_spaced(np.arange(len(rec.x)), eval_per_trial)
            query = rec.x[indices]
            actual = apply_alignment(rec.y[indices], alignments[sensor])
            distance, _ = tree.query(query)
            masks = {"all": np.ones(len(query), dtype=bool), "spatial_hole": in_hole(query),
                     "outside_spatial_exclusion": ~in_hole(query, BUFFER),
                     "within_0.25m_of_training": distance <= .25}
            predictions = {name: model.predict(query) for name, model in models.items()}
            predictions["training_mean"] = np.tile(mean, (len(query), 1))
            prefix = f"{sensor}_{trial}"
            saved.update({f"{prefix}_x": query, f"{prefix}_actual_aligned": actual,
                          f"{prefix}_actual_raw": rec.y[indices], f"{prefix}_row_indices": rec.original_rows[indices],
                          f"{prefix}_nearest_training_distance": distance})
            for name, prediction in predictions.items():
                saved[f"{prefix}_{name}"] = prediction
                row = {"sensor": sensor, "trial": trial,
                       "environment": "stable" if trial in TEST_TRIALS else "changed",
                       "model": name,
                       "scores": {key: metrics(actual, prediction, mask) for key, mask in masks.items()},
                       "median_distance_to_training_metres": float(np.median(distance))}
                # The identity transfer is a descriptive secondary result, not used to select models.
                if sensor == "nexus":
                    row["scores_against_unadapted_phone_units"] = metrics(rec.y[indices], prediction, masks["all"])
                rows.append(row)
    result = {
        "experiment": "real_measured_cross_instrument_static_field_pilot",
        "synthetic_observations": False,
        "exact_two_phone_experiment_completed": False,
        "data_provenance": {key: value for key, value in provenance.items() if key != "files"},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("run_experiment.py", "models.py", "fetch_data.py")},
        "protocol": {"source": SOURCE, "train_trials": TRAIN_TRIALS, "validation_trials": VALIDATION_TRIALS,
                     "stable_test_trials": TEST_TRIALS, "changed_environment_trials": SHIFT_TRIALS,
                     "spatial_hole_xy_metres": HOLE, "training_validation_exclusion_buffer_metres": BUFFER,
                     "train_per_trial_cap": train_per_trial, "test_per_trial_cap": eval_per_trial,
                     "alignment_trial": 1, "alignment_pair_cap": 500, "shuffle_seed": 20261116,
                     "units": "reference Invensense numerical recording units; not independently calibrated SI"},
        "training_rows": train_audit, "validation_rows": validation_audit,
        "quality_control": [{"sensor": rec.sensor, "trial": rec.trial,
                             "accepted_rows": len(rec.x), "nonfinite_rows_removed": rec.rejected_rows.tolist()}
                            for rec in recordings.values()],
        "training_geometry": [{"trial": audit["trial"],
                               "xy": recordings[SOURCE, audit["trial"]].x[audit["row_indices"]].tolist()}
                              for audit in train_audit],
        "device_alignment": calibration, "model_selection": selection, "evaluations": rows,
        "interpolated_model_curl_rms": {name: float(np.sqrt(np.mean(planar_curl(model, xv[:200])**2)))
                                         for name, model in models.items()},
        "curl_diagnostic_scope": "Finite differences of the interpolated maps, not independent measured derivatives; "
                                 "near-zero curl of the potential map is imposed by construction.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **saved)
    write_report(result, output)
    plot_results(result, output)
    print(f"Results: {output / 'RESULTS.md'}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "aalto_magnetic")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--train-per-trial", type=int, default=250)
    parser.add_argument("--eval-per-trial", type=int, default=1500)
    args = parser.parse_args()
    if args.train_per_trial < 20 or args.eval_per_trial < 20:
        parser.error("Sample caps must be at least 20")
    with threadpool_limits(limits=1):
        run(args.data, args.output, args.train_per_trial, args.eval_per_trial)

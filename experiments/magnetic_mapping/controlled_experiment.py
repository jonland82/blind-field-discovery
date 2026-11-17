"""Plan and analyze the controlled two-phone experiment using measured CSV data.

Commands:
    python experiments/magnetic_mapping/controlled_experiment.py plan
    python experiments/magnetic_mapping/controlled_experiment.py analyze

The plan contains apparatus settings, never generated magnetic observations.
An empty or incomplete recording produces an explicit awaiting-measurements
status, not invented predictions or an experimental success report.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from models import FieldMap, LinearMap
from run_experiment import affine_fit, apply_alignment, metrics, vector_rmse

ROOT = Path(__file__).resolve().parent
POSITIONS = {"P1": (.20, .00, .10), "P2": (.05, .25, .15), "P3": (-.20, .05, .20)}
# Quaternions are w,x,y,z; R maps sensor-coordinate vectors into the fixture frame.
S = float(np.sqrt(.5))
ORIENTATIONS = {"O1": (1., 0., 0., 0.), "O2": (S, 0., 0., S),
                "O3": (S, S, 0., 0.), "O4": (S, 0., S, 0.)}
# present, source center (m), marked magnet-axis direction in the fixture frame.
SOURCES = {"absent": (0, (0., 0., 0.), (0., 0., 1.)),
           "M1": (1, (0., 0., 0.), (0., 0., 1.)),
           "M2": (1, (.05, .05, 0.), (1., 0., 0.)),
           "M3": (1, (-.05, .05, 0.), (S, 0., S))}
PLAN_FIELDS = ["trial_id", "session_id", "device_id", "split", "position_id", "orientation_id", "source_id",
               "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "magnet_present",
               "source_x_m", "source_y_m", "source_z_m", "axis_x", "axis_y", "axis_z", "duration_s"]
SAMPLE_FIELDS = ["trial_id", "timestamp_s", "u0", "u1", "u2", "unit"]


def assigned_split(session, device, position, orientation, source):
    if session == "S2":
        return "test_session"
    if device == "B":
        if source == "M3":
            return "test_source_device"
        if position == "P3":
            return "calibration"
        return "test_device"
    if source == "M3":
        return "test_source"
    if (position, orientation) == ("P1", "O3"):
        return "validation"
    if (position, orientation) == ("P2", "O4"):
        return "test_geometry"
    return "train"


def planned_trials(seed=20261116):
    rng = np.random.default_rng(seed)
    rows = []
    for session in ("S1", "S2"):
        conditions = [(p, o, s) for p in POSITIONS for o in ORIENTATIONS for s in SOURCES]
        for index in rng.permutation(len(conditions)):
            p, o, s = conditions[index]
            for device in rng.permutation(["A", "B"]):
                present, center, axis = SOURCES[s]
                row = {"trial_id": f"{session}_{device}_{p}_{o}_{s}", "session_id": session,
                       "device_id": str(device), "split": assigned_split(session, device, p, o, s),
                       "position_id": p, "orientation_id": o, "source_id": s,
                       "duration_s": 5, "magnet_present": present}
                row.update(dict(zip(("x_m", "y_m", "z_m"), POSITIONS[p])))
                row.update(dict(zip(("qw", "qx", "qy", "qz"), ORIENTATIONS[o])))
                row.update(dict(zip(("source_x_m", "source_y_m", "source_z_m"), center)))
                row.update(dict(zip(("axis_x", "axis_y", "axis_z"), axis)))
                rows.append(row)
    return rows


def make_plan(folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "plan.csv"
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=PLAN_FIELDS)
            writer.writeheader()
            writer.writerows(planned_trials())
    samples = folder / "samples.csv"
    if not samples.exists():
        with samples.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(SAMPLE_FIELDS)
    print(f"Plan: {path}; record actual measurements in {samples}")


def numeric(row, names):
    values = np.array([float(row[name]) for name in names])
    if not np.isfinite(values).all():
        raise ValueError("Non-finite numeric value in collection data")
    return values


def rotation(row):
    quaternion = numeric(row, ("qw", "qx", "qy", "qz"))
    if abs(np.linalg.norm(quaternion) - 1) > .01:
        raise ValueError(f"Non-unit fixture quaternion in {row['trial_id']}")
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def scene_features(row):
    present = float(row["magnet_present"])
    if present not in (0., 1.):
        raise ValueError("magnet_present must be zero or one")
    position = numeric(row, ("x_m", "y_m", "z_m"))
    center = numeric(row, ("source_x_m", "source_y_m", "source_z_m"))
    axis = numeric(row, ("axis_x", "axis_y", "axis_z"))
    if present and abs(np.linalg.norm(axis) - 1) > .01:
        raise ValueError("Magnet-axis direction must have unit norm")
    return np.r_[position, rotation(row).reshape(-1), present * center, present * axis, present]


def dipole_design(row):
    """Linear in uniform background (3), one common magnet strength (1), bias (3).

    This explicitly assumes a point dipole, quasistatic conditions, uniform
    background, and an ideal vector response plus constant reference-phone bias.
    It is a source/sensor-specific Maxwell-based comparator, not full Maxwell.
    """
    position = numeric(row, ("x_m", "y_m", "z_m"))
    center = numeric(row, ("source_x_m", "source_y_m", "source_z_m"))
    axis = numeric(row, ("axis_x", "axis_y", "axis_z"))
    delta = position - center
    distance = np.linalg.norm(delta)
    if distance < .05:
        raise ValueError("Planned source/sensor separation must be at least 0.05 m")
    direction = delta / distance
    world = float(row["magnet_present"]) * (3 * direction * (direction @ axis) - axis) / distance**3
    return np.column_stack((rotation(row).T, rotation(row).T @ world, np.eye(3)))


def condition(row):
    return row["position_id"], row["orientation_id"], row["source_id"]


def load_collection(folder):
    with (folder / "plan.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Plan must have unique trial IDs")
    # Lock condition identities and splits; numeric settings can be corrected to
    # measured apparatus geometry without altering the held-out assignments.
    expected = {r["trial_id"]: r for r in planned_trials()}
    if {r["trial_id"] for r in rows} != expected.keys():
        raise ValueError("Plan IDs differ from the fixed controlled protocol")
    for row in rows:
        for key in ("session_id", "device_id", "split", "position_id", "orientation_id", "source_id"):
            if row[key] != expected[row["trial_id"]][key]:
                raise ValueError(f"Modified protocol field {key} in {row['trial_id']}")
        scene_features(row)
        dipole_design(row)
    by_id = {r["trial_id"]: r for r in rows}
    measurements = {r["trial_id"]: [] for r in rows}
    units = {"A": set(), "B": set()}
    with (folder / "samples.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(SAMPLE_FIELDS) <= set(reader.fieldnames or []):
            raise ValueError("Sample CSV is missing required columns")
        for sample in reader:
            identifier = sample["trial_id"]
            if identifier not in by_id:
                raise ValueError(f"Unknown trial ID: {identifier}")
            value = numeric(sample, ("timestamp_s", "u0", "u1", "u2"))
            unit = sample["unit"].strip()
            if not unit:
                raise ValueError("Every observation must declare a unit or raw device-unit label")
            units[by_id[identifier]["device_id"]].add(unit)
            measurements[identifier].append(value)
    if any(len(unit) > 1 for unit in units.values()):
        raise ValueError("Units changed within a device's recordings")
    summaries, incomplete = {}, []
    for identifier, samples in measurements.items():
        if not samples:
            incomplete.append(identifier)
            continue
        values = np.array(samples)
        if not np.all(np.diff(values[:, 0]) > 0):
            raise ValueError(f"Non-increasing timestamps within {identifier}")
        elapsed = values[:, 0] - values[0, 0]
        trimmed = values[elapsed >= 1., 1:]
        if elapsed[-1] < 4. or len(trimmed) < 20:
            incomplete.append(identifier)
            continue
        summaries[identifier] = {"mean": trimmed.mean(axis=0), "std": trimmed.std(axis=0),
                                 "n_after_settling": len(trimmed), "span_seconds": float(elapsed[-1])}
    return rows, summaries, incomplete, {k: sorted(v) for k, v in units.items()}


def analyze(folder, output):
    rows, observations, incomplete, units = load_collection(folder)
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "awaiting_measurements" if incomplete else "complete_collection",
              "expected_trials": len(rows), "complete_trials": len(observations),
              "incomplete_trial_ids": incomplete, "device_units": units,
              "synthetic_measurements_created": False}
    (output / "collection_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if incomplete:
        print(f"Awaiting real measurements: {len(observations)}/{len(rows)} trials complete. "
              f"See {output / 'collection_status.json'}")
        return status
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    lookup = {condition(r): r for r in train}
    calibration = [r for r in rows if r["split"] == "calibration"]
    target_values, reference_values = [], []
    for row in calibration:
        if condition(row) not in lookup:
            raise ValueError("Target calibration condition lacks a source training pair")
        reference_values.append(observations[lookup[condition(row)]["trial_id"]]["mean"])
        target_values.append(observations[row["trial_id"]]["mean"])
    alignment, condition_number = affine_fit(np.asarray(target_values), np.asarray(reference_values), minimum_pairs=12)
    x = np.array([scene_features(r) for r in train])
    y = np.array([observations[r["trial_id"]]["mean"] for r in train])
    xv = np.array([scene_features(r) for r in valid])
    yv = np.array([observations[r["trial_id"]]["mean"] for r in valid])
    location, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1.
    x = (x - location) / scale
    xv = (xv - location) / scale
    selection, fitted = {}, {}
    shuffled = np.random.default_rng(20261116).permutation(y)
    for name, labels in (("unrestricted", y), ("shuffled_labels", shuffled)):
        grid, best = [], None
        for length in (.5, 1., 2., 4.):
            for ridge in (.0001, .001, .01, .1):
                model = FieldMap("unrestricted", length, ridge).fit(x, labels)
                error = vector_rmse(yv, model.predict(xv))
                grid.append({"length": length, "ridge": ridge, "validation_vector_rmse": error})
                if best is None or error < best[0]:
                    best = error, model, grid[-1]
        fitted[name] = best[1]
        selection[name] = {"chosen": best[2], "grid": grid}
    linear_trials, best = [], None
    for ridge in (.001, .01, .1, 1., 10.):
        model = LinearMap(ridge).fit(x, y)
        error = vector_rmse(yv, model.predict(xv))
        linear_trials.append({"ridge": ridge, "validation_vector_rmse": error})
        if best is None or error < best[0]:
            best = error, model, linear_trials[-1]
    fitted["linear_geometry"] = best[1]
    selection["linear_geometry"] = {"chosen": best[2], "grid": linear_trials}
    design = np.concatenate([dipole_design(r) for r in train])
    coefficients, _, rank, singular = np.linalg.lstsq(design, y.reshape(-1), rcond=None)
    if rank != 7:
        raise ValueError("Apparatus geometry cannot identify the 7-parameter dipole/background/bias model")
    tests = [r for r in rows if r["split"].startswith("test_")]
    xt = (np.array([scene_features(r) for r in tests]) - location) / scale
    actual = np.array([observations[r["trial_id"]]["mean"] for r in tests])
    for i, row in enumerate(tests):
        if row["device_id"] == "B":
            actual[i] = apply_alignment(actual[i:i+1], alignment)[0]
    predictions = {name: model.predict(xt) for name, model in fitted.items()}
    predictions["training_mean"] = np.tile(y.mean(axis=0), (len(tests), 1))
    predictions["dipole_background_bias"] = np.array([dipole_design(r) @ coefficients for r in tests])
    evaluation = []
    for device in ("A", "B"):
        for split in sorted({r["split"] for r in tests}):
            mask = np.array([r["device_id"] == device and r["split"] == split for r in tests])
            if mask.any():
                for name, prediction in predictions.items():
                    evaluation.append({"device": device, "split": split, "model": name,
                                       **metrics(actual, prediction, mask)})
    result = {**status, "status": "analyzed_measured_collection", "model_selection": selection,
              "evaluations": evaluation, "training_trial_ids": [r["trial_id"] for r in train],
              "validation_trial_ids": [r["trial_id"] for r in valid],
              "test_trial_ids": [r["trial_id"] for r in tests],
              "device_B_alignment": {"independent_trial_means": len(calibration),
                                     "target_to_A_affine": alignment.tolist(), "condition_number": condition_number},
              "dipole": {"parameters": coefficients.tolist(), "design_condition_number": float(singular[0]/singular[-1]),
                          "scope": "one shared point-dipole strength, uniform background, ideal vector response plus fixed bias"},
              "trial_summaries": {key: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
                                  for key, value in observations.items()},
              "interpretation": "Measured prediction comparison; neither latent ontology nor full Maxwell recovery is established."}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", actual=actual, trial_ids=[r["trial_id"] for r in tests], **predictions)
    lines = ["# Controlled two-phone results", "", "Real measured trial means; errors are in phone A's declared recording units.", "",
             "| Device | Held-out split | Model | Trials | Vector RMSE |", "| --- | --- | --- | ---: | ---: |"]
    for row in evaluation:
        lines.append(f"| {row['device']} | {row['split']} | {row['model']} | {row['n']} | {row['vector_rmse']:.6g} |")
    lines += ["", "Phone B uses a frozen affine alignment from 12 calibration arrangements paired with phone A training trials. "
              "No test trials select models or alignment. The Maxwell-based comparator additionally assumes a point magnet, "
              "uniform background, and an ideal vector sensor plus fixed bias. A prediction advantage supports that complete "
              "description on this task; it does not identify a unique fundamental theory.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Controlled experiment results: {output / 'RESULTS.md'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "analyze"))
    parser.add_argument("--collection", type=Path, default=ROOT / "collection")
    parser.add_argument("--output", type=Path, default=ROOT / "controlled_results")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == "plan":
            make_plan(args.collection)
        else:
            analyze(args.collection, args.output)

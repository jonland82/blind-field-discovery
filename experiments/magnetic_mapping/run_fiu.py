"""Public-only test of representations across independently tracked orientations.

Uses real FIU MARG measurements. No generated fields, phone-compass orientations,
or held-out recording labels enter fitting or hyperparameter selection.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from fetch_fiu import COMMIT, REPOSITORY
from fiu_models import PoseMap, WorldMap
from run_experiment import evenly_spaced, metrics, vector_rmse

ROOT = Path(__file__).resolve().parent
TRAIN = tuple(range(1, 11))
VALID = tuple(range(11, 16))
TEST = tuple(range(16, 31))
LENGTHS = (.02, .06, .18)
RIDGES = (.001, .01, .1)
ORIENTATION_LENGTHS = (.5, 1.5, 4.)
CONVENTIONS = ("as_recorded", "transposed")


@dataclass
class Recording:
    number: int
    x: np.ndarray
    rotation: np.ndarray
    y: np.ndarray
    t: np.ndarray
    original_rows: np.ndarray
    qc: dict


def verify(data):
    manifest = json.loads((data / "provenance.json").read_text())
    if manifest["commit"] != COMMIT:
        raise ValueError("Unexpected FIU data version")
    entries = {r["path"]: r for r in manifest["files"]}
    if not {f"recordings/rec{i:02d}.csv" for i in range(1, 31)} <= entries.keys():
        raise ValueError("Incomplete FIU provenance manifest")
    for name, entry in entries.items():
        path = data / name
        if not path.resolve().is_relative_to(data.resolve()):
            raise ValueError("Provenance path escapes the data folder")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Data hash mismatch: {name}")
    return manifest


def load_recording(data, number):
    # The CSV has one header row. The upstream MATLAB reader skips two rows,
    # dropping its first observation; this loader preserves that observation.
    raw = np.genfromtxt(data / "recordings" / f"rec{number:02d}.csv", delimiter=",", skip_header=1)
    if raw.ndim != 2 or raw.shape[1] != 23:
        raise ValueError(f"Unexpected FIU CSV layout in recording {number}")
    if not np.isfinite(raw[:, 0]).all() or not np.all(np.diff(raw[:, 0]) > 0):
        raise ValueError(f"Invalid timestamps in recording {number}")
    quaternion = raw[:, 4:8]  # Optical x,y,z,w. Onboard filter columns 8:12 are unused.
    norm = np.linalg.norm(quaternion, axis=1)
    finite = np.isfinite(raw[:, 1:8]).all(axis=1) & np.isfinite(raw[:, 18:21]).all(axis=1)
    tracked = raw[:, 22] == 1
    valid_quaternion = abs(norm - 1.) <= .01
    accepted = finite & tracked & valid_quaternion
    indices = np.flatnonzero(accepted)
    if len(indices) < 100:
        raise ValueError(f"Insufficient independently tracked observations in recording {number}")
    return Recording(number, raw[indices, 1:4], Rotation.from_quat(quaternion[indices]).as_matrix(),
                     raw[indices, 18:21], raw[indices, 0] / 1000., indices,
                     {"recording": number, "total_rows": len(raw), "accepted_rows": len(indices),
                      "untracked_rows": int((~tracked).sum()), "invalid_quaternion_rows": int((~valid_quaternion).sum()),
                      "nonfinite_relevant_rows": int((~finite).sum())})


def sample_recording(recording, maximum):
    indices = evenly_spaced(np.arange(len(recording.x)), maximum)
    return (recording.x[indices], recording.rotation[indices], recording.y[indices],
            {"recording": recording.number, "original_row_indices": recording.original_rows[indices].tolist()})


def combine(recordings, maximum):
    blocks = [sample_recording(rec, maximum) for rec in recordings]
    return tuple(np.concatenate([block[k] for block in blocks]) for k in range(3)) + ([block[3] for block in blocks],)


def choose_model(kind, train, valid):
    x, rotations, y = train
    xv, rv, yv = valid
    configurations = []
    if kind in ("pose_learner", "shuffled_pose"):
        for length in LENGTHS:
            for orientation_length in ORIENTATION_LENGTHS:
                for ridge in RIDGES:
                    configurations.append({"length_metres": length, "orientation_length": orientation_length, "ridge": ridge})
    else:
        for convention in CONVENTIONS:
            for length in LENGTHS:
                for ridge in RIDGES:
                    configurations.append({"length_metres": length, "ridge": ridge, "convention": convention})
    search, best = [], None
    for settings in configurations:
        if kind in ("pose_learner", "shuffled_pose"):
            model = PoseMap(settings["length_metres"], settings["orientation_length"], settings["ridge"])
        else:
            model = WorldMap("vector" if kind == "shared_vector" else "potential", settings["length_metres"],
                             settings["ridge"], settings["convention"])
        model.fit(x, rotations, y)
        error = vector_rmse(yv, model.predict(xv, rv))
        entry = {**settings, "validation_vector_rmse_gauss": error}
        search.append(entry)
        if best is None or error < best[0]:
            best = error, model, entry
    return best[1], {"selected": best[2], "grid": search}


def pooled(evaluations, name):
    rows = [r for r in evaluations if r["model"] == name]
    return float(np.sqrt(sum(r["n"] * r["vector_rmse"]**2 for r in rows) / sum(r["n"] for r in rows)))


def write_report(result, output):
    scores = result["pooled_test_vector_rmse_gauss"]
    lines = ["# Public-data experiment: does a shared field explain changing observations?", "",
             "**Completed using 30 public measured recordings. No new hardware or synthetic observations are required.**", "",
             "The FIU MARG dataset records a magnetic sensor's readings, positions, and orientations obtained from "
             "an independent optical system. We compare descriptions that make different assumptions about how "
             "a common field appears through a rotating instrument. This extends the earlier "
             "[multi-instrument phone pilot](../results/RESULTS.md); it uses a different dataset and a single MARG sensor, not two phones.", "",
             "## Result", "",
             "| Description | Pooled held-out vector RMSE (gauss) | Improvement over constant |",
             "| --- | ---: | ---: |"]
    for name in ("constant", "shuffled_pose", "pose_learner", "shared_vector", "static_potential"):
        value = scores[name]
        lines.append(f"| {name} | {value:.6f} | {100*(1-value/scores['constant']):.1f}% |")
    best = min(("pose_learner", "shared_vector", "static_potential"), key=scores.get)
    medians = {name: float(np.median([r["vector_rmse"] for r in result["evaluations"] if r["model"] == name]))
               for name in scores}
    lines += ["", f"The lowest pooled test error among the three substantive descriptions is **{best}**. "
              "This is a prediction result under the stated observation, calibration, stationarity, and kernel assumptions. "
              "It is not independent discovery or rejection of Maxwell's equations.", "",
              f"Median per-recording RMSE is {medians['pose_learner']:.6f} gauss for the pose learner, "
              f"{medians['shared_vector']:.6f} for the shared-vector model, and {medians['static_potential']:.6f} "
              "for the static-potential model. Recording 20 has much larger errors for every description and "
              "strongly affects the pooled metric. It is retained in every comparison; the table below shows "
              "the complete per-recording results.", "",
              "## What each description assumes", "",
              "- **pose_learner:** learns three separate functions from supplied optical position and orientation "
              "to sensor readings. It is not told that its three outputs form a rotating vector.",
              "- **shared_vector:** learns a three-component field over optical position. The model explicitly "
              "assumes a vector transformation between that field and the sensor's readings, plus a fixed sensor bias.",
              "- **static_potential:** uses the same observation transformation and learns a scalar-potential field, "
              "adding a curl-free constraint consistent with a locally current-free quasistatic region.",
              "- **constant:** predicts the training-channel mean.",
              "- **shuffled_pose:** fits and selects the same kind of model as pose_learner after measured vectors "
              "are shuffled within each training recording, destroying their alignment with pose.", "",
              "The shared-vector observation equation is", "", "$$",
              r"\mathbf b_i=R_i^{\mathsf T}\mathbf B(\mathbf x_i)+\mathbf c+\boldsymbol\eta_i.", "$$", "",
              "The potential model additionally uses", "", "$$",
              r"\mathbf B(\mathbf x)=-\nabla\psi(\mathbf x),\qquad\nabla\times\mathbf B=0.", "$$", "",
              "These are explicit candidate representations. A good result for a shared-vector description would "
              "support its usefulness across these recordings; it would not show that the vector transformation "
              "was discovered without assumptions. The unconstrained learner's outputs remain in the instrument frame.", "",
              "## Split and controls", "",
              "- Recordings 01–10 train the models; 11–15 select settings; 16–30 supply held-out scores.",
              f"- At most {result['protocol']['train_per_recording']} equally spaced accepted observations per training recording "
              f"and {result['protocol']['validation_per_recording']} per validation recording are used. "
              f"Each test recording contributes at most {result['protocol']['test_per_recording']} accepted observations.",
              "- Missing optical tracks and invalid quaternions are excluded by fixed rules. Time gaps are not "
              "interpolated. The camera quaternion columns are used; onboard magnetically informed orientation estimates are not.",
              "- Quaternion matrices and their transposes are two declared coordinate-convention candidates "
              "for the shared-vector/potential maps. Validation chooses between them. This is convention selection, "
              "not discovery of a new physical transformation.",
              "- A common background vector and fixed sensor bias are fitted to training observations only. "
              "Each spatial field is learned from the residuals. No test-recording calibration or magnetic labels "
              "are used for fitting or selection.",
              "- Whole recordings are held out. These are repeated observations in one laboratory; they are not "
              "15 independent laboratories or proof of new-device transfer.", "",
              "## Per-recording held-out errors", "",
              "All errors below are vector RMSE in gauss. Each row is one held-out recording.", "",
              "| Recording | Constant | Shuffled | Pose learner | Shared vector | Static potential |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lookup = {(r["recording"], r["model"]): r for r in result["evaluations"]}
    for number in TEST:
        values = [lookup[number, name]["vector_rmse"] for name in
                  ("constant", "shuffled_pose", "pose_learner", "shared_vector", "static_potential")]
        lines.append(f"| {number:02d} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "## Selected settings", "", "| Model | Validation-selected configuration |", "| --- | --- |"]
    for name, selection in result["model_selection"].items():
        lines.append(f"| {name} | `{json.dumps(selection['selected'], sort_keys=True)}` |")
    lines += ["", "## What this gets us toward a theory", "",
              "The bridge tested here is whether one field description can explain how observations change with "
              "independently measured pose. The unconstrained model can learn pose-dependent regularities; the "
              "shared-vector and potential descriptions compress those regularities using stronger structure. "
              "Prediction on new recordings evaluates the usefulness of that structure.", "",
              "The experiment does not learn a unique latent ontology, establish electromagnetic constants, "
              "observe electric fields or charge/current channels, or recover the complete Maxwell system. "
              "No divergence constraint is imposed. Although positions have three coordinates, the observations "
              "follow a restricted manipulation protocol rather than a dense 3D field survey, so full spatial-law "
              "identification should not be inferred from the potential-model score.", "",
              "Failures may reflect sensor response, optical/sensor alignment, marker-to-sensor displacement, "
              "time synchronization, nonstationarity, or the finite feature/hyperparameter choices. We do not "
              "select away those failures based on held-out magnetic errors. Optical coordinates and a smooth "
              "kernel are supplied assumptions in every substantive description. Agreement among models is "
              "not itself proof of a unique Platonic representation.", "",
              "## Evidence and reproduction", "",
              f"Data: [FIUMARGDB]({REPOSITORY}), commit `{COMMIT}`. Retained source documentation and LICENSE "
              "accompany the local download. Cite Sonchan and colleagues (2023), "
              "[Benchmarking Dataset of Signals from a Commercial MEMS MARG Sensor]"
              "(https://doi.org/10.3390/s23083786). The paper describes the optical reference and magnetic data; "
              "its magnetic readings are in gauss and positions in metres.", "",
              "From the workspace root:", "", "```powershell", "python experiments/magnetic_mapping/fetch_fiu.py",
              "python experiments/magnetic_mapping/run_fiu.py", "```", "",
              "[results.json](results.json) includes every selected setting, grid score, original row index, "
              "data-quality count, bias/background fit, software version, and provenance reference. "
              "`predictions.npz` preserves held-out observations, optical poses, and every prediction. "
              "The original phone pilot is preserved in `experiments/magnetic_mapping/results`.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ("constant", "shuffled_pose", "pose_learner", "shared_vector", "static_potential")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ("#9ca4ac", "#bd925c", "#346fac", "#258c78", "#8062a3")
    axes[0].bar(np.arange(len(names)), [result["pooled_test_vector_rmse_gauss"][name] for name in names], color=colors)
    axes[0].set_xticks(np.arange(len(names)), [name.replace("_", "\n") for name in names], fontsize=9)
    axes[0].set(ylabel="Vector RMSE (gauss)", title="Prediction on 15 held-out recordings")
    for name, color in zip(names[2:], colors[2:]):
        rows = [r for r in result["evaluations"] if r["model"] == name]
        axes[1].plot([r["recording"] for r in rows], [r["vector_rmse"] for r in rows],
                     marker=".", color=color, label=name.replace("_", " "))
    axes[1].set(xlabel="Held-out recording", ylabel="Vector RMSE (gauss)", title="Whole recordings remain separate tests")
    axes[1].legend(fontsize=8)
    figure.suptitle("Public magnetic observations: testing a shared field representation", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def run(data, output, train_per_recording=50, test_per_recording=400):
    manifest = verify(data)
    # Load fitting records first; test recordings are loaded only after all choices are frozen.
    training = [load_recording(data, number) for number in TRAIN]
    validation = [load_recording(data, number) for number in VALID]
    x, rotations, y, train_audit = combine(training, train_per_recording)
    xv, rv, yv, valid_audit = combine(validation, 100)
    fitted, selections = {}, {}
    for name in ("pose_learner", "shared_vector", "static_potential"):
        print(f"Selecting {name} using recordings 11–15 only...", flush=True)
        fitted[name], selections[name] = choose_model(name, (x, rotations, y), (xv, rv, yv))
    shuffled = y.copy()
    rng = np.random.default_rng(20261116)
    offset = 0
    for block in train_audit:
        n = len(block["original_row_indices"])
        shuffled[offset:offset+n] = rng.permutation(y[offset:offset+n])
        offset += n
    print("Selecting the shuffled-label control...", flush=True)
    fitted["shuffled_pose"], selections["shuffled_pose"] = choose_model(
        "shuffled_pose", (x, rotations, shuffled), (xv, rv, yv))
    # A deliberately simple baseline: no use of optical information.
    mean = y.mean(axis=0)
    evaluations, archive, test_audit, qc = [], {}, [], [rec.qc for rec in training + validation]
    for number in TEST:
        rec = load_recording(data, number)
        query, orientation, actual, audit = sample_recording(rec, test_per_recording)
        qc.append(rec.qc)
        test_audit.append(audit)
        prefix = f"rec{number:02d}"
        archive.update({f"{prefix}_x": query, f"{prefix}_optical_rotation": orientation,
                        f"{prefix}_actual": actual, f"{prefix}_original_row_indices": np.array(audit["original_row_indices"])})
        predictions = {name: model.predict(query, orientation) for name, model in fitted.items()}
        predictions["constant"] = np.tile(mean, (len(query), 1))
        for name, prediction in predictions.items():
            evaluations.append({"recording": number, "model": name,
                                **metrics(actual, prediction, np.ones(len(query), dtype=bool))})
            archive[f"{prefix}_{name}"] = prediction
    result = {"experiment": "public_optically_tracked_representation_transfer", "synthetic_observations": False,
              "requires_new_measurements": False, "data_provenance": {k: v for k, v in manifest.items() if k != "files"},
              "protocol": {"training_recordings": TRAIN, "validation_recordings": VALID, "test_recordings": TEST,
                           "train_per_recording": train_per_recording, "validation_per_recording": 100,
                           "test_per_recording": test_per_recording, "orientation_source": "optical camera quaternion columns 4:8",
                           "onboard_magnetic_orientation_used": False, "units": "magnetic gauss, position metres",
                           "shuffle_seed": 20261116, "max_quaternion_norm_error": .01},
              "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in ("run_fiu.py", "fiu_models.py", "fetch_fiu.py", "models.py")},
              "training_rows": train_audit, "validation_rows": valid_audit, "test_rows": test_audit,
              "model_selection": selections, "quality_control": qc, "evaluations": evaluations,
              "pooled_test_vector_rmse_gauss": {name: pooled(evaluations, name) for name in
                                                ("constant", "shuffled_pose", "pose_learner", "shared_vector", "static_potential")},
              "world_model_nuisance_fits": {name: {"background": model.background.tolist(), "sensor_bias": model.bias.tolist()}
                                            for name, model in fitted.items() if isinstance(model, WorldMap)}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **archive)
    write_report(result, output)
    plot(result, output)
    print(json.dumps(result["pooled_test_vector_rmse_gauss"], indent=2), flush=True)
    print(f"Results: {output / 'RESULTS.md'}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "fiu_marg")
    parser.add_argument("--output", type=Path, default=ROOT / "public_transfer")
    parser.add_argument("--train-per-recording", type=int, default=50)
    parser.add_argument("--test-per-recording", type=int, default=400)
    args = parser.parse_args()
    if args.train_per_recording < 10 or args.test_per_recording < 20:
        parser.error("Use at least 10 training and 20 test samples per recording")
    with threadpool_limits(limits=1):
        run(args.data, args.output, args.train_per_recording, args.test_per_recording)

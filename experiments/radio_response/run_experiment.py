"""Fit and test a radio response kernel using public measured CSI only.

The learned target is Q_ar(f, x)=H_a(f,x)/H_r(f,x), a relative port-response
operator. It is not the absolute free-space retarded Green function.
"""
import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from records import read_record, tensor
from kernel_models import KernelMap, fit_constant, fit_green_gain, green_quotient, normalized_error

ROOT = Path(__file__).resolve().parent
LENGTHS = (.25, .5, 1., 2., 4., 8.)
RIDGES = (.001, .01, .1)
BINS = np.arange(0, 1024, 16)
MODELS = ("zero_filter", "constant_filter", "shuffled_geometry", "learned_filter", "geometric_green", "green_envelope")
HOLE_RADIUS = 2.0
BUFFER = .5


@dataclass
class Snapshots:
    name: str
    x: np.ndarray
    csi: np.ndarray
    snr: np.ndarray
    age: np.ndarray
    time: np.ndarray
    indices: np.ndarray


def load_group(data, manifest, name):
    entries = sorted((r for r in manifest["records"] if r["file"] == name), key=lambda r: r["record_index"])
    expected = manifest["files"][name]["selected_record_indices"]
    if [r["record_index"] for r in entries] != expected:
        raise ValueError(f"Missing sampled records for {name}")
    x, csi, snr, age, times = [], [], [], [], []
    for entry in entries:
        path = data / entry["path"]
        if not path.resolve().is_relative_to(data.resolve()):
            raise ValueError("Record path escapes the data directory")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Record hash mismatch: {path}")
        record = read_record(content)
        values = tensor(record["csi"])
        if values.shape != (64, 1024, 2):
            raise ValueError("Unexpected CSI tensor shape")
        csi.append((values[:, BINS, 0]+1j*values[:, BINS, 1]).astype(np.complex128))
        x.append(tensor(record["pos-tachy"]))
        snr.append(tensor(record["snr"]))
        age.append(record["gt-interp-age-tachy"])
        times.append(record["time"])
    return Snapshots(name, np.asarray(x), np.asarray(csi), np.asarray(snr), np.asarray(age),
                     np.asarray(times), np.asarray(expected))


def hole(x, buffer=0.):
    return np.linalg.norm(x[:, :2], axis=1) <= HOLE_RADIUS+buffer


def base_quality(group):
    return (np.isfinite(group.x).all(axis=1) & np.isfinite(group.csi).all(axis=(1, 2))
            & np.isfinite(group.snr).all(axis=1) & np.isfinite(group.age) & (abs(group.age) <= .1))


def receiver_quality(group, reference_port, floor):
    reference = group.csi[:, reference_port]
    return (base_quality(group) & (group.snr[:, reference_port] > 0)
            & np.all(abs(reference) > floor[None, :], axis=1))


def score(actual, reference, prediction, mask):
    if not mask.any():
        return {"n": 0, "transfer_nmse": None, "relative_filter_nmse": None}
    predicted_response = prediction[mask] * reference[mask, None, :]
    actual_quotient = actual[mask] / reference[mask, None, :]
    per_snapshot = np.sum(abs(actual[mask]-predicted_response)**2, axis=(1, 2)) / np.sum(abs(actual[mask])**2, axis=(1, 2))
    per_receiver = np.sum(abs(actual[mask]-predicted_response)**2, axis=(0, 2)) / np.sum(abs(actual[mask])**2, axis=(0, 2))
    return {"n": int(mask.sum()), "transfer_nmse": normalized_error(actual[mask], predicted_response),
            "relative_filter_nmse": normalized_error(actual_quotient, prediction[mask]),
            "median_snapshot_transfer_nmse": float(np.median(per_snapshot)),
            "per_receiver_transfer_nmse": per_receiver.tolist()}


def tune_kernel(x, target, reference, xv, reference_v, actual_v, propagation=None):
    best, search = None, []
    for length, ridge in itertools.product(LENGTHS, RIDGES):
        model = KernelMap(length, ridge).fit(x, target, reference)
        prediction = model.predict(xv)
        if propagation is not None:
            prediction *= propagation
        error = normalized_error(actual_v, prediction*reference_v[:, None, :])
        entry = {"length_metres": length, "ridge": ridge, "validation_transfer_nmse": error}
        search.append(entry)
        if best is None or error < best[0]:
            best = error, model, entry
    return best[1], {"selected": best[2], "search": search}


def tune_green(x, reference, actual, xv, reference_v, actual_v, array, reference_port, ports, frequencies):
    best, search = None, []
    for horizontal, vertical, phase in itertools.product((1, -1), repeat=3):
        configuration = {"horizontal_sign": horizontal, "vertical_sign": vertical, "phase_sign": phase}
        propagation = green_quotient(x, array, reference_port, ports, frequencies, configuration)
        gain = fit_green_gain(propagation, reference, actual)
        validation = green_quotient(xv, array, reference_port, ports, frequencies, configuration) * gain[None, :, :]
        error = normalized_error(actual_v, validation*reference_v[:, None, :])
        entry = {**configuration, "validation_transfer_nmse": error}
        search.append(entry)
        if best is None or error < best[0]:
            best = error, configuration, gain, entry
    return best[1], best[2], {"selected": best[3], "search": search}


def write_report(result, output):
    main = {(r["array"], r["model"]): r["scores"]["all"]["transfer_nmse"]
            for r in result["evaluations"] if r["file"] == "dc04"}
    north_gain = 100 * (1-main[0, "green_envelope"]/main[0, "constant_filter"])
    south_gain = 100 * (1-main[1, "geometric_green"]/main[1, "constant_filter"])
    spatial_counts = [r["scores"]["spatial_hole"]["n"] for r in result["evaluations"]
                      if r["file"] == "dc04" and r["model"] == "zero_filter"]
    lines = ["# Measured radio response kernels: held-out experiment", "",
             "**This experiment uses measured DICHASUS CSI. No simulated propagation data are used as labels.**", "",
             "The target is a relative response between receiver antennas, not the transmitter-to-field Green function. "
             "The transmitter's absolute time and phase reference is unavailable; the common ambiguity cancels in the ratio. "
             "This is the experimentally identifiable operator we can test with these data.", "",
             "Read [the theory and protocol](../THEORY_AND_EXPERIMENT.md) for definitions, assumptions, and the connection to Maxwell.", "",
             "## Observed takeaway", "",
             f"In the north array, the propagation-plus-learning model has test NMSE **{main[0, 'green_envelope']:.3f}**, "
             f"compared with **{main[0, 'learned_filter']:.3f}** for the generic learner and "
             f"**{main[0, 'constant_filter']:.3f}** for a constant filter. In the south array, the geometric model "
             f"has the lowest error, **{main[1, 'geometric_green']:.3f}**, compared with "
             f"**{main[1, 'learned_filter']:.3f}** for the generic learner and "
             f"**{main[1, 'constant_filter']:.3f}** for the constant filter.", "",
             f"These are modest improvements of {north_gain:.1f}% and {south_gain:.1f}% in squared error over "
             "the constant-filter baseline. Substantial residual error remains. The added geometric propagation "
             "structure helps this task; it does not make the predictions complete or identify a fundamental law. "
             "In the changed environment, every nonzero fitted model is worse than the zero-output baseline. "
             f"Spatial-hole scores have {spatial_counts[0]} and {spatial_counts[1]} accepted snapshots in the two arrays, "
             "too few here for a strong spatial-transfer conclusion.", "",
             "## Main result: new recording, unchanged hallway", "",
             "Models train on sampled snapshots from dc01, select settings on dc02, and test on dc04. "
             "A two-metre-radius disk around the coordinate origin is excluded from training and validation, "
             "with an extra 0.5 m buffer. The disk is a spatial holdout, not a different room.", "",
             "The primary score is squared complex prediction error divided by measured response energy. "
             "Zero is perfect; the zero-output filter scores one. The input at test time is position and the "
             "measured reference-antenna response. The other antennas' responses are held-out targets.", "",
             "| Array | Model | Test snapshots | Transfer NMSE | Spatial-hole snapshots | Spatial-hole NMSE |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["evaluations"]:
        if row["file"] != "dc04":
            continue
        all_scores, spatial = row["scores"]["all"], row["scores"]["spatial_hole"]
        spatial_text = "n/a" if spatial["n"] == 0 else f"{spatial['transfer_nmse']:.6f}"
        lines.append(f"| {row['array']} | {row['model']} | {all_scores['n']} | {all_scores['transfer_nmse']:.6f} "
                     f"| {spatial['n']} | {spatial_text} |")
    lines += ["", "## Different-environment stress test", "",
              "The frozen models also predict dc13, where a metal obstruction changes the source environment. "
              "No fitting or convention selection uses its labels. Position alone cannot specify that changed "
              "environment, so failure here is a limit of the supplied representation.", "",
              "| Array | Model | Snapshots | Transfer NMSE |", "| --- | --- | ---: | ---: |"]
    for row in result["evaluations"]:
        if row["file"] == "dc13":
            values = row["scores"]["all"]
            lines.append(f"| {row['array']} | {row['model']} | {values['n']} | {values['transfer_nmse']:.6f} |")
    lines += ["", "## Models compared", "",
              "- **zero_filter:** predicts no received output, providing an NMSE reference of one.",
              "- **constant_filter:** a measured least-squares complex coupling per receiver and frequency, independent of position.",
              "- **shuffled_geometry:** the same learner and selection grid as learned_filter, with measured quotient vectors "
              "shuffled across training positions within each array task.",
              "- **learned_filter:** smooth kernel regression from position to the complex relative response; it contains no "
              "propagation-speed or Maxwell constraint.",
              "- **geometric_green:** a direct-path free-space Green-function ratio, with constant receiver/frequency gains "
              "fitted to training observations. It supplies the known propagation speed and uses published antenna geometry.",
              "- **green_envelope:** the same learned regression applied after factoring out geometric propagation, "
              "testing a combined physics/learning representation.", "",
              "Every model is evaluated on the same accepted snapshots for its array. Discrete antenna row/column "
              "axis signs and complex phase sign are selected on validation data only. They calibrate a coordinate/IQ "
              "convention, not a newly measured physical law. Models predict the same receiver ports used during training; "
              "this experiment does not claim unseen-instrument transfer.", "",
              "## Data audit", "",
              f"The experiment downloads {len(result['data_provenance']['records'])} whole TFRecords from four public "
              "files using fixed evenly spaced record indices and bounded HTTP byte ranges. Each selected record has "
              "its framing, CRC32C and local SHA-256 checked. The full multi-gigabyte files are not downloaded, so their "
              "published full-file MD5 values are recorded but not claimed as verified.", "",
              "Only 64 equally spaced frequency bins from each 1024-bin measured spectrum are modeled. "
              "The native bin ordering follows the published measurement-processing code: applying fftshift yields "
              "the centered frequency ordering. The nominal carrier is 3.438 GHz and bandwidth 50 MHz.", "",
              "Reference antennas are selected by median training SNR within each array. Fixed quality rules require "
              "finite relevant data, a reference SNR statistic above zero, position interpolation age within 0.1 s, "
              "and non-negligible reference amplitude in every selected bin. The amplitude floor is 2% of the "
              "per-bin training reference RMS; no validation/test labels set it. Exclusions and retained original "
              "indices are listed in results.json. No snapshot is removed because a model predicts it poorly.", "",
              "Per-port, per-frequency gain fits use only training responses. The spec warns that offsets may "
              "be accurate within arrays but not between arrays, so each relative-response task stays within "
              "one physical receive array. No cross-array phase coherence is assumed.", "",
              "## What the result means", "",
              "Lower transfer error demonstrates prediction of a measured radio response operator at held-out "
              "positions/recordings. A benefit from green_envelope would support the usefulness of geometric "
              "propagation as a representation for this task. A good learned_filter alone demonstrates empirical "
              "regularity; it does not uniquely identify its physical explanation.", "",
              "The direct-path comparator is one limited Maxwell-based approximation. It omits full reflections, "
              "diffraction, scattering, antenna patterns, coupling and a complete boundary-value solution. Its "
              "success or failure does not settle Maxwell versus ML in general. The geometric propagation speed "
              "is supplied, not discovered. Frequency-dependent constant electronic responses are learned as nuisance parameters.", "",
              "The relative operator can have negative delay because the reference receiver may see a wave later "
              "than another receiver. Its finite-band IFFT is periodic and only a representation of measured relative "
              "response. Therefore it cannot independently establish absolute causality, a light cone, physical "
              "travel time, or light speed. No new transmitted waveforms or source currents are measured here. "
              "The experiment evaluates the kernel through the actual common sounding response, not invented "
              "test waveforms. See the theory document for the distinction between a causal field Green function "
              "and this relative antenna filter.", "",
              "## Reproduce and inspect", "", "```powershell", "python experiments/radio_response/fetch_data.py",
              "python experiments/radio_response/run_experiment.py", "python -m pytest experiments/radio_response/tests -q", "```", "",
              "[results.json](results.json) contains the split audit, quality counts, training-selected reference "
              "ports and amplitude floors, all validation searches, per-receiver errors, spatial coverage and "
              "source hashes. `predictions.npz` contains measured reference/target responses and the predicted "
              "complex relative filters. `comparison.png` shows the test results. No earlier phone or optical "
              "study's outputs are overwritten.", "",
              "[Relative impulse-response examples](relative_impulse_responses.png) show the fixed first accepted "
              "test snapshot and first target port for each array. `impulse_examples.npz` stores those curves. "
              "They illustrate the modeled relative filter; they are not additional independent validation.", "",
              "Data attribution: Euchner, Stephan, and Gauger (2023), "
              "[CSI Dataset dichasus-dcxx: Institute hallway with known 3D model, distributed antennas]"
              "(https://doi.org/10.18419/DARUS-3831), DaRUS version 1.0, CC BY 4.0. "
              "The retained `spec.json` provides the source conventions and warnings. "
              "Frequency-order interpretation is also supported by "
              "[the authors' measurement-processing code](https://github.com/NVlabs/diff-rt-calibration/blob/main/code/utils.py).", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def save_impulse_examples(saved, bandwidth, output):
    """Fixed first accepted test snapshot/first output port, never best-case selected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lags = (np.arange(len(BINS))-len(BINS)//2) / bandwidth
    archive = {"relative_lags_seconds": lags}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for array_index, axis in enumerate(axes):
        prefix = f"dc04_array{array_index}"
        measured = saved[f"{prefix}_actual"][0, 0] / saved[f"{prefix}_reference"][0]
        curves = {"measured": measured, **{name: saved[f"{prefix}_{name}"][0, 0]
                   for name in ("learned_filter", "geometric_green", "green_envelope")}}
        for name, values in curves.items():
            response = np.fft.fftshift(np.fft.ifft(values))
            archive[f"array{array_index}_{name}"] = response
            axis.plot(lags*1e9, abs(response), label=name.replace("_", " "), lw=1.2)
        axis.set(xlabel="Relative periodic delay (ns)", ylabel="Magnitude (relative filter units)", title=f"Array {array_index+1}: fixed first test example")
        axis.legend(fontsize=7)
    figure.suptitle("Measured and predicted relative impulse responses — no absolute arrival-time claim", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "relative_impulse_responses.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(output / "impulse_examples.npz", **archive)


def plot_results(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = ("#a1a8b0", "#9380aa", "#c39b67", "#3374ac", "#479c82", "#15574b")
    for axis, group in zip(axes, ("dc04", "dc13")):
        for index, (name, color) in enumerate(zip(MODELS, colors)):
            values = [r["scores"]["all"]["transfer_nmse"] for r in result["evaluations"]
                      if r["file"] == group and r["model"] == name]
            axis.bar(np.arange(len(values))+(index-2.5)*.13, values, width=.125, color=color, label=name.replace("_", " "))
        axis.set_xticks([0, 1], ["North array", "South array"])
        axis.set(ylabel="Complex response NMSE (lower is better)", title="Held-out recording" if group == "dc04" else "Changed environment")
        axis.axhline(1., color="#666666", lw=.8, linestyle="--")
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle("Public radio measurements: learning a relative response kernel", fontsize=14)
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def run(data, output):
    manifest = json.loads((data / "provenance.json").read_text())
    if manifest["doi"] != "10.18419/DARUS-3831" or manifest["version"] != "1.0":
        raise ValueError("Unexpected dataset version")
    spec_bytes = (data / "spec.json").read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != manifest["spec_sha256"]:
        raise ValueError("Dataset specification hash mismatch")
    spec = json.loads(spec_bytes)
    frequencies = spec["frequency"] + np.fft.fftfreq(1024, d=1/spec["bandwidth"])[BINS]
    # Test-file observations are not decoded until all fits and choices are frozen.
    training, validation = load_group(data, manifest, "dc01"), load_group(data, manifest, "dc02")
    tasks, selections, calibration, audit = {}, {}, {}, {}
    for array_index, array in enumerate(spec["antennas"]):
        print(f"Fitting array {array_index+1}: {array['name']}...", flush=True)
        ports = np.asarray(array["assignments"]).reshape(-1)
        available = base_quality(training) & ~hole(training.x, BUFFER)
        reference_port = int(ports[np.argmax(np.median(training.snr[available][:, ports], axis=0))])
        output_ports = [int(p) for p in ports if p != reference_port]
        floor = .02 * np.sqrt(np.mean(abs(training.csi[available, reference_port])**2, axis=0))
        training_mask = receiver_quality(training, reference_port, floor) & ~hole(training.x, BUFFER)
        validation_mask = receiver_quality(validation, reference_port, floor) & ~hole(validation.x, BUFFER)
        if training_mask.sum() < 20 or validation_mask.sum() < 10:
            raise ValueError("Insufficient quality-controlled fitting data")
        x, xv = training.x[training_mask], validation.x[validation_mask]
        reference, reference_v = training.csi[training_mask, reference_port], validation.csi[validation_mask, reference_port]
        actual, actual_v = training.csi[training_mask][:, output_ports], validation.csi[validation_mask][:, output_ports]
        quotient = actual / reference[:, None, :]
        constant = fit_constant(reference, actual)
        learned, learned_selection = tune_kernel(x, quotient, reference, xv, reference_v, actual_v)
        configuration, gains, geometric_selection = tune_green(x, reference, actual, xv, reference_v, actual_v,
                                                              array, reference_port, output_ports, frequencies)
        propagation = green_quotient(x, array, reference_port, output_ports, frequencies, configuration)
        propagation_v = green_quotient(xv, array, reference_port, output_ports, frequencies, configuration)
        envelope, envelope_selection = tune_kernel(x, quotient/propagation, reference, xv, reference_v, actual_v, propagation_v)
        shuffled = np.random.default_rng(20261116+array_index).permutation(quotient)
        shuffled_model, shuffled_selection = tune_kernel(x, shuffled, reference, xv, reference_v, actual_v)
        tasks[array_index] = {"array": array, "reference_port": reference_port, "ports": output_ports,
                              "floor": floor, "x": x, "constant": constant, "learned": learned,
                              "configuration": configuration, "gains": gains, "envelope": envelope, "shuffled": shuffled_model}
        selections[str(array_index)] = {"learned_filter": learned_selection, "geometric_green": geometric_selection,
                                         "green_envelope": envelope_selection, "shuffled_geometry": shuffled_selection}
        calibration[str(array_index)] = {"reference_port": reference_port, "output_ports": output_ports,
                                          "reference_amplitude_floor": floor.tolist(), "green_configuration": configuration,
                                          "receiver_gain_real": gains.real.tolist(), "receiver_gain_imag": gains.imag.tolist()}
        audit[str(array_index)] = {"training_record_indices": training.indices[training_mask].tolist(),
                                    "validation_record_indices": validation.indices[validation_mask].tolist(),
                                    "downloaded_training": len(training.x), "downloaded_validation": len(validation.x),
                                    "accepted_training": int(training_mask.sum()), "accepted_validation": int(validation_mask.sum()),
                                    "training_geometry": x.tolist()}
    # No test labels have been used above. All models, gains and conventions now stay fixed.
    evaluations, saved = [], {"frequencies_hz": frequencies}
    test_quality = []
    for group_name in ("dc04", "dc13"):
        group = load_group(data, manifest, group_name)
        for array_index, task in tasks.items():
            reference_port, ports = task["reference_port"], task["ports"]
            accepted = receiver_quality(group, reference_port, task["floor"])
            if accepted.sum() == 0:
                raise ValueError(f"No valid test snapshots in {group_name}, array {array_index}")
            x = group.x[accepted]
            reference = group.csi[accepted, reference_port]
            actual = group.csi[accepted][:, ports]
            propagation = green_quotient(x, task["array"], reference_port, ports, frequencies, task["configuration"])
            predictions = {"zero_filter": np.zeros_like(actual),
                           "constant_filter": np.broadcast_to(task["constant"], actual.shape),
                           "learned_filter": task["learned"].predict(x),
                           "geometric_green": propagation*task["gains"][None, :, :],
                           "green_envelope": propagation*task["envelope"].predict(x),
                           "shuffled_geometry": task["shuffled"].predict(x)}
            distance, _ = cKDTree(task["x"]).query(x)
            masks = {"all": np.ones(len(x), dtype=bool), "spatial_hole": hole(x),
                     "near_training_within_1m": distance <= 1.}
            prefix = f"{group_name}_array{array_index}"
            saved.update({f"{prefix}_x": x, f"{prefix}_reference": reference, f"{prefix}_actual": actual,
                          f"{prefix}_record_indices": group.indices[accepted], f"{prefix}_nearest_training_distance": distance})
            for name, prediction in predictions.items():
                values = {key: score(actual, reference, prediction, mask) for key, mask in masks.items()}
                evaluations.append({"array": array_index, "file": group_name, "model": name, "scores": values,
                                    "median_nearest_training_distance_metres": float(np.median(distance))})
                saved[f"{prefix}_{name}"] = prediction
            test_quality.append({"array": array_index, "file": group_name, "downloaded_snapshots": len(group.x),
                                 "accepted_snapshots": int(accepted.sum()), "rejected_record_indices": group.indices[~accepted].tolist()})
    result = {"experiment": "measured_relative_radio_response_kernel", "synthetic_observations": False,
              "absolute_propagation_kernel_identified": False, "absolute_causality_or_light_speed_tested": False,
              "data_provenance": manifest, "protocol": {"training_file": "dc01", "validation_file": "dc02",
                  "test_file": "dc04", "environment_change_file": "dc13", "frequency_bins": BINS.tolist(),
                  "carrier_hz": spec["frequency"], "bandwidth_hz": spec["bandwidth"],
                  "spatial_hole_radius_metres": HOLE_RADIUS, "exclusion_buffer_metres": BUFFER,
                  "max_position_interpolation_age_seconds": .1, "reference_floor_fraction": .02,
                  "reference_selection": "highest median training SNR outside spatial exclusion, separately per array"},
              "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                        for name in ("run_experiment.py", "kernel_models.py", "records.py", "fetch_data.py")},
              "model_selection": selections, "calibration": calibration, "split_audit": audit,
              "test_quality_control": test_quality, "evaluations": evaluations}
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(output / "predictions.npz", **saved)
    write_report(result, output)
    plot_results(result, output)
    save_impulse_examples(saved, spec["bandwidth"], output)
    for row in evaluations:
        if row["file"] == "dc04":
            print(row["array"], row["model"], round(row["scores"]["all"]["transfer_nmse"], 6), flush=True)
    print(f"Report: {output / 'RESULTS.md'}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data" / "dichasus")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data, args.output)

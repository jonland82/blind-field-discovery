"""Compress an existing real-data kernel field into explicit small functions.

Exploratory reuse of previously scored test acquisitions. No generated fields,
Maxwell targets, test-driven fitting, or inferred source/current labels.
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from pathlib import Path
import hashlib
import json
import platform

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from compact_models import fit_polynomial, fit_sparse_path, teacher_gradient
from fetch_data import SENSORS
from models import FieldMap
from run_experiment import (ROOT, SOURCE, TEST_TRIALS, SHIFT_TRIALS, TRAIN_TRIALS,
                            VALIDATION_TRIALS, load_recording, spatial_split,
                            verify_archive, vector_rmse, metrics, in_hole, BUFFER)

OUTPUT = ROOT / "compact_field"
TOLERANCE = .05  # Select minimum basis count within 5% of teacher validation RMSE.
WIDTHS = (.2, .35, .7, 1.4)
COUNTS = (1, 2, 4, 8, 16, 32, 64)


def run():
    provenance = verify_archive(ROOT.parents[1] / "data/aalto_magnetic")
    old = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))
    for filename, expected in old["implementation_sha256"].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected
    recordings = {(SOURCE, trial): load_recording(ROOT.parents[1] / "data/aalto_magnetic", SOURCE, trial)
                  for trial in TRAIN_TRIALS + VALIDATION_TRIALS}
    x, y, training_audit = spatial_split(recordings, TRAIN_TRIALS, old["protocol"]["train_per_trial_cap"])
    xv, yv, validation_audit = spatial_split(recordings, VALIDATION_TRIALS, 600)
    config = old["model_selection"]["unrestricted"]["selected"]
    teacher = FieldMap("unrestricted", config["length_metres"], config["ridge"]).fit(x, y)
    teacher_targets = teacher.predict(x)
    teacher_validation = vector_rmse(yv, teacher.predict(xv))
    np.testing.assert_allclose(teacher_validation,
        old["model_selection"]["unrestricted"]["validation_vector_rmse"], rtol=1e-12)
    candidates = {}
    for degree in range(6):
        candidates[f"polynomial_degree_{degree}"] = fit_polynomial(x, teacher_targets, degree)
    for length in WIDTHS:
        for count, model in fit_sparse_path(x, teacher_targets, length, COUNTS).items():
            candidates[f"affine_plus_{count}_gaussians_length_{length:g}"] = model
    search = [{"name": name, "basis_count": len(model.coefficients),
               "validation_vector_rmse": vector_rmse(yv, model.predict(xv)),
               "validation_teacher_disagreement_rmse": vector_rmse(teacher.predict(xv), model.predict(xv))}
              for name, model in candidates.items()]
    accepted = [r for r in search if r["validation_vector_rmse"] <= (1+TOLERANCE)*teacher_validation]
    chosen = min(accepted, key=lambda r: (r["basis_count"], r["validation_vector_rmse"])) if accepted else min(
        search, key=lambda r: r["validation_vector_rmse"])
    selected = candidates[chosen["name"]]
    # A matched direct-data control uses the selected shape; it makes no selections.
    direct = type(selected)(selected.center, selected.scale, selected.powers,
        np.linalg.lstsq(selected.features(x), y, rcond=None)[0], selected.centers, selected.length)
    models = {"kernel_teacher": teacher, "compact_distilled": selected,
              "matched_direct_data": direct, "distilled_quadratic": candidates["polynomial_degree_2"]}
    print("Frozen before test scoring:", chosen, flush=True)
    teacher_g, student_g = teacher_gradient(teacher, xv), selected.gradient(xv)

    # Test labels are opened only after all candidate choices and controls freeze.
    archive_path = ROOT / "results/predictions.npz"
    archive = np.load(archive_path)
    rows, saved = [], {"training_xy": x, "training_actual": y,
                      "training_teacher_targets": teacher_targets, "validation_xy": xv}
    tree = cKDTree(x)
    for sensor in SENSORS:
        for trial in TEST_TRIALS + SHIFT_TRIALS:
            prefix = f"{sensor}_{trial}"
            query, actual = archive[prefix+"_x"], archive[prefix+"_actual_aligned"]
            nearest, _ = tree.query(query)
            masks = {"all": np.ones(len(query), dtype=bool), "spatial_hole": in_hole(query),
                     "outside_spatial_exclusion": ~in_hole(query, BUFFER),
                     "within_0.25m_of_training": nearest <= .25}
            predictions = {name: model.predict(query) for name, model in models.items()}
            predictions["training_mean"] = np.tile(np.mean(y, axis=0), (len(query), 1))
            np.testing.assert_allclose(predictions["kernel_teacher"], archive[prefix+"_unrestricted"], atol=1e-10)
            saved.update({prefix+"_xy": query, prefix+"_actual": actual,
                          prefix+"_original_rows": archive[prefix+"_row_indices"]})
            for name, prediction in predictions.items():
                saved[prefix+"_"+name] = prediction
                rows.append({"sensor": sensor, "trial": trial, "model": name,
                             "environment": "stable" if trial in TEST_TRIALS else "changed",
                             "scores": {name: metrics(actual, prediction, mask) for name, mask in masks.items()},
                             "teacher_disagreement_rmse": vector_rmse(predictions["kernel_teacher"], prediction)})
    pooled = {}
    for environment, trials in (("stable", TEST_TRIALS), ("changed", SHIFT_TRIALS)):
        pooled[environment] = {}
        for sensor in SENSORS:
            pooled[environment][sensor] = {}
            for model in ("training_mean", *models):
                scores = [row["scores"]["all"] for row in rows if row["sensor"] == sensor
                          and row["trial"] in trials and row["model"] == model]
                pooled[environment][sensor][model] = float(np.sqrt(
                    sum(r["n"]*r["vector_rmse"]**2 for r in scores)/sum(r["n"] for r in scores)))
    result = {"purpose": "Extract an explicit compact field function from a kernel trained on real magnetic measurements.",
        "synthetic_physical_observations": False,
        "distillation_targets": "Predictions of the real-data-trained teacher at actual training positions; model outputs, not independent measurements.",
        "test_reuse": "Exploratory reuse of Aalto trials already scored in earlier work. No fresh confirmatory test or significance claim.",
        "physical_equation_or_source_law_discovered": False,
        "units": old["protocol"]["units"], "data_provenance": provenance,
        "protocol": {"source": SOURCE, "train_trials": TRAIN_TRIALS, "validation_trials": VALIDATION_TRIALS,
            "test_trials": TEST_TRIALS, "stress_trials": SHIFT_TRIALS,
            "selection": "Minimum basis count within 5% of teacher source-validation RMSE; break ties by RMSE; if none qualify report failure and best-error fallback.",
            "polynomial_degrees": list(range(6)), "gaussian_widths_metres": WIDTHS,
            "gaussian_counts": COUNTS, "candidate_centers": "Actual training positions only.",
            "fitting_targets": "Kernel teacher predictions at training positions only.",
            "held_out_hole_and_alignment": old["protocol"],
            "device_alignment": old["device_alignment"]},
        "teacher_config": config, "teacher_basis_count": len(x)+1,
        "teacher_validation_vector_rmse": teacher_validation,
        "training_rows": training_audit, "validation_rows": validation_audit,
        "candidate_search": search, "selected": chosen,
        "met_five_percent_validation_tolerance": bool(accepted),
        "selected_function": selected.to_dict(),
        "validation_gradient_relative_disagreement": float(np.linalg.norm(student_g-teacher_g)/np.linalg.norm(teacher_g)),
        "derivative_scope": "Analytic sensitivities of fitted functions; no independently measured field derivatives or currents are available.",
        "pooled_vector_rmse": pooled, "evaluations": rows,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("run_compact_field.py", "compact_models.py", "models.py", "run_experiment.py")},
        "input_results_sha256": {name: hashlib.sha256((ROOT/"results"/name).read_bytes()).hexdigest()
            for name in ("results.json", "predictions.npz")}}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(OUTPUT/"predictions.npz", **saved)
    write_report(result)
    plot_results(result, saved)
    print(json.dumps({"selected": chosen, "pooled": pooled}, indent=2))
    return result


def write_report(result):
    spec = result["selected_function"]
    center, scale = spec["coordinate_center_metres"], spec["coordinate_scale_metres"]
    lines = ["# A compact magnetic-field function extracted from real-data learning", "",
        "**Question:** can the kernel map learned from public magnetic measurements be expressed as a smaller, explicit function without losing its predictive usefulness?", "",
        "This is an empirical field map for one room and measurement frame. No Maxwell constraints, dipole formula, current labels, or synthetic physical fields enter fitting.", "",
        "## Selected function", "",
        f"The selected function is **{result['selected']['name']}**, with **{spec['basis_count']} basis terms per channel**, compared with {result['teacher_basis_count']} for the original kernel map (including its mean).",
        f"It {'met' if result['met_five_percent_validation_tolerance'] else 'did not meet'} the predeclared source-validation requirement of at most 5% higher RMSE than the teacher.", "",
        "Coordinates $x,y$ are supplied positions in metres. Define normalized coordinates", "",
        "$$", f"u=(x-({center[0]:.12g}))/({scale[0]:.12g}),\\qquad v=(y-({center[1]:.12g}))/({scale[1]:.12g}).", "$$", "",
        "The three measured field channels are represented by", "", "$$",
        "\\mathbf B_{\\mathrm{compact}}(x,y)=\\sum_{p,q}\\mathbf c_{pq}u^pv^q"
        + ("+\\sum_{m=1}^{M}\\mathbf w_m\\exp\\!\\left[-\\frac{(x-a_m)^2+(y-b_m)^2}{2\\ell^2}\\right]." if spec["gaussian_centers_metres"] else "."),
        "$$", "",
        "Coefficients below are in the reference recording's numerical field units; no unverified SI conversion is made. All three channels share the displayed spatial basis.", "",
        "| Polynomial powers (p,q) | Channel 1 | Channel 2 | Channel 3 |", "| --- | ---: | ---: | ---: |"]
    count = len(spec["polynomial_powers"])
    for powers, coefficients in zip(spec["polynomial_powers"], spec["coefficients_by_basis_then_channel"][:count]):
        lines.append(f"| {tuple(powers)} | " + " | ".join(f"{v:.12g}" for v in coefficients) + " |")
    if spec["gaussian_centers_metres"]:
        lines += ["", f"Here M = {len(spec['gaussian_centers_metres'])} and $\\ell={spec['gaussian_length_metres']:g}$ m.", "",
            "| m | Center x (m) | Center y (m) | Weight channel 1 | Weight channel 2 | Weight channel 3 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
        for i, (xy, weights) in enumerate(zip(spec["gaussian_centers_metres"], spec["coefficients_by_basis_then_channel"][count:]), 1):
            lines.append(f"| {i} | " + " | ".join(f"{v:.12g}" for v in (*xy, *weights)) + " |")
    lines += ["", "## Predictions against measured recordings", "",
        "Vector RMSE pools all three components and both stable test trials. The teacher and compact map predict exactly the same observations. Other instruments use the earlier trial-1-only frozen affine alignment.", "",
        "| Instrument | Constant | Quadratic from teacher | Original kernel | Compact distilled | Matched direct-data fit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sensor in SENSORS:
        scores = result["pooled_vector_rmse"]["stable"][sensor]
        lines.append(f"| {sensor} | " + " | ".join(f"{scores[m]:.4f}" for m in
            ("training_mean", "distilled_quadratic", "kernel_teacher", "compact_distilled", "matched_direct_data")) + " |")
    lines += ["", "## Frozen-map changed-environment stress test", "",
        "| Instrument | Constant | Original kernel | Compact distilled |", "| --- | ---: | ---: | ---: |"]
    for sensor in SENSORS:
        scores = result["pooled_vector_rmse"]["changed"][sensor]
        lines.append(f"| {sensor} | " + " | ".join(f"{scores[m]:.4f}" for m in
            ("training_mean", "kernel_teacher", "compact_distilled")) + " |")
    lines += ["", "## What can be differentiated", "",
        "For a Gaussian term $g_m$, its spatial derivatives are analytic:", "", "$$",
        "\\partial_x g_m=-\\frac{x-a_m}{\\ell^2}g_m,\\qquad\\partial_y g_m=-\\frac{y-b_m}{\\ell^2}g_m.",
        "$$", "", "Polynomial terms differentiate in the usual way, with the coordinate-scale factors above.",
        f"The compact map's gradient differs from the teacher's by {100*result['validation_gradient_relative_disagreement']:.1f}% in relative Frobenius norm at validation positions. This is model-to-model disagreement, not error against a measured true derivative.", "",
        "A short function can preserve values while changing derivatives. These recordings supply no independent current density, electric field, or normal spatial derivative. We cannot identify $B'=aJ$, full curl, divergence, or Maxwell merely by differentiating this map.", "",
        "## Protocol and limits", "",
        "- Public Aalto data, pinned and hash-verified; train Invensense trials 1–2, select the teacher and student on source trial 3, evaluate trials 4–5 and stress trials 6–9.",
        "- The earlier spatial rectangle and buffer remain excluded from training and validation. Per-trial and spatial-hole results are preserved in JSON.",
        "- Student fitting uses teacher predictions only at actual training positions. Those targets are derived model outputs, not additional independent observations. Validation and evaluation use actual measured labels.",
        "- Candidate polynomials have degree 0–5; sparse affine-plus-Gaussian maps use widths 0.2, 0.35, 0.7, 1.4 m and 1, 2, 4, 8, 16, 32, 64 shared centers. Geometry and coefficient fitting use training only. The matched direct-data control reuses the selected shape and fits coefficients to measured training labels.",
        "- Pick the fewest basis terms within 5% of the original teacher's source-validation RMSE, breaking ties by error. If no candidate qualifies, disclose failure and report the minimum-error fallback. The tolerance is a modeling choice, not a confidence interval.",
        "- These test recordings have already been examined in prior experiments. This is exploratory reuse, not a fresh confirmatory test. No test scores tune this run's candidates or settings.",
        "- Known coordinates, upstream calibration/orientation correction, the basis family and static-map assumption are supplied. A compact empirical field function does not establish a unique source arrangement or a universal law.", "",
        "## How much further to a theory?", "",
        "This step asks whether a measured-data predictor admits a smaller explicit representation. Next, quantify coefficient and derivative stability across training subsets, then replicate the frozen extraction procedure on an independent public magnetic dataset. A source-response equation requires independently measured sources and suitable geometry; this dataset cannot supply them. A new dataset is needed to establish that stronger claim.", "",
        "[Comparison plot](comparison.png) · [Machine-readable results and full coefficients](results.json).", "",
        "Reproduce from the project root: `python experiments/magnetic_mapping/run_compact_field.py`.", "",
        "Source: Solin et al. (2018), [Aalto magnetic recordings](https://github.com/AaltoML/magnetic-data), [paper](https://doi.org/10.1109/TRO.2018.2830326). The upstream data are already calibrated and orientation-corrected. Attribution and original provenance are retained; no new redistribution licence is asserted."]
    (OUTPUT/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(result, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    search = result["candidate_search"]
    for group, color in (("polynomial", "#9472a6"), ("affine", "#338b87")):
        subset = [row for row in search if row["name"].startswith(group)]
        axes[0].scatter([r["basis_count"] for r in subset], [r["validation_vector_rmse"] for r in subset],
                        label="Polynomials" if group == "polynomial" else "Affine + Gaussian terms", color=color, alpha=.75)
    axes[0].axhline(result["teacher_validation_vector_rmse"], color="#193c66", label="Original kernel")
    axes[0].axhline(result["teacher_validation_vector_rmse"]*(1+TOLERANCE), ls="--", color="#555555", label="5% acceptance margin")
    axes[0].scatter([result["selected"]["basis_count"]], [result["selected"]["validation_vector_rmse"]], marker="*", s=160, color="#d47b2b", zorder=3, label="Selected compact map")
    axes[0].set(xlabel="Basis terms per output channel", ylabel="Measured validation vector RMSE",
                title="Selection uses source validation only", xscale="log")
    axes[0].legend(fontsize=8)
    prefix = "invensense_4"
    query = saved[prefix+"_xy"]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(query, axis=0), axis=1))]
    axes[1].scatter(distance, saved[prefix+"_actual"][:, 2], color="#abb4bf", s=4, label="Measured channel 3")
    axes[1].plot(distance, saved[prefix+"_kernel_teacher"][:, 2], color="#193c66", lw=1.2, label="Original kernel")
    axes[1].plot(distance, saved[prefix+"_compact_distilled"][:, 2], color="#d47b2b", lw=1.2, label="Compact function")
    axes[1].set(xlabel="Cumulative distance along sampled trial-4 path (m)",
                ylabel="Channel 3 (reference recording units)", title="First held-out trial, fixed channel")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Extracting a small empirical magnetic-field function from real-data learning")
    fig.savefig(OUTPUT/"comparison.png", dpi=170)
    fig.savefig(OUTPUT/"comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()

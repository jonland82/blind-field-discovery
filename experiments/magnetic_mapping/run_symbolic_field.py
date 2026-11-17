"""Test kernel-to-equation-to-analytic-function extraction on real magnetic data.

Fixed geometry-only band and four local intervals; no synthetic experimental
labels, Maxwell-derived source channels, or test-based model selection.
"""
import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from pathlib import Path
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from fetch_data import SENSORS
from run_experiment import (ROOT, SOURCE, TEST_TRIALS, SHIFT_TRIALS, load_recording,
                            verify_archive, evenly_spaced, apply_alignment)
from symbolic_models import ScalarKernel, extract_candidates

OUTPUT = ROOT/"symbolic_field"
HALF_BAND = .15
LENGTHS = (.1, .2, .35, .7, 1.4)
RIDGES = (.001, .01, .1, 1.)
DERIVATIVE_TOLERANCE = .10
VALUE_TOLERANCE = .10
MIN_CONSTANT_IMPROVEMENT = .05


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual)-predicted)**2)))


def safe_rmse(actual, predicted):
    return rmse(actual, predicted) if len(actual) and np.isfinite(predicted).all() else None


def band_sample(rec, band_y, bounds, maximum, alignment=None):
    lower, upper, include_upper = bounds
    mask = ((abs(rec.x[:, 1]-band_y) <= HALF_BAND) & (rec.x[:, 0]>=lower)
            & ((rec.x[:, 0]<=upper) if include_upper else (rec.x[:, 0]<upper)))
    indices = evenly_spaced(np.flatnonzero(mask), maximum)
    readings = rec.y[indices] if alignment is None else apply_alignment(rec.y[indices], alignment)
    return {"s": rec.x[indices, 0], "xy": rec.x[indices], "b": np.linalg.norm(readings, axis=1),
            "original_rows": rec.original_rows[indices], "trial": rec.trial, "sensor": rec.sensor}


def combine(blocks):
    return {key: np.concatenate([block[key] for block in blocks]) for key in ("s", "xy", "b", "original_rows")}


def select_region(train, validation):
    s, b, sv, bv = train["s"], train["b"], validation["s"], validation["b"]
    if len(s)<30 or len(sv)<15 or np.ptp(s)<.1:
        return {"status": "insufficient_training_or_validation_geometry"}, None
    best, grid = None, []
    for length in LENGTHS:
        for ridge in RIDGES:
            kernel = ScalarKernel(length, ridge).fit(s, b)
            score = rmse(bv, kernel.predict(sv))
            grid.append({"length": length, "ridge": ridge, "validation_rmse": score})
            if best is None or score<best[0]:
                best = score, kernel
    kernel_error, kernel = best
    candidates, search = extract_candidates(kernel, s, b, sv)
    constant = float(np.mean(b))
    constant_error = rmse(bv, constant)
    for row in search:
        model = candidates[row["name"]]
        score = safe_rmse(bv, model.predict(sv))
        # Reject poles/overflow over the whole training-defined interval, not just sample positions.
        grid_prediction = model.predict(np.linspace(min(s), max(s), 401))
        finite_domain = bool(np.isfinite(grid_prediction).all() and
            np.max(abs(grid_prediction-constant)) <= 20*max(float(np.std(b)), 1e-9))
        row.update({"validation_measured_rmse": score, "finite_on_training_domain": finite_domain})
        row["accepted"] = bool(finite_domain and score is not None
            and row["validation_kernel_derivative_relative_residual"] <= DERIVATIVE_TOLERANCE
            and score <= (1+VALUE_TOLERANCE)*kernel_error
            and score <= (1-MIN_CONSTANT_IMPROVEMENT)*constant_error)
    accepted = [r for r in search if r["accepted"]]
    finite = [r for r in search if r["finite_on_training_domain"] and r["validation_measured_rmse"] is not None]
    chosen = min(accepted, key=lambda r: (r["derivative_terms_plus_reference"], r["validation_measured_rmse"])) if accepted else min(
        finite, key=lambda r: r["validation_measured_rmse"])
    selected = candidates[chosen["name"]]
    riccati_rows = [r for r in search if r["name"].startswith("riccati") and r["validation_measured_rmse"] is not None and r["finite_on_training_domain"]]
    riccati_choice = min(riccati_rows, key=lambda r: r["validation_measured_rmse"]) if riccati_rows else None
    models = {"kernel": kernel, "analytic_selected": selected}
    if riccati_choice:
        models["best_riccati_diagnostic"] = candidates[riccati_choice["name"]]
    return {"status": "accepted_analytic" if accepted else "no_analytic_candidate_passed",
        "training_count": len(s), "validation_count": len(sv),
        "kernel_selection": grid, "kernel_length": kernel.length, "kernel_ridge": kernel.ridge,
        "kernel_validation_rmse": kernel_error, "constant_validation_rmse": constant_error,
        "constant_value": constant, "candidate_search": search, "selected": chosen,
        "selected_function": selected.to_dict(),
        "best_riccati_diagnostic": riccati_choice,
        "best_riccati_function": candidates[riccati_choice["name"]].to_dict() if riccati_choice else None,
        "strict_one_percent_derivative_candidates": [r["name"] for r in search
            if r["validation_kernel_derivative_relative_residual"]<=.01],
        "training_rows": train["original_rows"].tolist(), "validation_rows": validation["original_rows"].tolist()}, models


def run():
    provenance = verify_archive(ROOT.parents[1]/"data/aalto_magnetic")
    previous = json.loads((ROOT/"results/results.json").read_text(encoding="utf-8"))
    # Only source training and validation recordings are opened before selection.
    recordings = {trial: load_recording(ROOT.parents[1]/"data/aalto_magnetic", SOURCE, trial) for trial in (1, 2, 3)}
    training_xy = np.concatenate([recordings[trial].x for trial in (1, 2)])
    band_y = float(np.median(training_xy[:, 1]))
    training_band_x = training_xy[abs(training_xy[:, 1]-band_y)<=HALF_BAND, 0]
    edges = np.linspace(training_band_x.min(), training_band_x.max(), 5)
    bounds = {"whole_band": (edges[0], edges[-1], True),
              **{f"interval_{i+1}": (edges[i], edges[i+1], i==3) for i in range(4)}}
    regions, models, saved = {}, {}, {}
    for name, limits in bounds.items():
        blocks = [band_sample(recordings[t], band_y, limits, 200) for t in (1, 2)]
        train = combine(blocks)
        valid = band_sample(recordings[3], band_y, limits, 400)
        regions[name], models[name] = select_region(train, valid)
        regions[name]["bounds_metres"] = list(limits)
        regions[name]["training_by_trial"] = [{"trial": b["trial"], "original_rows": b["original_rows"].tolist()} for b in blocks]
        saved.update({name+"_train_s": train["s"], name+"_train_b": train["b"],
                      name+"_validation_s": valid["s"], name+"_validation_b": valid["b"]})
        if models[name]:
            saved[name+"_kernel_centers"] = models[name]["kernel"].s
            saved[name+"_kernel_weights"] = models[name]["kernel"].weights
        print("Frozen:", name, regions[name]["status"], regions[name].get("selected"), flush=True)
    # All region definitions and function selections now freeze; read test labels.
    rows = []
    for sensor in SENSORS:
        alignment = np.vstack((np.eye(3), np.zeros(3))) if sensor == SOURCE else np.asarray(
            previous["device_alignment"][sensor]["target_to_reference_affine"])
        for trial in TEST_TRIALS + SHIFT_TRIALS:
            rec = load_recording(ROOT.parents[1]/"data/aalto_magnetic", sensor, trial)
            for region, limits in bounds.items():
                if not models[region]:
                    continue
                block = band_sample(rec, band_y, limits, 1000, alignment)
                if not len(block["s"]):
                    continue
                prefix = f"{region}_{sensor}_{trial}"
                saved.update({prefix+"_s": block["s"], prefix+"_xy": block["xy"],
                              prefix+"_actual": block["b"], prefix+"_original_rows": block["original_rows"]})
                predictions = {name: model.predict(block["s"]) for name, model in models[region].items()}
                predictions["constant"] = np.full(len(block["s"]), regions[region]["constant_value"])
                for name, prediction in predictions.items():
                    saved[prefix+"_"+name] = prediction
                    rows.append({"region": region, "sensor": sensor, "trial": trial,
                        "environment": "stable" if trial in TEST_TRIALS else "changed",
                        "model": name, "n": len(block["s"]),
                        "rmse": safe_rmse(block["b"], prediction),
                        "nonfinite_predictions": int((~np.isfinite(prediction)).sum()),
                        "analytic_selection_passed": regions[region]["status"] == "accepted_analytic"})
    pooled = []
    for region in (*bounds, "local_intervals_pooled"):
        allowed = [region] if region != "local_intervals_pooled" else list(bounds)[1:]
        for sensor in SENSORS:
            for environment in ("stable", "changed"):
                for model in ("constant", "kernel", "analytic_selected", "best_riccati_diagnostic"):
                    group = [r for r in rows if r["region"] in allowed and r["sensor"]==sensor
                             and r["environment"]==environment and r["model"]==model]
                    if not group:
                        continue
                    total = sum(r["n"] for r in group)
                    score = float(np.sqrt(sum(r["n"]*r["rmse"]**2 for r in group)/total)) if all(r["rmse"] is not None for r in group) else None
                    pooled.append({"region": region, "sensor": sensor, "environment": environment,
                                   "model": model, "n": total, "rmse": score})
    result = {"purpose": "Real-data kernel-to-differential-equation-to-analytic-function experiment on a simplified magnetic profile.",
        "synthetic_physical_observations": False, "source_law_or_maxwell_recovered": False,
        "observable": "Euclidean norm of three real magnetometer channels, in frozen reference instrument units.",
        "protocol": {"train_trials": [1,2], "validation_trials": [3], "test_trials": TEST_TRIALS,
            "stress_trials": SHIFT_TRIALS, "band_center_y_metres": band_y, "band_half_width_metres": HALF_BAND,
            "scalar_coordinate": "s=x in published metres; y varies within the narrow band.",
            "local_edges_metres": edges.tolist(), "geometry_selection": "Training positions only, no field values.",
            "original_spatial_hole": "Not used in this separately defined 1D experiment. Whole acquisition trials remain held out.",
            "per_training_trial_region_cap": 200, "validation_region_cap": 400, "test_trial_region_cap": 1000,
            "equation_derivative_relative_tolerance": DERIVATIVE_TOLERANCE,
            "equation_value_tolerance_relative_to_kernel": VALUE_TOLERANCE,
            "minimum_validation_improvement_over_constant": MIN_CONSTANT_IMPROVEMENT,
            "candidate_families": "All seven supports in z'=c0+c1*z+c2*z^2, plus coordinate polynomials of derivative degree 0..3.",
            "kernel_lengths_metres": LENGTHS, "kernel_ridges": RIDGES,
            "selection": "Smallest derivative support plus one kernel reference, meeting all validation tolerances. Otherwise label failure and show best-value fallback.",
            "test_reuse": "Exploratory reuse of acquisitions already scored previously; not a fresh confirmatory experiment.",
            "all_regions_reported": True},
        "units": previous["protocol"]["units"], "data_provenance": provenance,
        "device_alignment": previous["device_alignment"],
        "regions": regions, "evaluations": rows, "pooled_scores": pooled,
        "derivative_scope": "Kernel derivatives are model outputs. No independent measured current, gradient, or normal derivative is supplied.",
        "implementation_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ("run_symbolic_field.py", "symbolic_models.py", "run_experiment.py", "fetch_data.py")},
        "input_pilot_results_sha256": hashlib.sha256((ROOT/"results/results.json").read_bytes()).hexdigest()}
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.savez_compressed(OUTPUT/"predictions.npz", **saved)
    write_report(result)
    plot_results(result, saved, models)
    print("Results:", OUTPUT/"RESULTS.md", flush=True)
    return result


def write_report(result):
    protocol = result["protocol"]
    lines = ["# From real magnetic measurements to explicit field functions", "",
        "**This experiment runs the kernel → derivative equation → analytic solution workflow on public measured magnetic recordings.** It is a simplified one-dimensional field-strength profile, not a recovery of Maxwell or source/current equations.", "",
        "## Observed result", ""]
    lookup = {(r["region"],r["sensor"],r["environment"],r["model"]): r for r in result["pooled_scores"]}
    get = lambda region, model: lookup[region,SOURCE,"stable",model]["rmse"]
    passed = [name for name,r in result["regions"].items() if r["status"]=="accepted_analytic"]
    lines += [f"The original 1D kernel predicts held-out source-instrument readings with RMSE **{get('whole_band','kernel'):.3f}**, versus **{get('whole_band','constant'):.3f}** for a constant prediction. No single global Riccati equation captures the multi-peaked profile: its analytic prediction has RMSE **{get('whole_band','analytic_selected'):.3f}**.", "",
        f"The four fixed local intervals give pooled RMSE **{get('local_intervals_pooled','analytic_selected'):.3f}** for their analytic candidates, **{get('local_intervals_pooled','kernel'):.3f}** for their kernels, and **{get('local_intervals_pooled','constant'):.3f}** for local constants. **{len(passed)} of 5 region extractions pass all validation requirements.** Good value prediction alone has not established a reliable derivative equation. The candidates remain useful explicit approximations, with acceptance failures retained.", "",
        "The 1D simplification also loses some cross-instrument predictive performance: the phone's whole-band kernel does not beat its constant baseline. The full tables retain these failures and changed-environment results.", "",
        "## Starting question and data", "",
        "Can a room's measured magnetic field, first learned as a kernel, be expressed by a short differential equation whose solved function predicts other recordings?", "",
        f"Use the public Aalto magnetometer data. The scalar B is the norm of all three channels, s is the published x position, and observations are restricted to y = {protocol['band_center_y_metres']:.6f} ± {HALF_BAND:.2f} m. This band is fixed by training geometry alone. Its finite width means B(s) is an approximation to a spatial profile, not a measurement along an exact mathematical line.", "",
        "Source trials 1–2 train, trial 3 selects, trials 4–5 evaluate, and trials 6–9 stress the frozen functions after environmental changes. Four equal-width local intervals are fixed from the training band's x extent; every interval and the whole band are reported. No magnetic values select a favorable location. Other sensors use the earlier trial-1-only frozen affine calibration.", "",
        "**Exploratory scope:** these acquisition groups have previously been scored in other project experiments. This run freezes its choices before opening their labels, but they are not a new confirmatory test. This separate 1D protocol does not retain the old 2D spatial hole.", "",
        "## Formalism", "", "$$",
        "\\widehat B(s)=\\overline B+\\sum_i\\alpha_i e^{-(s-s_i)^2/(2\\ell^2)},\\qquad"
        "\\widehat B'(s)=-\\sum_i\\alpha_i\\frac{s-s_i}{\\ell^2}e^{-(s-s_i)^2/(2\\ell^2)}.", "$$", "",
        "Normalize the fitted field as z=(B−m)/σ, where m and σ come from training measurements. Fit all supports of the candidate equation", "", "$$",
        "z'=c_0+c_1z+c_2z^2.", "$$", "",
        "Each candidate is solved analytically using z=u/v and", "", "$$",
        "\\frac{d}{ds}\\begin{pmatrix}u\\\\v\\end{pmatrix}="
        "\\begin{pmatrix}c_1&c_0\\\\-c_2&0\\end{pmatrix}\\begin{pmatrix}u\\\\v\\end{pmatrix}.", "$$", "",
        "Its matrix exponential yields an explicit ratio of hyperbolic functions, trigonometric functions, or their linear/rational limits. This includes the toy example's tanh branch without fitting tanh features. An interior kernel value at the median training s fixes the integration constant.", "",
        "Explicitly, set $t=s-s_r$, $z_r=(B_r-m)/\\sigma$, $\\Delta=c_1^2-4c_0c_2$, and let $h=\\cosh(\\sqrt{\\Delta}t/2)$, $g=2\\sinh(\\sqrt{\\Delta}t/2)/\\sqrt{\\Delta}$ for $\\Delta>0$. Then", "", "$$",
        "B(s)=m+\\sigma\\frac{h z_r+g(c_1z_r/2+c_0)}{h+g(-c_2z_r-c_1/2)}.", "$$", "",
        "For negative discriminant use cosine and sine with $\\sqrt{-\\Delta}$; at zero discriminant use $h=1$, $g=t$. Poles/nonfinite predictions remain failures rather than being silently dropped.", "",
        "Also search coordinate-dependent derivative polynomials of degree 0–3. Integrating them gives a field polynomial of degree 1–4. Their derivative coefficients are fitted to the kernel derivative, not directly to the measured field values; the integration reference comes from the kernel.", "",
        "Validation requires derivative residual ≤10%, measured prediction RMSE ≤110% of the kernel RMSE, and ≥5% improvement over the training-mean baseline. Select the fewest derivative terms plus reference among passing candidates, breaking ties by measured validation error. Failure is reported if no candidate passes; a best-value analytic fallback remains visible but is not accepted as a recovered equation. The original toy's stricter 1% derivative diagnostic is also preserved.", "",
        "## Region-by-region extraction", "",
        "| Region | x interval (m) | Train / validate | Status | Function | Kernel validation RMSE | Analytic validation RMSE | Derivative residual |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |"]
    for name, region in result["regions"].items():
        if "selected" not in region:
            lines.append(f"| {name} | {region['bounds_metres'][:2]} | n/a | {region['status']} | n/a | n/a | n/a | n/a |")
            continue
        selected = region["selected"]
        lo, hi, _ = region["bounds_metres"]
        lines.append(f"| {name} | {lo:.3f} to {hi:.3f} | {region['training_count']} / {region['validation_count']} | {region['status']} | {selected['name']} | {region['kernel_validation_rmse']:.4f} | {selected['validation_measured_rmse']:.4f} | {100*selected['validation_kernel_derivative_relative_residual']:.1f}% |")
    lines += ["", "## Scores against actual held-out measurements", "",
        "RMSE is for scalar field magnitude in reference numerical recording units, not an asserted SI unit. Local pooled scores combine all four intervals, including unsuccessful extractions. The local constant baseline has a separately fitted mean for each interval. Analytical fallback scores must be read together with their region's acceptance status.", "",
        "| Scope | Instrument | Environment | N | Constant | Kernel | Selected analytic | Best Riccati diagnostic |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for scope in ("whole_band", "local_intervals_pooled"):
        for sensor in SENSORS:
            for environment in ("stable", "changed"):
                group = [r for r in result["pooled_scores"] if r["region"]==scope and r["sensor"]==sensor and r["environment"]==environment]
                if not group:
                    continue
                scores = {r["model"]: r for r in group}
                values = ["failed" if scores.get(m, {}).get("rmse") is None else f"{scores[m]['rmse']:.4f}"
                    for m in ("constant", "kernel", "analytic_selected", "best_riccati_diagnostic")]
                lines.append(f"| {scope} | {sensor} | {environment} | {group[0]['n']} | " + " | ".join(values) + " |")
    lines += ["", "## The actual selected functions", ""]
    for name, region in result["regions"].items():
        if "selected_function" not in region:
            continue
        spec = region["selected_function"]
        lines.extend([f"### {name}", "", f"Status: **{region['status']}**. Reference position $s_r={spec['reference_s_metres']:.12g}$ m.", ""])
        if spec["kind"] == "polynomial":
            formula = "+".join(f"({c:.12g})(s-({spec['reference_s_metres']:.12g}))^{k}" for k,c in enumerate(spec["coefficients"]))
            lines.extend(["$$", f"B_{{\\mathrm{{analytic}}}}(s)={formula}.", "$$", "",
                "Its derivative is the fitted coordinate-polynomial equation; its closed-form field is obtained by exact integration. This is a local spatial approximation, not a source law.", ""])
        else:
            c0, c1, c2 = spec["coefficients"]
            lines.extend(["$$", f"z'=({c0:.12g})+({c1:.12g})z+({c2:.12g})z^2,\\qquad"
                f"z=(B-({spec['field_mean']:.12g}))/({spec['field_scale']:.12g}).", "$$", "",
                f"Solution family: {spec['solution_family']}; at s = {spec['reference_s_metres']:.12g}, B = {spec['reference_b']:.12g}.", ""])
            if "tanh_equivalent" in spec:
                t = spec["tanh_equivalent"]
                lines += ["Equivalently, the solved candidate is", "", "$$",
                    f"B_{{\\mathrm{{analytic}}}}(s)=({t['background']:.12g})+({t['amplitude']:.12g})"
                    f"\\tanh\\!\\left[\\frac{{s-({t['center_metres']:.12g})}}{{{t['width_metres']:.12g}}}\\right].", "$$", "",
                    "This is an analytic re-expression of the fitted equation, not a tanh fit directly to measured labels. Its validation status above still applies.", ""]
    lines += ["## What this establishes and the remaining gap", "",
        "An accepted function is a compact empirical description within its band/interval under fixed validation rules. Agreement of its solved function with measured values supports that scope. It is not unique: other basis families or source arrangements may fit the same finite observations.", "",
        "A successful local polynomial does not demonstrate that the source-free autonomous toy equation applies. Conversely, a failed global Riccati fit says that this scalar autonomous family cannot represent the observed room profile well under this protocol; it does not refute Maxwell.", "",
        "Derivative labels are outputs of a fitted kernel. The 0.3 m band has transverse variation; sensor noise, smoothing, position errors and drift also affect them. B here is field magnitude, so its derivative is not a curl component. Current density, electric field and out-of-plane gradients are unobserved. No missing source is manufactured from Maxwell.", "",
        "A nonconstant solution of a smooth scalar autonomous first-order equation cannot repeatedly turn up and down: at each B value, the equation assigns only one slope. The multi-peaked learned whole-band profile is therefore structurally incompatible with a single Riccati/tanh transition. This concerns the fitted profile and tested representation, not an independent physical derivative measurement.", "",
        "Next: test whether coefficients and derivatives remain stable under changed training subsets and band width, and replicate the fixed procedure on an independent public dataset. A genuine source-response law needs independent source measurements or a suitable controlled-source public dataset. Compact fit, derivative agreement, and a universal field law are different claims.", "",
        "Whole-band and local pooled sample counts can differ because each region has its own fixed cap. Compare competing models within the same table row; cross-scope differences are not matched-sample error estimates.", "",
        "[Plots](comparison.png) · [Full audit, model equations and all per-trial results](results.json).", "",
        "Reproduce: `python experiments/magnetic_mapping/run_symbolic_field.py` from the workspace root.", "",
        "Source: Solin et al. (2018), [public Aalto magnetic recordings](https://github.com/AaltoML/magnetic-data), [paper](https://doi.org/10.1109/TRO.2018.2830326). Hashes, pinned version and upstream preprocessing are retained in results.json. The authors calibrated and orientation-corrected these measurements; no raw-variable discovery is claimed."]
    (OUTPUT/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot_results(result, saved, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    names = list(result["regions"])
    for axis, region in zip(axes.flat, names):
        if not models[region]:
            axis.set_title(region+": insufficient geometry")
            continue
        prefix = f"{region}_{SOURCE}_4"
        for trial, color in ((4,"#9aa7b7"),(5,"#bac3ae")):
            prefix = f"{region}_{SOURCE}_{trial}"
            if prefix+"_s" in saved:
                axis.scatter(saved[prefix+"_s"], saved[prefix+"_actual"], color=color,
                             s=8, alpha=.55, label=f"Actual trial-{trial} measurements")
        lo, hi, _ = result["regions"][region]["bounds_metres"]
        query = np.linspace(lo, hi, 500)
        for name, color, style in (("kernel", "#1e587f", "-"), ("analytic_selected", "#c97625", "--"),
                                   ("best_riccati_diagnostic", "#8b629a", ":")):
            if name in models[region]:
                axis.plot(query, models[region][name].predict(query), color=color, ls=style, lw=1.5, label=name.replace("_", " "))
        axis.axhline(result["regions"][region]["constant_value"], color="#999999", ls="--", lw=.7, label="Training mean")
        axis.set(title=region+": "+result["regions"][region]["status"].replace("_", " "),
                 xlabel="s = measured x (m)", ylabel="Magnetic magnitude (reference units)")
        axis.legend(fontsize=6.8)
        axis.grid(alpha=.2)
    axis = axes.flat[-1]
    axis.scatter(saved["whole_band_train_s"], np.zeros(len(saved["whole_band_train_s"])), s=6, label="Training band positions")
    for edge in result["protocol"]["local_edges_metres"]:
        axis.axvline(edge, color="#777777", lw=.7)
    axis.set(title="All local regions fixed from training geometry", xlabel="x (m)", yticks=[])
    axis.legend(fontsize=8)
    fig.suptitle("Real magnetic measurements: kernel → derivative equation → analytic function", fontsize=14)
    fig.savefig(OUTPUT/"comparison.png", dpi=160)
    fig.savefig(OUTPUT/"comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()

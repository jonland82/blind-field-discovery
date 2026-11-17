"""Run the predeclared synthetic hidden-wire workflow and sensor comparison."""
from pathlib import Path
import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from learner import Expression, discover, fit_teacher, mse
from world import polynomial_field, sample_inputs, wire_field, wire_jacobian


ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
REPO = ROOT.parents[1]
SEEDS = (20261116, 20261117, 20261118)
CASES = (
    ("noiseless", 480, 0., (-np.pi, np.pi)),
    ("main", 480, .01, (-np.pi, np.pi)),
    ("noisy", 480, .05, (-np.pi, np.pi)),
    ("sparse", 80, .01, (-np.pi, np.pi)),
    ("limited_angle", 480, .01, (-np.pi/6, np.pi/6)),
)


def relative(predicted, actual):
    return float(np.linalg.norm(predicted-actual)/np.linalg.norm(actual))


def scores(predicted, actual, observed):
    finite = np.isfinite(predicted).all(axis=1)
    if not finite.all():
        return dict(nonfinite_rows=int((~finite).sum()), oracle_relative_rmse=None,
                    oracle_vector_rmse=None, measured_vector_rmse=None)
    return dict(nonfinite_rows=0, oracle_relative_rmse=relative(predicted, actual),
                oracle_vector_rmse=float(np.sqrt(mse(predicted, actual))),
                measured_vector_rmse=float(np.sqrt(mse(predicted, observed))))


def diagnostics(model, query):
    jacobian = model.jacobian(query)
    oracle = wire_jacobian(query)
    denominator = np.linalg.norm(oracle[:, :, :2])
    divergence = jacobian[:, 0, 0]+jacobian[:, 1, 1]
    curl = jacobian[:, 1, 0]-jacobian[:, 0, 1]
    loops = []
    theta = np.linspace(0, 2*np.pi, 1024, endpoint=False)
    for radius in (.15, .5, 1., 1.8, 3.):
        points = np.column_stack((radius*np.cos(theta), radius*np.sin(theta),
                                  np.full(len(theta), 2.)))
        prediction = model.predict(points)
        tangent = radius*np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(len(theta))))
        circulation = float(np.sum(prediction*tangent)*2*np.pi/len(theta))
        loops.append(dict(radius_metres=radius, current_amperes=2.,
                          circulation_microtesla_metres=circulation,
                          oracle_circulation=float(2*np.pi*.2*2.)))
    return dict(spatial_derivative_relative_error=relative(jacobian[:, :, :2], oracle[:, :, :2]),
                planar_divergence_relative_norm=float(np.linalg.norm(divergence)/denominator),
                planar_curl_relative_norm=float(np.linalg.norm(curl)/denominator),
                circulation=loops)


def make_observations(seed, n, noise, angle):
    rng = np.random.default_rng(seed)
    train_x = sample_inputs(rng, n, angle=angle)
    validation_x = sample_inputs(rng, 180, angle=angle)
    clean = wire_field(train_x)
    sigma = noise*np.sqrt(np.mean(clean**2))
    return dict(train_x=train_x, train_y=clean+rng.normal(0, sigma, clean.shape),
                validation_x=validation_x,
                validation_y=wire_field(validation_x)+rng.normal(0, sigma, (180, 3))), float(sigma)


def make_tests(seed, angle, sigma):
    rng = np.random.default_rng(seed+10000)
    tests = {}
    specifications = dict(in_domain=dict(angle=angle), full_angles={},
                          inward=dict(radius=(.10, .22)), outward=dict(radius=(2.3, 4.)),
                          new_current=dict(current=(5., 8.)), moved_wire={})
    for name, kwargs in specifications.items():
        x = sample_inputs(rng, 800, **kwargs)
        if name == "moved_wire":
            # Fixed rejection sampling, not a favorable held-out window.
            while np.any(np.linalg.norm(x[:, :2]-[.35, -.25], axis=1) < .15):
                bad = np.linalg.norm(x[:, :2]-[.35, -.25], axis=1) < .15
                x[bad] = sample_inputs(rng, int(bad.sum()))
            actual = wire_field(x, center=(.35, -.25))
        else:
            actual = wire_field(x)
        tests[name] = (x, actual, actual+rng.normal(0, sigma, actual.shape))
    return tests


def run_case(name, count, noise, angle, seed, saved):
    start = time.perf_counter()
    observations, sigma = make_observations(seed, count, noise, angle)
    x, y, xv, yv = [observations[key] for key in
                    ("train_x", "train_y", "validation_x", "validation_y")]
    teacher, teacher_audit = fit_teacher(x, y, xv, yv)
    teacher_targets = teacher.predict(x)
    student, student_audit = discover(x, teacher_targets, xv, yv)
    direct, direct_audit = discover(x, y, xv, yv)
    # All choices above are frozen. Only now construct/open independent tests.
    tests = make_tests(seed, angle, sigma)
    constant = y.mean(axis=0)
    models = dict(kernel=teacher, distilled=student, direct=direct)
    evaluations = {}
    prefix = f"{name}_{seed}_"
    for key, value in observations.items():
        saved[prefix+key] = value
    saved[prefix+"kernel_training"] = teacher_targets
    saved[prefix+"distilled_training"] = student.predict(x)
    for domain, (query, actual, observed) in tests.items():
        predictions = {key: model.predict(query) for key, model in models.items()}
        predictions["constant"] = np.tile(constant, (len(query), 1))
        evaluations[domain] = {key: scores(value, actual, observed) for key, value in predictions.items()}
        for key, value in dict(x=query, oracle=actual, observed=observed, **predictions).items():
            saved[prefix+domain+"_"+key] = value
    stage = dict(kernel_training_measured_rmse=float(np.sqrt(mse(teacher_targets, y))),
                 kernel_validation_measured_rmse=teacher_audit["validation_vector_rmse"],
                 distilled_validation_measured_rmse=student_audit["validation_vector_rmse"],
                 direct_validation_measured_rmse=direct_audit["validation_vector_rmse"],
                 teacher_to_expression_training_relative_error=relative(student.predict(x), teacher_targets),
                 teacher_to_expression_test_relative_error=relative(student.predict(tests["in_domain"][0]),
                                                                      teacher.predict(tests["in_domain"][0])))
    diag = {key: diagnostics(model, tests["in_domain"][0]) for key, model in models.items()}
    # Fixed diagnostic cuts generated after selection. Do not use in fitting.
    radius = np.linspace(.1, 4., 400)
    cut = np.column_stack((radius, np.zeros_like(radius), np.full_like(radius, 2.)))
    saved[prefix+"cut_x"], saved[prefix+"cut_oracle"] = cut, wire_field(cut)
    for key, model in models.items():
        saved[prefix+"cut_"+key] = model.predict(cut)
    row = dict(case=name, seed=seed, training_count=count, validation_count=len(xv),
               noise_fraction=noise, noise_std_microtesla=sigma, angle_range=list(angle),
               teacher=teacher_audit, distilled=student_audit, direct=direct_audit,
               stages=stage, evaluations=evaluations, diagnostics=diag)
    print(f"{name:14s} {seed}: kernel {100*evaluations['in_domain']['kernel']['oracle_relative_rmse']:.3f}%, "
          f"distilled {100*evaluations['in_domain']['distilled']['oracle_relative_rmse']:.3f}%, "
          f"direct {100*evaluations['in_domain']['direct']['oracle_relative_rmse']:.3f}% "
          f"({time.perf_counter()-start:.1f}s)", flush=True)
    print("  "+"; ".join(student.formula()), flush=True)
    return row


def sensor_comparison():
    directory = REPO/"experiments/magnetic_mapping"
    import sys
    sys.path.insert(0, str(directory))
    from dipole_models import DipoleField
    from models import FieldMap
    result = json.loads((directory/"dipole_field/results.json").read_text())
    original = json.loads((directory/"results/results.json").read_text())
    archive = np.load(directory/"dipole_field/predictions.npz")
    x, y = archive["training_xy"], archive["training_actual"]
    # Recover the frozen teacher parameters from the existing report, not a new search.
    selected = original["model_selection"]["unrestricted"]["selected"]
    teacher = FieldMap("unrestricted", selected["length_metres"], selected["ridge"]).fit(x, y)
    field = result["field"]
    student = DipoleField(field["sources_metres"], field["background"], field["effective_moments"])
    control = result["direct_control"]
    direct_model = DipoleField(control["sources_metres"], control["background"], control["effective_moments"])
    xt = np.concatenate([archive[f"invensense_{trial}_xy"] for trial in (4, 5)])
    yt = np.concatenate([archive[f"invensense_{trial}_actual"] for trial in (4, 5)])
    kt = np.concatenate([archive[f"invensense_{trial}_kernel"] for trial in (4, 5)])
    np.testing.assert_allclose(teacher.predict(xt), kt, atol=1e-9)
    dx = np.concatenate([archive[f"invensense_{trial}_dipoles"] for trial in (4, 5)])
    direct = np.concatenate([archive[f"invensense_{trial}_direct_dipoles"] for trial in (4, 5)])
    xv, yv = archive["validation_xy"], archive["validation_actual"]
    metrics = dict(training_count=len(x), validation_count=len(xv), stable_test_count=len(xt),
                   kernel_training_measured_rmse=float(np.sqrt(mse(teacher.predict(x), y))),
                   kernel_validation_measured_rmse=float(np.sqrt(mse(teacher.predict(xv), yv))),
                   distilled_validation_measured_rmse=float(np.sqrt(mse(student.predict(xv), yv))),
                   direct_validation_measured_rmse=float(np.sqrt(mse(direct_model.predict(xv), yv))),
                   teacher_to_expression_training_relative_error=relative(student.predict(x), teacher.predict(x)),
                   teacher_to_expression_test_relative_error=relative(dx, kt),
                   stable={key: dict(vector_rmse=float(np.sqrt(mse(value, yt))),
                                     observed_relative_rmse=relative(value, yt))
                           for key, value in dict(kernel=kt, distilled=dx, direct=direct).items()},
                   changed={key: result["pooled"]["changed"]["invensense"][name]["vector_rmse"]
                            for key, name in dict(kernel="kernel", distilled="dipoles", direct="direct_dipoles").items()},
                   normalization="Aalto: observed vector RMS; synthetic: clean oracle vector RMS",
                   status="Frozen exploratory Aalto results, not new sensor trials.")
    paths = ("dipole_field/results.json", "dipole_field/predictions.npz", "results/results.json")
    return metrics, {str((directory/p).relative_to(REPO)).replace("\\", "/"):
                     hashlib.sha256((directory/p).read_bytes()).hexdigest() for p in paths}


def negative_control():
    rng = np.random.default_rng(20261119)
    x, xv = sample_inputs(rng, 240), sample_inputs(rng, 140)
    y, yv = polynomial_field(x), polynomial_field(xv)
    teacher, _ = fit_teacher(x, y, xv, yv)
    student, audit = discover(x, teacher.predict(x), xv, yv)
    direct, direct_audit = discover(x, y, xv, yv)
    query = sample_inputs(rng, 800, radius=(2.3, 4.), current=(5., 8.))
    return dict(distilled=audit, direct=direct_audit,
                outward_and_current_error={key: relative(model.predict(query), polynomial_field(query))
                                          for key, model in dict(kernel=teacher, distilled=student, direct=direct).items()})


def write_report(result):
    rows = result["runs"]
    lines = ["# Hidden-wire experiment: results", "", "Synthetic workflow verification. "
             "Read the [fixed protocol](../PROTOCOL.md) and [theory and comparison](../README.md).", "",
             "Errors below are percentages of clean vector-field RMS, averaged over all three fixed seeds. "
             "Each fit was selected on noisy validation observations before test scoring.", "",
             "| Case | Kernel: same domain | Kernel to expression | Direct expression | Distilled: full angles | Distilled: outward | Distilled: new currents |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case, *_ in CASES:
        group = [row for row in rows if row["case"] == case]
        def average(domain, model):
            return 100*np.mean([r["evaluations"][domain][model]["oracle_relative_rmse"] for r in group])
        lines.append(f"| {case} | {average('in_domain', 'kernel'):.4f}% | "
                     f"{average('in_domain', 'distilled'):.4f}% | {average('in_domain', 'direct'):.4f}% | "
                     f"{average('full_angles', 'distilled'):.4f}% | {average('outward', 'distilled'):.4f}% | "
                     f"{average('new_current', 'distilled'):.4f}% |")
    lines += ["", "## Every recovered expression", "", "x,y in metres, I in amperes, B in microtesla. "
              "These are displayed to nine significant figures; scores use full saved precision. "
              "No rounding to the generating law enters predictions."]
    for row in rows:
        lines += ["", f"### {row['case']}, seed {row['seed']}", "",
                  f"Selected {row['distilled']['selected']['kind']}; "
                  f"{row['distilled']['selected']['coefficient_count']} free nonzero coefficients.", "", "```text"]
        lines += [f"B{axis} = {formula}" for axis, formula in
                  zip("xyz", row["distilled"]["selected"]["formula"])]
        lines += ["```"]
    main = next(row for row in rows if row["case"] == "main" and row["seed"] == SEEDS[0])
    sensor = result["sensor"]
    stage = main["stages"]
    lines += ["", "## Matched workflow audit", "",
              "Synthetic entries use the fixed first main seed, not a selected best seed. "
              "Aalto entries reproduce the frozen reference-instrument results. Absolute errors "
              "have different units and noise levels; this table compares stages, not difficulty.", "",
              "| Stage | Hidden wire | Measured Aalto sensor workflow |",
              "| --- | --- | --- |",
              f"| Training / validation / stable test observations | {main['training_count']} / {main['validation_count']} / 800 | "
              f"{sensor['training_count']} / {sensor['validation_count']} / {sensor['stable_test_count']} |",
              "| Kernel inputs / outputs | x,y,I / Bx,By,Bz | x,y / Bx,By,Bz; upstream orientation correction |",
              "| Explicit representation | General polynomial and rational search | Supplied 3D magnetic dipole response, assumed source sheet |",
              "| Expression-fit targets | Teacher at training positions | Teacher at training positions |",
              f"| Kernel training measurement RMSE | {stage['kernel_training_measured_rmse']:.6f} microtesla | "
              f"{sensor['kernel_training_measured_rmse']:.6f} reference units |",
              f"| Kernel validation measurement RMSE | {stage['kernel_validation_measured_rmse']:.6f} microtesla | "
              f"{sensor['kernel_validation_measured_rmse']:.6f} reference units |",
              f"| Extracted field validation measurement RMSE | {stage['distilled_validation_measured_rmse']:.6f} microtesla | "
              f"{sensor['distilled_validation_measured_rmse']:.6f} reference units |",
              f"| Direct field validation measurement RMSE | {stage['direct_validation_measured_rmse']:.6f} microtesla | "
              f"{sensor['direct_validation_measured_rmse']:.6f} reference units |",
              f"| Expression vs teacher at training inputs | {100*stage['teacher_to_expression_training_relative_error']:.3f}% | "
              f"{100*sensor['teacher_to_expression_training_relative_error']:.3f}% |",
              f"| Expression vs teacher at stable test inputs | {100*stage['teacher_to_expression_test_relative_error']:.3f}% | "
              f"{100*sensor['teacher_to_expression_test_relative_error']:.3f}% |"]
    for key, label in (("kernel", "Kernel"), ("distilled", "Extracted field"), ("direct", "Direct fit")):
        synthetic = main["evaluations"]["in_domain"][key]["oracle_relative_rmse"]
        actual = sensor["stable"][key]["observed_relative_rmse"]
        lines.append(f"| {label}: stable test normalized error | {100*synthetic:.3f}% of clean field RMS | "
                     f"{100*actual:.3f}% of observed field RMS |")
    lines += ["| Analytic truth available? | Yes, used only in generation and final scoring | No room-field formula or true source locations |",
              "| Changed environment | Wire moves; held-out geometry excluded from inputs | Recorded toolbox/shelf changes; source geometry unavailable |",
              "| Independent physical-structure discovery? | Sparse algebra selected within a supplied grammar | Magnetic structure supplied by dipoles |",
              "", "The teacher-to-expression discrepancy need not shrink on new positions: the expression "
              "can disagree with the teacher and agree better with observations. The reconstruction is "
              "therefore an additional modeling assumption, not an exact symbolic rewrite of the kernel."]
    lines += ["", "## Fixed diagnostics for the first main seed", "",
              "Post-fit model diagnostics, not independently observed derivatives.", "",
              "| Model | Spatial derivative error | Planar divergence / derivative norm | Planar curl / derivative norm |",
              "| --- | ---: | ---: | ---: |"]
    for key, d in main["diagnostics"].items():
        lines.append(f"| {key} | {d['spatial_derivative_relative_error']:.6g} | "
                     f"{d['planar_divergence_relative_norm']:.6g} | {d['planar_curl_relative_norm']:.6g} |")
    lines += ["", "## Changed environment", "",
              "The wire moves without updating the inputs or retraining. Each entry averages all main-case seeds.", "",
              "| Model | Error against moved wire |", "| --- | ---: |"]
    for model in ("kernel", "distilled", "direct"):
        mean = np.mean([row["evaluations"]["moved_wire"][model]["oracle_relative_rmse"]
                        for row in rows if row["case"] == "main"])
        lines.append(f"| {model} | {100*mean:.3f}% |")
    control = result["negative_control"]
    lines += ["", "## Nonmagnetic control", "", f"Distilled selection: {control['distilled']['selected']['kind']}, "
              "with constant denominator (an ordinary polynomial). "
              f"Outward-radius plus unseen-current relative error: {100*control['outward_and_current_error']['distilled']:.6g}%. "
              "See the JSON for the full expressions and direct-fit control.", "",
              "[Plots](workflow.png), [all fits and scores](results.json), [sample and prediction arrays](predictions.npz)."]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def plot(result, saved):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    prefix = f"main_{SEEDS[0]}_"
    cut = saved[prefix+"cut_x"][:, 0]
    for key, label in (("oracle", "Hidden field"), ("kernel", "Kernel"),
                       ("distilled", "Kernel to equation"), ("direct", "Direct equation")):
        color = dict(oracle="black", kernel="C0", distilled="C1", direct="C2")[key]
        axes[0, 0].plot(cut, saved[prefix+"cut_"+key][:, 1], label=label, color=color,
                        lw=2 if key == "oracle" else 1.2, ls="--" if key == "direct" else "-")
    axes[0, 0].axvspan(.25, 2., alpha=.08, color="black", label="Observed radius range")
    axes[0, 0].set(xlabel="Radius along +x axis (m), I=2 A", ylabel="By (microtesla)",
                    title="Shape: first main seed, fixed cut")
    axes[0, 0].legend(fontsize=8)
    domains = ("in_domain", "inward", "outward", "new_current", "moved_wire")
    main = [row for row in result["runs"] if row["case"] == "main"]
    for key, label in (("kernel", "Kernel"), ("distilled", "Kernel to equation"), ("direct", "Direct equation")):
        values = np.array([[100*r["evaluations"][d][key]["oracle_relative_rmse"] for d in domains] for r in main])
        axes[0, 1].errorbar(range(len(domains)), values.mean(axis=0), yerr=values.std(axis=0),
                           marker="o", capsize=3, label=label)
    axes[0, 1].set_xticks(range(len(domains)), ["Same", "Inward", "Outward", "Current", "Moved"], rotation=15)
    axes[0, 1].set(yscale="log", ylabel="Field error (% of clean vector RMS)",
                    title="Main case: mean and SD across three seeds")
    axes[0, 1].legend(fontsize=8)
    for key, label in (("kernel", "Kernel"), ("distilled", "Kernel to equation"), ("direct", "Direct equation")):
        means, deviations = [], []
        for name, *_ in CASES:
            v = [100*r["evaluations"]["full_angles"][key]["oracle_relative_rmse"]
                 for r in result["runs"] if r["case"] == name]
            means.append(max(float(np.mean(v)), 1e-3))
            deviations.append(float(np.std(v)))
        axes[1, 0].errorbar(range(len(CASES)), means, yerr=deviations, marker="o", capsize=3, label=label)
    axes[1, 0].set_xticks(range(len(CASES)), ["No noise", "1% noise", "5% noise", "80 samples", "Narrow angles"], rotation=20)
    axes[1, 0].set(yscale="log", ylabel="Full-angle field error (%)", title="All fixed observation regimes")
    axes[1, 0].set_ylim(bottom=5e-4)
    axes[1, 0].text(.02, .03, "Noiseless direct fit <0.001% (plotted at floor)",
                    transform=axes[1, 0].transAxes, fontsize=8)
    diag = next(r for r in main if r["seed"] == SEEDS[0])["diagnostics"]
    for key, label in (("kernel", "Kernel"), ("distilled", "Kernel to equation"), ("direct", "Direct equation")):
        loops = diag[key]["circulation"]
        axes[1, 1].plot([d["radius_metres"] for d in loops],
                        [d["circulation_microtesla_metres"] for d in loops], "o-", label=label)
    axes[1, 1].axhline(2*np.pi*.2*2., color="black", ls=":", label="Hidden circulation")
    axes[1, 1].set(xlabel="Loop radius (m), I=2 A", ylabel="Circulation (microtesla m)",
                    title="Post-fit integral diagnostic")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=.2)
    fig.savefig(OUT/"workflow.png", dpi=170)
    fig.savefig(OUT/"workflow.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    saved, runs = {}, []
    for case in CASES:
        for seed in SEEDS:
            runs.append(run_case(*case, seed, saved))
    sensor, input_hashes = sensor_comparison()
    result = dict(runs=runs, sensor=sensor, negative_control=negative_control(),
                  protocol_sha256=hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest(),
                  input_sha256=input_hashes,
                  implementation_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                         for name in ("learner.py", "world.py", "run_experiment.py")})
    np.savez_compressed(OUT/"predictions.npz", **saved)
    (OUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_report(result)
    plot(result, saved)
    print("Saved all cases, controls, sensor comparison and figures:", OUT, flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

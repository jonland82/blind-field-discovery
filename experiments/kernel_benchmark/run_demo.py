"""Synthetic workflow verification, not evidence from physical measurements.

The generator knows the analytic field. The learner receives only sample arrays.
Run from the workspace root: python experiments/kernel_benchmark/run_demo.py
"""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import erf

ROOT = Path(__file__).resolve().parent
TERMS = ("1", "B", "J", "x", "x*B", "x*J", "B^2", "B*J", "J^2")


def hidden_functions(x):
    """Oracle for data generation and final scoring only; dimensionless units."""
    field_scale, length_scale, source_scale = 1.7, .9, .8
    shape = 1 / np.cosh(x / length_scale)**2
    return (field_scale * np.tanh(x / length_scale), source_scale * shape,
            field_scale / length_scale * shape)


class KernelFit:
    def __init__(self, length, ridge):
        self.length, self.ridge = length, ridge

    def fit(self, x, y):
        self.x, self.mean = x.copy(), y.mean(axis=0)
        kernel = self.kernel(x)
        kernel.flat[::len(x)+1] += self.ridge
        self.weights = cho_solve(cho_factor(kernel, lower=True), y - self.mean)
        return self

    def kernel(self, x):
        return np.exp(-.5 * ((np.asarray(x)[:, None] - self.x) / self.length)**2)

    def predict(self, x):
        return self.mean + self.kernel(x) @ self.weights

    def derivative(self, x):
        delta = np.asarray(x)[:, None] - self.x
        return (-delta / self.length**2 * self.kernel(x)) @ self.weights

    def integral(self, x, anchor):
        """Exact integral of the fitted Gaussian representation, per channel."""
        scaled = (np.asarray(x)[:, None] - self.x) / (np.sqrt(2)*self.length)
        start = (anchor - self.x) / (np.sqrt(2)*self.length)
        integrated_kernel = self.length*np.sqrt(np.pi/2) * (erf(scaled)-erf(start))
        return ((np.asarray(x)-anchor)[:, None]*self.mean
                + integrated_kernel @ self.weights)


def library(x, b, j):
    return np.column_stack((np.ones_like(x), b, j, x, x*b, x*j, b*b, b*j, j*j))


def learn(samples):
    """No access to hidden_functions, its parameters, or oracle derivatives."""
    x, y = samples["train_x"], samples["train_y"]
    vx, vy = samples["validation_x"], samples["validation_y"]
    scale = np.maximum(y.std(axis=0), 1e-12)
    candidates = []
    for length in (.15, .3, .5, .8, 1.2, 2.):
        for ridge in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
            fit = KernelFit(length, ridge).fit(x, y)
            score = float(np.mean(((fit.predict(vx) - vy) / scale)**2))
            candidates.append((score, fit))
    score, fit = min(candidates, key=lambda item: item[0])

    # Independent observed source values accompany discovery and law validation.
    dx, dj = samples["discovery_x"], samples["discovery_j"]
    lx, lj = samples["law_validation_x"], samples["law_validation_j"]
    design = library(dx, fit.predict(dx)[:, 0], dj)
    target = fit.derivative(dx)[:, 0]
    check = library(lx, fit.predict(lx)[:, 0], lj)
    check_target = fit.derivative(lx)[:, 0]
    equations = []
    for size in (1, 2, 3):
        for support in combinations(range(len(TERMS)), size):
            coefficients = np.linalg.lstsq(design[:, support], target, rcond=None)[0]
            residual = check[:, support] @ coefficients - check_target
            error = float(np.linalg.norm(residual) / np.linalg.norm(check_target))
            equations.append((size, error, support, coefficients))
    # Predeclared numerical tolerance, not a physical confidence interval.
    eligible = [row for row in equations if row[1] <= .01]
    selected = min(eligible, key=lambda row: (row[0], row[1])) if eligible else min(
        equations, key=lambda row: (row[1], row[0]))
    return fit, selected, score, bool(eligible)


def relative_rmse(predicted, actual):
    return float(np.linalg.norm(predicted - actual) / np.linalg.norm(actual))


def recover_symbolic(fit, samples, coupling_equation):
    """Discover a field-only polynomial ODE, then solve its bounded branch.

    No oracle calls or tanh/sech regression candidates. The analytic solution
    rule for a selected quadratic ODE is supplied mathematical machinery.
    """
    x, xv = samples["discovery_x"], samples["law_validation_x"]
    b, bv = fit.predict(x)[:, 0], fit.predict(xv)[:, 0]
    derivative, derivative_v = fit.derivative(x)[:, 0], fit.derivative(xv)[:, 0]
    design, check = np.vander(b, 5, increasing=True), np.vander(bv, 5, increasing=True)
    equations = []
    for size in (1, 2, 3):
        for support in combinations(range(5), size):
            coefficients = np.linalg.lstsq(design[:, support], derivative, rcond=None)[0]
            residual = check[:, support] @ coefficients - derivative_v
            error = float(np.linalg.norm(residual)/np.linalg.norm(derivative_v))
            equations.append((size, error, support, coefficients))
    eligible = [row for row in equations if row[1] <= .01]
    selected = min(eligible, key=lambda row: (row[0], row[1])) if eligible else min(
        equations, key=lambda row: (row[1], row[0]))
    _, error, support, coefficients = selected
    result = {"candidate_powers": list(range(5)), "maximum_support": 3,
              "selected_powers": list(support), "coefficients": coefficients.tolist(),
              "validation_relative_residual": error,
              "met_one_percent_tolerance": bool(eligible),
              "closed_form_recovered": False}
    if not eligible or support != (0, 2):
        result["status"] = "Selected ODE is not the accepted constant-minus-square form."
        return result
    p, q = float(coefficients[0]), -float(coefficients[1])
    if p <= 0 or q <= 0:
        result["status"] = "Signs do not support the bounded tanh branch."
        return result
    amplitude, length = np.sqrt(p/q), 1/np.sqrt(p*q)
    reference_x = float(np.median(samples["train_x"]))
    reference_b = float(fit.predict(np.array([reference_x]))[0, 0])
    if abs(reference_b) >= amplitude:
        result["status"] = "Reference value does not select the bounded tanh branch."
        return result
    center = reference_x-length*np.arctanh(reference_b/amplitude)
    result.update({"closed_form_recovered": True, "status": "Bounded tanh branch selected by an interior kernel reference value.",
                   "p": p, "q": q, "amplitude": float(amplitude),
                   "length": float(length), "center": float(center),
                   "reference_x": reference_x, "reference_kernel_b": reference_b})
    if coupling_equation[2] == (2,):
        a = float(coupling_equation[3][0])
        result["coupling_a"] = a
        result["source_amplitude"] = p/a
    return result


def evaluate_symbolic(x, symbolic):
    """Evaluate the solved equation, rather than a regression with tanh features."""
    if not symbolic["closed_form_recovered"]:
        raise ValueError("No accepted closed form to evaluate")
    z = (np.asarray(x)-symbolic["center"])/symbolic["length"]
    b = symbolic["amplitude"]*np.tanh(z)
    derivative = symbolic["p"]/np.cosh(z)**2
    j = derivative/symbolic["coupling_a"] if "coupling_a" in symbolic else None
    return b, j, derivative


def make_samples(noise, seed):
    rng = np.random.default_rng(seed)
    data = {}
    # Fixed physical scales normalize channel noise; neither enters learn().
    for name, count in (("train", 80), ("validation", 60)):
        x = np.sort(rng.uniform(-3, 3, count))
        b, j, _ = hidden_functions(x)
        data[name+"_x"] = x
        data[name+"_y"] = np.column_stack((b, j)) + rng.normal(
            size=(count, 2)) * noise * np.array([1.7, .8])
    for name in ("discovery", "law_validation"):
        x = np.sort(rng.uniform(-2.8, 2.8, 100))
        data[name+"_x"] = x
        data[name+"_j"] = hidden_functions(x)[1] + rng.normal(size=len(x)) * noise * .8
    data["anchor_b"] = float(hidden_functions(np.array([-3.]))[0][0] + rng.normal()*noise*1.7)
    return data


def run_case(noise, seed):
    samples = make_samples(noise, seed)
    fit, equation, score, accepted = learn(samples)
    symbolic = recover_symbolic(fit, samples, equation)
    size, law_error, support, coefficients = equation
    # Oracle is revealed only AFTER kernel, both equations and branch selection.
    x = np.linspace(-3, 3, 901)
    b, j, derivative = hidden_functions(x)
    predicted = fit.predict(x)
    predicted_derivative = fit.derivative(x)[:, 0]

    def rhs(position, state):
        source = fit.predict(np.array([position]))[0, 1]
        features = library(np.array([position]), np.array([state[0]]), np.array([source]))
        return [float(features[0, list(support)] @ coefficients)]

    if support == (2,):
        # Integrate the selected B' = a J law in closed kernel form.
        integrated = samples["anchor_b"] + coefficients[0]*fit.integral(x, -3.)[:, 1]
    else:
        solution = solve_ivp(rhs, (-3., 3.), [samples["anchor_b"]], t_eval=x,
                             rtol=1e-9, atol=1e-11)
        if not solution.success:
            raise RuntimeError(solution.message)
        integrated = solution.y[0]
    outside = np.linspace(3.05, 4., 150)
    outside_b, _, outside_derivative = hidden_functions(outside)
    expected_a = 1.7 / (.9 * .8)
    equation_values = library(x, predicted[:, 0], j)[:, support] @ coefficients
    report = {
        "synthetic_data": True, "noise_fraction_of_fixed_channel_scales": noise,
        "seed": seed, "train_count": 80, "kernel_validation_count": 60,
        "equation_fit_count": 100, "equation_validation_count": 100,
        "kernel_length": fit.length, "kernel_ridge": fit.ridge,
        "kernel_validation_standardized_mse": score,
        "equation_terms": [TERMS[i] for i in support],
        "equation_coefficients": coefficients.tolist(),
        "equation_validation_relative_residual": law_error,
        "equation_met_predeclared_one_percent_tolerance": accepted,
        "oracle_a_revealed_after_fit": expected_a,
        "coefficient_relative_error": abs(float(coefficients[0])/expected_a-1)
            if support == (2,) else None,
        "field_relative_rmse": relative_rmse(predicted[:, 0], b),
        "source_relative_rmse": relative_rmse(predicted[:, 1], j),
        "derivative_relative_rmse": relative_rmse(predicted_derivative, derivative),
        "equation_test_relative_residual": relative_rmse(equation_values, predicted_derivative),
        "integrated_law_field_relative_rmse": relative_rmse(integrated, b),
        "extrapolated_field_relative_rmse": relative_rmse(fit.predict(outside)[:, 0], outside_b),
        "extrapolated_derivative_relative_rmse": relative_rmse(fit.derivative(outside)[:, 0], outside_derivative),
        "test_scope": "Dense unseen positions within the sampled interval; same analytic generator.",
        "symbolic_recovery": symbolic,
    }
    arrays = {**samples, "test_x": x, "true_b": b, "true_j": j,
              "true_derivative": derivative, "kernel_b": predicted[:, 0],
              "kernel_j": predicted[:, 1], "kernel_derivative": predicted_derivative,
              "integrated_law_b": integrated, "centers": fit.x, "weights": fit.weights,
              "mean": fit.mean, "length": fit.length, "ridge": fit.ridge,
              "equation_support": support, "equation_coefficients": coefficients}
    if symbolic["closed_form_recovered"]:
        symbolic_b, symbolic_j, symbolic_derivative = evaluate_symbolic(x, symbolic)
        extrap_b, extrap_j, extrap_derivative = evaluate_symbolic(outside, symbolic)
        outside_j = hidden_functions(outside)[1]
        symbolic.update({
            "field_relative_rmse": relative_rmse(symbolic_b, b),
            "source_relative_rmse": relative_rmse(symbolic_j, j) if symbolic_j is not None else None,
            "derivative_relative_rmse": relative_rmse(symbolic_derivative, derivative),
            "extrapolated_field_relative_rmse": relative_rmse(extrap_b, outside_b),
            "extrapolated_source_relative_rmse": relative_rmse(extrap_j, outside_j) if extrap_j is not None else None,
            "extrapolated_derivative_relative_rmse": relative_rmse(extrap_derivative, outside_derivative),
            "oracle_p_after_fit": 1.7/.9, "oracle_q_after_fit": 1/(1.7*.9),
            "amplitude_relative_error": abs(symbolic["amplitude"]/1.7-1),
            "length_relative_error": abs(symbolic["length"]/.9-1),
            "center_absolute_error": abs(symbolic["center"]),
            "source_amplitude_relative_error": abs(symbolic["source_amplitude"]/.8-1)
                if "source_amplitude" in symbolic else None})
        arrays.update({"symbolic_b": symbolic_b, "symbolic_derivative": symbolic_derivative,
                       "extrapolation_x": outside, "extrapolation_true_b": outside_b,
                       "extrapolation_true_j": outside_j,
                       "extrapolation_kernel_b": fit.predict(outside)[:, 0],
                       "extrapolation_kernel_j": fit.predict(outside)[:, 1],
                       "extrapolation_symbolic_b": extrap_b})
        if symbolic_j is not None:
            arrays["symbolic_j"] = symbolic_j
            arrays["extrapolation_symbolic_j"] = extrap_j
    return report, arrays


def main():
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    reports, arrays = {}, {}
    for name, noise in (("noiseless", 0.), ("noisy", .001)):
        reports[name], arrays[name] = run_case(noise, 20261116)
        np.savez_compressed(output / (name+".npz"), **arrays[name])
    report = {"purpose": "Verify the numerical kernel-to-equation workflow on known synthetic functions.",
              "physical_discovery_claim": False,
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": reports}
    (output / "results.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for col, name in enumerate(reports):
        a, r = arrays[name], reports[name]
        axes[0, col].plot(a["test_x"], a["true_b"], label="Hidden analytic field", lw=2)
        axes[0, col].plot(a["test_x"], a["kernel_b"], "--", label="Kernel fit")
        axes[0, col].plot(a["test_x"], a["integrated_law_b"], ":", label="Recovered equation")
        if "symbolic_b" in a:
            axes[0, col].plot(a["test_x"], a["symbolic_b"], "-.", label="Solved tanh form", lw=1.2)
        axes[0, col].scatter(a["train_x"], a["train_y"][:, 0], s=7, alpha=.4, label="Training samples")
        axes[0, col].set_title(name.capitalize())
        axes[0, col].set_ylabel("B (normalized units)")
        axes[1, col].plot(a["test_x"], a["true_derivative"], label="Hidden analytic derivative", lw=2)
        axes[1, col].plot(a["test_x"], a["kernel_derivative"], "--", label="Kernel derivative")
        axes[1, col].plot(a["test_x"], r["equation_coefficients"][0]*a["true_j"], ":", label="Recovered a J")
        axes[1, col].set_xlabel("x (normalized units)")
        axes[1, col].set_ylabel("dB/dx")
    for axis in axes.flat:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# Synthetic kernel-to-equation verification", "",
             "These results verify a numerical workflow using known artificial functions. They are not physical measurements.", "",
             "The oracle is B(x) = 1.7 tanh(x/0.9), J(x) = 0.8 sech²(x/0.9). Their exact relation is B′ = (85/36) J.", "",
             "The learner receives only sample arrays, a Gaussian kernel family, and a generic candidate library. Oracle formulas and derivatives are used only for generation and final scoring.", "",
             "| Metric | Noiseless | 0.1% scale noise |", "| --- | ---: | ---: |"]
    metrics = [("Field relative RMS error", "field_relative_rmse"),
               ("Derivative relative RMS error", "derivative_relative_rmse"),
               ("Coefficient relative error", "coefficient_relative_error"),
               ("Integrated equation field relative RMS error", "integrated_law_field_relative_rmse"),
               ("Extrapolated field relative RMS error (x = 3.05–4)", "extrapolated_field_relative_rmse"),
               ("Extrapolated derivative relative RMS error (x = 3.05–4)", "extrapolated_derivative_relative_rmse")]
    for label, key in metrics:
        values = ["n/a" if reports[n][key] is None else f"{100*reports[n][key]:.4g}%" for n in reports]
        lines.append(f"| {label} | {' | '.join(values)} |")
    for name, r in reports.items():
        formula = " + ".join(f"{coefficient:.9g} ({term})" for term, coefficient in zip(
            r["equation_terms"], r["equation_coefficients"]))
        lines.extend(["", f"**{name.capitalize()} recovered equation:** B′ = {formula}.",
                      f"Selected Gaussian length {r['kernel_length']}, ridge {r['kernel_ridge']:g}."])
    lines.extend(["", "## Symbolic recovery of the original function family", "",
        "A second search excludes J and x, using the autonomous field-only library {1, B, B², B³, B⁴}, with the same support-size rule and 1% residual tolerance. This supplies a search class, not the equation coefficients. No tanh or sech functions occur in the regression library.", "",
        "Both runs select B′ = p − q B². Solving this equation on its bounded branch gives B(x) = A tanh((x − x_c)/L), with A = sqrt(p/q) and L = 1/sqrt(pq). One interior kernel value at the median training position fixes x_c. Together with B′ = a J, this gives J(x) = (p/a) sech²((x − x_c)/L).", "",
        "The ODE solution rule is supplied analytic mathematics. The polynomial support, coefficients, scales, center and compatible source amplitude come from the samples. Branch and center selection finish before oracle scoring.", "",
        "| Quantity | Noiseless | Noisy | Hidden truth |", "| --- | ---: | ---: | ---: |"])
    for label, key, truth in (("p", "p", 1.7/.9), ("q", "q", 1/(1.7*.9)),
                               ("Field amplitude A", "amplitude", 1.7),
                               ("Transition width L", "length", .9),
                               ("Center x_c", "center", 0.),
                               ("Source amplitude", "source_amplitude", .8)):
        values = [f"{r['symbolic_recovery'][key]:.9g}" for r in reports.values()]
        lines.append(f"| {label} | {' | '.join(values)} | {truth:.9g} |")
    lines.extend(["", "| Symbolic prediction relative RMS error | Noiseless | Noisy |",
                  "| --- | ---: | ---: |"])
    for label, key in (("Field", "field_relative_rmse"), ("Source", "source_relative_rmse"),
                       ("Derivative", "derivative_relative_rmse"),
                       ("Extrapolated field (x = 3.05–4)", "extrapolated_field_relative_rmse"),
                       ("Extrapolated derivative (x = 3.05–4)", "extrapolated_derivative_relative_rmse")):
        values = [f"{100*r['symbolic_recovery'][key]:.4g}%" for r in reports.values()]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.extend(["", "Errors compare against the hidden analytic oracle on 901 unseen positions in [-3, 3]. Noise is Gaussian with standard deviations 0.001 × 1.7 for B and 0.001 × 0.8 for J; it is not relative to each local value.", "",
                  "The integrated prediction uses only the selected equation, the learned J function and one observed boundary value B(-3). Neither the true B curve nor its derivative is supplied to that solver.", "",
                  "Agreement is numerical on the tested interval, not exact equality of functions or evidence that physical laws were independently discovered. Extrapolation is reported separately. Only one seed and one generator family were tested.", "",
                  "[Explanation and formulas](../../README.md) · [Machine-readable report](results.json) · [Comparison plot](comparison.png)"])
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

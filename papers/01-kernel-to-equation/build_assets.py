"""Build the note's figure and table directly from the saved benchmark outputs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEMO = HERE.parents[1] / "experiments/kernel_benchmark"
report = json.loads((DEMO / "results/results.json").read_text(encoding="utf-8"))
assert report["implementation_sha256"] == hashlib.sha256((DEMO / "run_demo.py").read_bytes()).hexdigest(), "Rerun benchmark before building the note."
cases = report["cases"]
for case in cases.values():
    assert case["synthetic_data"] and case["equation_terms"] == ["J"]
    assert case["symbolic_recovery"]["closed_form_recovered"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "axes.labelsize": 9, "legend.fontsize": 7.3})
with np.load(DEMO / "results/noisy.npz") as data:
    x = data["test_x"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.30), constrained_layout=True)
    axes[0].plot(x, data["true_b"], color="#192b42", lw=2, label="Hidden analytic field")
    axes[0].scatter(data["train_x"], data["train_y"][:, 0], s=7, color="#9aabbc",
                    alpha=.7, label="Noisy training samples", zorder=1)
    axes[0].plot(x, data["kernel_b"], color="#16858c", ls="--", lw=1.5, label="Kernel fit")
    axes[0].plot(x, data["integrated_law_b"], color="#c76b2c", ls=":", lw=1.9,
                 label="Integrated recovered equation")
    axes[0].set(xlabel="Position x", ylabel="Field B", title="(a) Field reconstruction")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].plot(x, data["true_derivative"], color="#192b42", lw=2, label="Hidden analytic derivative")
    axes[1].plot(x, data["kernel_derivative"], color="#16858c", ls="--", lw=1.5,
                 label="Derivative of kernel fit")
    axes[1].plot(x, cases["noisy"]["equation_coefficients"][0]*data["kernel_j"],
                 color="#c76b2c", ls=":", lw=1.9, label="Recovered a times learned J")
    axes[1].set(xlabel="Position x", ylabel="Field derivative", title="(b) Differential relationship")
    axes[1].legend(loc="upper right", frameon=False)
    for axis in axes:
        axis.grid(alpha=.16)
        axis.set_xlim(-3.08, 3.08)
        axis.tick_params(labelsize=8)
    fig.savefig(HERE / "reconstruction.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(HERE / "reconstruction.png", dpi=200, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)

    extended_x = np.r_[x, data["extrapolation_x"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.10), constrained_layout=True)
    for axis, channel, title, ylabel in (
            (axes[0], "b", "(a) Recovered tanh field", "Field B"),
            (axes[1], "j", "(b) Recovered sech-squared source", "Source J")):
        axis.axvspan(3., 4., color="#e9edf3", label="Extrapolation", zorder=0)
        axis.plot(extended_x, np.r_[data["true_"+channel], data["extrapolation_true_"+channel]],
                  color="#192b42", lw=2, label="Hidden analytic truth")
        axis.plot(extended_x, np.r_[data["kernel_"+channel], data["extrapolation_kernel_"+channel]],
                  color="#16858c", ls="--", lw=1.5, label="Kernel fit")
        axis.plot(extended_x, np.r_[data["symbolic_"+channel], data["extrapolation_symbolic_"+channel]],
                  color="#c76b2c", ls=":", lw=1.9, label="Solved analytic form")
        axis.axvline(3., color="#758190", lw=.7)
        axis.set(xlabel="Position x", ylabel=ylabel, title=title, xlim=(-3.05, 4.05))
        axis.grid(alpha=.16)
        axis.tick_params(labelsize=8)
        axis.legend(loc="upper left" if channel == "b" else "upper right", frameon=False, fontsize=6.7)
    fig.savefig(HERE / "symbolic_recovery.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(HERE / "symbolic_recovery.png", dpi=200, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)

metrics = [(r"Field $\widehat B$", "field_relative_rmse"),
           (r"Derivative $\widehat B'$", "derivative_relative_rmse"),
           (r"Recovered coefficient $\widehat a$", "coefficient_relative_error"),
           (r"Integrated field $\widetilde B$", "integrated_law_field_relative_rmse")]
lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
         r"Relative error (\%) & Noiseless & Noisy \\", r"\midrule"]
for label, key in metrics:
    values = []
    for case in cases.values():
        number = 100*case[key]
        values.append(r"$%.2f\times10^{%d}$" % (number/10**np.floor(np.log10(number)), int(np.floor(np.log10(number))))
                      if number < .0001 else f"{number:.4g}")
    lines.append(label + " & " + " & ".join(values) + r" \\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(HERE / "results_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
(HERE / "values.tex").write_text(
    "\\newcommand{\\CleanCoefficient}{%.9f}\n\\newcommand{\\NoisyCoefficient}{%.9f}\n" %
    (cases["noiseless"]["equation_coefficients"][0], cases["noisy"]["equation_coefficients"][0]),
    encoding="utf-8")
symbolic_metrics = [(r"Field amplitude $A$", "amplitude", "1.7", False),
                    (r"Transition width $L$", "length", "0.9", False),
                    (r"Center $x_c$", "center", "0", False),
                    (r"Source amplitude $p/\widehat a$", "source_amplitude", "0.8", False),
                    (r"Field error (\%)", "field_relative_rmse", "--", True),
                    (r"Source error (\%)", "source_relative_rmse", "--", True),
                    (r"Derivative error (\%)", "derivative_relative_rmse", "--", True),
                    (r"Extrapolated field error (\%)", "extrapolated_field_relative_rmse", "--", True)]
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"Solved quantity & Noiseless & Noisy & Truth \\", r"\midrule"]
for label, key, truth, percent in symbolic_metrics:
    values = []
    for case in cases.values():
        number = case["symbolic_recovery"][key] * (100 if percent else 1)
        exponent = int(np.floor(np.log10(abs(number)))) if number else 0
        values.append(r"$%.2f\times10^{%d}$" % (number/10**exponent, exponent)
                      if number and abs(number) < .0001 else f"{number:.7g}" if not percent else f"{number:.4g}")
    lines.append(label + " & " + " & ".join(values) + " & " + truth + r" \\")
lines.extend([r"\bottomrule", r"\end{tabular}"])
(HERE/"symbolic_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
with (HERE/"values.tex").open("a", encoding="utf-8") as handle:
    for prefix, case in (("Clean", cases["noiseless"]), ("Noisy", cases["noisy"])):
        symbolic = case["symbolic_recovery"]
        for key, suffix in (("p", "P"), ("q", "Q")):
            handle.write("\\newcommand{\\%s%s}{%.9f}\n" % (prefix, suffix, symbolic[key]))
print("Built vector figure and LaTeX result assets from saved predictions.")

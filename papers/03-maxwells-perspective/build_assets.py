"""Build the expository note from checked, retained experimental outputs."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WIRE = REPO/"experiments/hidden_wire"
sys.path.insert(0, str(WIRE))
from learner import Expression


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report_path = WIRE/"results/results.json"
    arrays_path = WIRE/"results/predictions.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name, expected in report["implementation_sha256"].items():
        assert sha(WIRE/name) == expected, name
    assert sha(WIRE/"PROTOCOL.md") == report["protocol_sha256"]
    for name, expected in report["input_sha256"].items():
        assert sha(REPO/name) == expected, name
    runs = report["runs"]
    assert len(runs) == 15
    main_runs = [r for r in runs if r["case"] == "main"]
    first = next(r for r in main_runs if r["seed"] == 20261116)
    model = Expression.from_dict(first["distilled"]["selected"])
    exponent_to_row = {tuple(p): j for j, p in enumerate(model.exponents)}
    # Verify the displayed simplified expression has exactly the saved support.
    allowed_q = {(2, 0, 0), (0, 2, 0)}
    allowed_p = {((0, 1, 1), 0), ((1, 0, 1), 1)}
    assert {tuple(p) for p, c in zip(model.exponents, model.denominator) if c} == allowed_q
    assert {(tuple(model.exponents[i]), int(j)) for i, j in zip(*np.nonzero(model.numerators))} == allowed_p
    unit = model.denominator[exponent_to_row[(0, 2, 0)]]
    a = -model.numerators[exponent_to_row[(0, 1, 1)], 0]/unit
    b = model.numerators[exponent_to_row[(1, 0, 1)], 1]/unit
    d = model.denominator[exponent_to_row[(2, 0, 0)]]/unit

    def mean_error(case, domain, key):
        return 100*float(np.mean([r["evaluations"][domain][key]["oracle_relative_rmse"]
                                 for r in runs if r["case"] == case]))

    table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Unseen conditions & Kernel & Kernel $\to$ equation & Direct equation \\", r"\midrule"]
    domains = (("in_domain", "Same domain"), ("inward", "Smaller radii"),
               ("outward", "Larger radii"), ("new_current", "Larger currents"),
               ("moved_wire", "Moved wire; no update"))
    for domain, label in domains:
        table.append(label+" & "+" & ".join(f"{mean_error('main', domain, key):.3f}"
                     for key in ("kernel", "distilled", "direct"))+r" \\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (HERE/"wire_table.tex").write_text("\n".join(table)+"\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "axes.labelsize": 8,
        "legend.fontsize": 6.7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    with np.load(arrays_path) as data:
        # Verify all selected expression predictions and all saved test scores.
        for row in runs:
            prefix = f"{row['case']}_{row['seed']}_"
            for key in ("distilled", "direct"):
                expression = Expression.from_dict(row[key]["selected"])
                for domain, evaluation in row["evaluations"].items():
                    prediction = data[prefix+domain+"_"+key]
                    np.testing.assert_allclose(expression.predict(data[prefix+domain+"_x"]),
                                               prediction, atol=1e-12)
                    actual = data[prefix+domain+"_oracle"]
                    relative = np.linalg.norm(prediction-actual)/np.linalg.norm(actual)
                    np.testing.assert_allclose(relative, evaluation[key]["oracle_relative_rmse"], rtol=1e-12)
        prefix = "main_20261116_"
        cut = data[prefix+"cut_x"][:, 0]
        np.testing.assert_allclose(model.predict(data[prefix+"cut_x"]), data[prefix+"cut_distilled"], atol=1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), constrained_layout=True)
        for key, label, color, style in (("oracle", "Hidden field", "#263849", "-"),
                                       ("kernel", "Gaussian kernel", "#16858c", "-"),
                                       ("distilled", "Kernel to equation", "#c76b2c", "-"),
                                       ("direct", "Direct equation", "#737373", "--")):
            axes[0].plot(cut, data[prefix+"cut_"+key][:, 1], label=label, color=color,
                         ls=style, lw=1.2)
        axes[0].axvspan(.25, 2., color="#263849", alpha=.06)
        axes[0].set(xlabel="Distance on +x axis (m), I = 2 A", ylabel=r"$B_y$ ($\mu$T)",
                    title="(a) First fixed main seed")
        axes[0].legend(frameon=False)
        cases = (("noiseless", "No noise"), ("main", "1%"), ("noisy", "5%"),
                 ("sparse", "80 points"), ("limited_angle", "Wedge"))
        for key, label, color in (("kernel", "Gaussian kernel", "#16858c"),
                                  ("distilled", "Kernel to equation", "#c76b2c"),
                                  ("direct", "Direct equation", "#737373")):
            means, deviations = [], []
            for case, _ in cases:
                errors = [100*r["evaluations"]["full_angles"][key]["oracle_relative_rmse"]
                          for r in runs if r["case"] == case]
                means.append(max(float(np.mean(errors)), 1e-3))
                deviations.append(float(np.std(errors)))
            axes[1].errorbar(range(len(cases)), means, yerr=deviations, marker="o", ms=3,
                             lw=1, capsize=2, color=color, label=label)
        axes[1].set_xticks(range(len(cases)), [label for _, label in cases], rotation=18)
        axes[1].set(yscale="log", ylim=(5e-4, 2e4), ylabel="Full-angle field error (%)",
                    title="(b) Observation process changes")
        axes[1].text(.025, .04, "Exact direct fit shown at 0.001% floor", fontsize=6.2,
                     transform=axes[1].transAxes)
        for ax in axes:
            ax.grid(alpha=.16)
            ax.margins(x=.025)
        fig.savefig(HERE/"perspective.pdf", bbox_inches="tight", pad_inches=.025)
        fig.savefig(HERE/"perspective.png", dpi=220, bbox_inches="tight", pad_inches=.025)
        plt.close(fig)

    benchmark_path = REPO/"experiments/kernel_benchmark/results/results.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert sha(REPO/"experiments/kernel_benchmark/run_demo.py") == benchmark["implementation_sha256"]
    symbolic = benchmark["cases"]["noisy"]["symbolic_recovery"]
    sensor = report["sensor"]
    values = dict(WireA=f"{a:.6f}", WireB=f"{b:.6f}", WireD=f"{d:.6f}",
        WireKernel=f"{mean_error('main', 'in_domain', 'kernel'):.3f}",
        WireEquation=f"{mean_error('main', 'in_domain', 'distilled'):.3f}",
        WireDirect=f"{mean_error('main', 'in_domain', 'direct'):.3f}",
        NoiselessEquation=f"{mean_error('noiseless', 'in_domain', 'distilled'):.3f}",
        SparseEquation=f"{mean_error('sparse', 'in_domain', 'distilled'):.2f}",
        WedgeInside=f"{mean_error('limited_angle', 'in_domain', 'distilled'):.3f}",
        WedgeOutside=f"{mean_error('limited_angle', 'full_angles', 'distilled'):.3f}",
        TanhAmplitude=f"{symbolic['amplitude']:.6f}", TanhLength=f"{symbolic['length']:.6f}",
        TanhCenter=f"{symbolic['center']:.6f}",
        TanhError=f"{100*symbolic['field_relative_rmse']:.3f}",
        SensorKernel=f"{sensor['stable']['kernel']['vector_rmse']:.3f}",
        SensorEquation=f"{sensor['stable']['distilled']['vector_rmse']:.3f}",
        SensorDirect=f"{sensor['stable']['direct']['vector_rmse']:.3f}")
    (HERE/"values.tex").write_text("".join(f"\\newcommand{{\\{key}}}{{{value}}}\n"
                                         for key, value in values.items()), encoding="utf-8")
    paths = [report_path, arrays_path, WIRE/"PROTOCOL.md", benchmark_path, Path(__file__)]
    paths += [REPO/name for name in report["input_sha256"]]
    paths += [WIRE/name for name in report["implementation_sha256"]]
    (HERE/"asset_manifest.json").write_text(json.dumps({p.relative_to(REPO).as_posix(): sha(p)
        for p in paths}, indent=2)+"\n", encoding="utf-8")
    print("Built checked equations, results table and figure for Maxwell's Perspective.")


if __name__ == "__main__":
    main()

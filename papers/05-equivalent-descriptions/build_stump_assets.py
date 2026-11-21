"""Build the stump paper's tables and vector plots from retained experiments.

No fitting, downloading, or test-set selection takes place in this builder.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXPERIMENT = REPO / "experiments/magnetic_mapping"
DATA = EXPERIMENT / "tree_comparison"
NAMES = ("constant", "boosted_stumps", "random_forest", "kernel", "direct_dipoles", "dipoles")
LABELS = ("Constant", "Boosted stumps", "Random forest", "Gaussian kernel", "Direct dipoles", "Kernel to dipoles")
WAVE_NAMES = ("random_forest", "boosted_stumps", "maxwell_stumps", "phase_random_forest")
WAVE_LABELS = ("Forest on (z,t)", "Stumps on (z,t)", "Stumps on phase", "Forest on phase")
COLORS = ("#b66a2a", "#9970ab", "#16817a", "#2357a2")


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected in manifest["inputs_sha256"].items():
        assert hashlib.sha256((EXPERIMENT / name.replace("\\", "/")).read_bytes()).hexdigest() == expected, name
    sensor = json.loads((DATA / "sensor_results.json").read_text())
    wave = json.loads((DATA / "wave_results.json").read_text())
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                        "axes.spines.right": False, "pdf.fonttype": 42})

    # Every instrument and environment is included; tables are not transcribed.
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Environment & Instrument & Stumps & Forest & Gaussian & Direct & Two-stage\\",
             r"\midrule"]
    for env in ("stable", "changed"):
        for sensor_name in ("invensense", "nexus", "trivisio"):
            scores = sensor["pooled"][env][sensor_name]["all"]["models"]
            values = " & ".join(f"{scores[k]['mean']:.2f}" for k in NAMES[1:])
            lines.append(f"{env.capitalize()} & {sensor_name.capitalize()} & {values}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "stump_sensor_table.tex").write_text("\n".join(lines)+"\n")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model & $t=0.5$ & $t=2$ & $t=4$\\", r"\midrule"]
    for name, label in zip(WAVE_NAMES, WAVE_LABELS):
        cells = []
        for t in (.5, 2., 4.):
            row = next(r for r in wave["summary"] if r["model"] == name and r["time"] == t)
            stat = row["relative_l2"]
            cells.append(f"${100*stat['mean']:.2f}\\pm{100*stat['sd']:.2f}$")
        lines.append(label + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "stump_wave_table.tex").write_text("\n".join(lines)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), sharey=True, constrained_layout=True)
    for ax, region, title in zip(axes, ("all", "hole"), ("(a) Full test recordings", "(b) Withheld spatial region")):
        for env, shift, color in (("stable", -.18, "#2357a2"), ("changed", .18, "#b66a2a")):
            scores = sensor["pooled"][env]["invensense"][region]["models"]
            ax.barh(np.arange(len(NAMES))+shift, [scores[n]["mean"] for n in NAMES], height=.34,
                    xerr=[scores[n]["sd"] for n in NAMES], color=color, label=env.capitalize(), capsize=2)
        ax.set(yticks=np.arange(len(NAMES)), yticklabels=LABELS,
               xlabel="Vector RMSE (reference units)", title=title, xlim=(0, 19))
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=8)
    fig.savefig(HERE / "stump_sensor_comparison.pdf")
    fig.savefig(HERE / "preview-stump-sensors.png", dpi=160)
    plt.close(fig)

    # Fixed first seed, with all models overlaid; other seeds remain in the score table.
    saved = np.load(DATA / "wave_predictions.npz")
    phase = saved["phase_grid"]
    exact = np.exp(-(phase/.7)**2)*np.cos(4*phase)
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.3), constrained_layout=True)
    for ax, t, title in ((axes[0, 0], .5, "(a) Inside training time: t=0.5"),
                         (axes[0, 1], 2., "(b) Future prediction: t=2")):
        ax.plot(phase, exact, color="black", linewidth=1.5, label="Target", zorder=5)
        for name, label, color in zip(WAVE_NAMES, WAVE_LABELS, COLORS):
            ax.plot(phase, saved[f"seed11_{name}_t{t}"][:, 0], color=color, linewidth=1,
                    alpha=.85, label=label)
        ax.set(xlabel="Co-moving position z-t", ylabel="Electric field", title=title)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    for ax, names, title in ((axes[1, 0], WAVE_NAMES, "(c) Error across all models"),
                             (axes[1, 1], WAVE_NAMES[2:], "(d) Models given the traveling phase")):
        for name, label, color in zip(WAVE_NAMES, WAVE_LABELS, COLORS):
            if name not in names:
                continue
            rows = [r for r in wave["summary"] if r["model"] == name]
            t = np.array([r["time"] for r in rows])
            mean = np.array([100*r["relative_l2"]["mean"] for r in rows])
            sd = np.array([100*r["relative_l2"]["sd"] for r in rows])
            ax.errorbar(t, mean, yerr=sd, color=color, marker="o", markersize=3,
                        capsize=2, linewidth=1.2, label=label)
        ax.axvspan(0, 1, color="gray", alpha=.1)
        ax.set(xlabel="Time (c=1)", ylabel="Relative field error (%)", title=title)
        ax.grid(alpha=.15)
    axes[1, 1].set_ylim(0, 10)
    axes[1, 1].legend(fontsize=7)
    fig.savefig(HERE / "stump_wave_comparison.pdf")
    fig.savefig(HERE / "preview-stump-waves.png", dpi=160)
    plt.close(fig)

    inputs = [Path(__file__), DATA/"sensor_results.json", DATA/"wave_results.json",
              DATA/"wave_predictions.npz", DATA/"manifest.json"]
    (HERE/"asset_manifest.json").write_text(json.dumps({p.relative_to(REPO).as_posix():
        hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)+"\n")
    print("Built two vector figures and two tables from verified retained results.")


if __name__ == "__main__":
    main()

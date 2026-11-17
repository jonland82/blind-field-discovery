"""Build the follow-up note directly from the saved real-data experiment."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
RESULTS = REPOSITORY / "experiments/magnetic_mapping/dipole_field"
EXPERIMENT = RESULTS.parent
sys.path.insert(0, str(EXPERIMENT))
from dipole_models import DipoleField
from models import FieldMap
from run_experiment import verify_archive

report = json.loads((RESULTS/"results.json").read_text(encoding="utf-8"))
for name, expected in report["code_sha256"].items():
    assert hashlib.sha256((EXPERIMENT/name).read_bytes()).hexdigest() == expected, name
data_archive = REPOSITORY / "data/aalto_magnetic"
if data_archive.exists():
    verify_archive(data_archive)
else:
    print("Raw data are not cached; building from the retained measured-data results.")
spec = report["field"]
dipoles = DipoleField(spec["sources_metres"], spec["background"], spec["effective_moments"])
old = json.loads((EXPERIMENT/"results/results.json").read_text(encoding="utf-8"))
config = old["model_selection"]["unrestricted"]["selected"]

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "axes.labelsize": 8,
    "legend.fontsize": 6.7, "xtick.labelsize": 7, "ytick.labelsize": 7})
with np.load(RESULTS/"predictions.npz") as data:
    x, y = data["training_xy"], data["training_actual"]
    kernel = FieldMap("unrestricted", config["length_metres"], config["ridge"]).fit(x, y)
    # Check plotted predictions and all reported scores against the saved observations.
    for row in report["evaluations"]:
        prefix = f"{row['sensor']}_{row['trial']}"
        actual, predicted = data[prefix+"_actual"], data[prefix+"_"+row["model"]]
        np.testing.assert_allclose(np.sqrt(np.mean(np.sum((predicted-actual)**2, axis=1))), row["vector_rmse"])
    prefix = "invensense_4"
    xy, actual = data[prefix+"_xy"], data[prefix+"_actual"]
    kt, ds = kernel.predict(xy), dipoles.predict(xy)
    np.testing.assert_allclose(kt, data[prefix+"_kernel"], atol=1e-10)
    np.testing.assert_allclose(ds, data[prefix+"_dipoles"], atol=1e-10)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    cut_y = float(np.median(x[:, 1]))
    cut = np.column_stack((np.linspace(x[:, 0].min(), x[:, 0].max(), 401), np.full(401, cut_y)))
    cut_kernel, cut_dipoles = kernel.predict(cut), dipoles.predict(cut)
    fig, axes = plt.subplots(3, 2, figsize=(7.05, 3.70), constrained_layout=True)
    for j in range(3):
        ax, cut_ax = axes[j]
        ax.scatter(distance, actual[:, j], s=3, c="#9aabbc", alpha=.55, rasterized=True, label="Measured")
        ax.plot(distance, kt[:, j], c="#16858c", lw=1, label="Kernel")
        ax.plot(distance, ds[:, j], c="#c76b2c", lw=1, label="Dipole reconstruction")
        cut_ax.plot(cut[:, 0], cut_kernel[:, j], c="#16858c", lw=1.2)
        cut_ax.plot(cut[:, 0], cut_dipoles[:, j], c="#c76b2c", lw=1.2)
        ax.set_ylabel(f"$B_{j+1}$")
        cut_ax.set_ylabel(f"$B_{j+1}$")
        for axis in (ax, cut_ax):
            axis.grid(alpha=.16)
            axis.margins(x=.015)
    axes[0, 0].set_title("(a) Full held-out trial 4", fontsize=9)
    axes[0, 1].set_title(f"(b) Model cut at y = {cut_y:.2f} m", fontsize=9)
    axes[2, 0].set_xlabel("Distance along sampled path (m)")
    axes[2, 1].set_xlabel("Position x (m)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(HERE/"field_reconstruction.pdf", bbox_inches="tight", pad_inches=.025)
    fig.savefig(HERE/"field_reconstruction.png", dpi=220, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)

labels = dict(invensense="Invensense", nexus="Nexus phone", trivisio="Trivisio")
for environment, filename in (("stable", "stable_table.tex"), ("changed", "stress_table.tex")):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Instrument & Constant & Kernel & Kernel $\to$ dipoles & Direct dipoles \\", r"\midrule"]
    for sensor, label in labels.items():
        scores = report["pooled"][environment][sensor]
        lines.append(label+" & "+" & ".join(f"{scores[name]['vector_rmse']:.2f}" for name in
            ("constant", "kernel", "dipoles", "direct_dipoles"))+r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE/filename).write_text("\n".join(lines)+"\n", encoding="utf-8")

chosen = report["selected"]
source_scores = report["pooled"]["stable"]["invensense"]
improvement = 100*(1-source_scores["dipoles"]["vector_rmse"]/source_scores["kernel"]["vector_rmse"])
values = dict(SourceCount=str(len(spec["sources_metres"])), Depth=f"{chosen['depth_metres']:g}",
    Ridge=f"{chosen['ridge']:g}", TeacherValidation=f"{report['teacher_validation_rmse']:.2f}",
    DipoleValidation=f"{chosen['validation_vector_rmse']:.2f}", Improvement=f"{improvement:.1f}",
    Correlations=",\\ ".join(f"{c:.3f}" for c in source_scores["dipoles"]["component_correlations"]),
    CutY=f"{cut_y:.2f}", KernelLength=f"{config['length_metres']:g}", KernelRidge=f"{config['ridge']:g}",
    TrainCount=str(len(x)), CandidateCount=str(len(report["candidates"])))
(HERE/"values.tex").write_text("".join(f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in values.items()), encoding="utf-8")
inputs = [RESULTS/"results.json", RESULTS/"predictions.npz", EXPERIMENT/"results/results.json", Path(__file__)]
(HERE/"asset_manifest.json").write_text(json.dumps({p.relative_to(REPOSITORY).as_posix():
    hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2)+"\n", encoding="utf-8")
print("Built checked figure, tables and values for the real-data follow-up.")

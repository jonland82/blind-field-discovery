"""Compute exact per-reading weights for one spatially held-out prediction."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENT = ROOT / "experiments/magnetic_mapping"
sys.path.insert(0, str(EXPERIMENT))
from dipole_models import design, fit_path
from models import FieldMap
from run_experiment import BUFFER, HOLE, in_hole


def dipole_weights(x, query, sources, ridge, smoother):
    blocks = design(x, sources).reshape(len(x), 3, -1)
    mean = blocks.mean(axis=0)
    centered = (blocks - mean).reshape(3 * len(x), -1)
    scales = np.maximum(np.linalg.norm(centered, axis=0), 1e-12)
    u, singular, vt = np.linalg.svd(centered / scales, full_matrices=False)
    query_design = (design(query, sources) - mean) / scales
    response = ((query_design @ vt.T) * (singular / (singular**2 + ridge))) @ u.T
    response = response.reshape(3, len(x), 3)
    # Center the targets and restore their unpenalized mean at the query.
    response += np.eye(3)[:, None, :] / len(x) - response.mean(axis=1, keepdims=True)
    return np.einsum("oac,ai->ioc", response, smoother)


def draw_map(x, query, kernel, dipoles):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.35), sharex=True, sharey=True,
                             layout="constrained")
    bound = max(np.abs(kernel).max(), np.abs(dipoles).max())
    norm = SymLogNorm(linthresh=.01, vmin=-bound, vmax=bound)
    selected = x[np.argmax(np.abs(kernel - dipoles))]
    left, right, bottom, top = HOLE
    for ax, weights, title in zip(axes, (kernel, dipoles), ("Gaussian field", "Dipole reconstruction")):
        order = np.argsort(np.abs(weights))
        ax.add_patch(Rectangle((left, bottom), right-left, top-bottom, facecolor="#eeeeea",
                               edgecolor="0.4", lw=0.8, ls="--"))
        dots = ax.scatter(*x[order].T, c=weights[order], cmap="RdBu_r", s=17, lw=.25,
                          edgecolors="0.52", norm=norm)
        ax.plot(*query, marker="x", color="black", ms=6, mew=1.4)
        ax.plot(*selected, marker="o", mfc="none", mec="black", ms=8, mew=.8)
        ax.set(aspect="equal", title=title, xlabel="Room x (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Room y (m)")
    ticks = [-.5, -.1, -.01, 0, .01, .1, .5]
    fig.colorbar(dots, ax=axes, location="bottom", shrink=.72, aspect=40, pad=.03,
                 ticks=ticks, format="%g",
                 label=r"Change in predicted $B_z$ per unit increase in one measured $B_z$")
    fig.savefig(HERE / "influence_map.pdf", bbox_inches="tight")
    plt.close(fig)


def build():
    report_path = EXPERIMENT / "dipole_field/results.json"
    kernel_path = EXPERIMENT / "results/results.json"
    archive_path = EXPERIMENT / "dipole_field/predictions.npz"
    report = json.loads(report_path.read_text())
    config = json.loads(kernel_path.read_text())["model_selection"]["unrestricted"]["selected"]
    sources = np.array(report["field"]["sources_metres"])
    ridge = report["selected"]["ridge"]
    with np.load(archive_path) as data:
        x, y = data["training_xy"], data["training_actual"]
        xy = data["invensense_4_xy"]
        center = np.array(HOLE).reshape(2, 2).mean(axis=1)
        candidates = np.flatnonzero(in_hole(xy))
        idx = candidates[np.argmin(np.linalg.norm(xy[candidates] - center, axis=1))]
        query = xy[[idx]]
        saved_k = data["invensense_4_kernel"][idx]
        saved_d = data["invensense_4_dipoles"][idx]
    assert not in_hole(x, BUFFER).any()
    def gaussian(targets):
        return FieldMap("unrestricted", config["length_metres"], config["ridge"]).fit(x, targets)

    basis = gaussian(np.eye(len(x)))
    k = basis.predict(query).ravel()
    weights = dipole_weights(x, query, sources, ridge, basis.predict(x))
    np.testing.assert_allclose(k @ y, saved_k, atol=1e-9, rtol=0)
    np.testing.assert_allclose(np.einsum("ioc,ic->o", weights, y), saved_d, atol=1e-9, rtol=0)
    np.testing.assert_allclose(weights.sum(axis=0), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(k.sum(), 1., atol=1e-10)
    w = weights[:, 2, 2]
    chosen = np.argmax(np.abs(w - k))
    changed = y.copy()
    changed[chosen, 2] += 1
    teacher = gaussian(changed)
    _, field = next(fit_path(x, teacher.predict(x), sources, [ridge]))
    np.testing.assert_allclose(teacher.predict(query)[0] - saved_k,
                               k[chosen] * np.eye(3)[:, 2], rtol=0, atol=1e-9)
    np.testing.assert_allclose(field.predict(query)[0] - saved_d,
                               weights[chosen, :, 2], rtol=0, atol=1e-9)
    distance = np.linalg.norm(x - query, axis=1)
    draw_map(x, query[0], k, w)
    values = dict(QueryX=query[0, 0], QueryY=query[0, 1], SampleX=x[chosen, 0],
                  SampleY=x[chosen, 1], KernelChange=k[chosen], DipoleChange=w[chosen],
                  Nearest=distance.min(), KernelFar=np.abs(k[distance > 1]).sum()/abs(k).sum(),
                  DipoleFar=np.abs(w[distance > 1]).sum()/abs(w).sum(),
                  KernelMass=abs(k).sum(), DipoleMass=abs(w).sum(),
                  CrossMax=np.abs(weights[:, 2, :2]).max())
    lines = []
    for name, value in values.items():
        number = f"{100*value:.1f}" if name.endswith("Far") else f"{value:.3f}"
        lines.append(r"\newcommand{\%s}{%s}" % (name, number))
    (HERE / "values.tex").write_text("\n".join(lines) + "\n")
    inputs = [archive_path, report_path, kernel_path, Path(__file__), EXPERIMENT/"models.py",
              EXPERIMENT/"dipole_models.py", EXPERIMENT/"run_experiment.py"]
    (HERE / "asset_manifest.json").write_text(json.dumps({
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs}, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        build()

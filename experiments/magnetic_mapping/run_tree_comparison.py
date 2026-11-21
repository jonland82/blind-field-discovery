"""Exploratory tree baselines and a separate controlled Maxwell-front test.

Run from the repository root with Python. Requires scikit-learn in addition to
the magnetic mapping requirements. Existing experiment artifacts are read only.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import hashlib
import argparse
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from run_experiment import (ROOT, SOURCE, TRAIN_TRIALS, VALIDATION_TRIALS,
    load_recording, spatial_split, verify_archive, in_hole, vector_rmse)

OUTPUT = ROOT / "tree_comparison"
SEEDS = (11, 23, 37)
RF_GRID = [dict(n_estimators=150, max_depth=d, min_samples_leaf=l, max_features=f)
           for d, l, f in itertools.product((6, 12, None), (1, 4, 12), (1.0, .7))]
STUMP_GRID = [dict(n_estimators=n, learning_rate=r, max_depth=1)
              for n, r in itertools.product((50, 200, 600), (.03, .1, .3))]


def estimator(kind, config, seed):
    if kind == "random_forest":
        return RandomForestRegressor(**config, random_state=seed, n_jobs=1)
    return MultiOutputRegressor(GradientBoostingRegressor(**config, random_state=seed))


def select(kind, x, y, xv, yv):
    """Select by mean validation error over fixed seeds, never by test labels."""
    grid = RF_GRID if kind == "random_forest" else STUMP_GRID
    candidates, best = [], None
    for config in grid:
        models = [estimator(kind, config, seed).fit(x, y) for seed in SEEDS]
        scores = [vector_rmse(yv, m.predict(xv)) for m in models]
        row = dict(config=config, validation_rmse_by_seed=scores,
                   validation_rmse_mean=float(np.mean(scores)))
        candidates.append(row)
        if best is None or row["validation_rmse_mean"] < best[0]["validation_rmse_mean"]:
            best = row, models
    print(kind, "selected", best[0], flush=True)
    return best[1], dict(selected=best[0], candidates=candidates)


def stats(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)),
                by_seed=[float(x) for x in values])


def sensor_experiment():
    provenance = verify_archive(ROOT.parents[1] / "data/aalto_magnetic")
    old = json.loads((ROOT / "results/results.json").read_text())
    records = {(SOURCE, t): load_recording(ROOT.parents[1] / "data/aalto_magnetic", SOURCE, t)
               for t in TRAIN_TRIALS + VALIDATION_TRIALS}
    x, y, train_audit = spatial_split(records, TRAIN_TRIALS, old["protocol"]["train_per_trial_cap"])
    xv, yv, val_audit = spatial_split(records, VALIDATION_TRIALS, 600)
    # Match the exact arrays underlying all saved physical comparator predictions.
    with np.load(ROOT / "dipole_field/predictions.npz") as a:
        for key, actual in (("training_xy", x), ("training_actual", y),
                            ("validation_xy", xv), ("validation_actual", yv)):
            np.testing.assert_array_equal(actual, a[key])
    models, selection = {}, {}
    for kind in ("random_forest", "boosted_stumps"):
        models[kind], selection[kind] = select(kind, x, y, xv, yv)
    print("Sensor settings frozen; opening test predictions.", flush=True)
    a = np.load(ROOT / "dipole_field/predictions.npz")
    baselines = ("constant", "kernel", "direct_dipoles", "dipoles")
    rows, pools, saved = [], {}, {}
    for environment, trials in (("stable", (4, 5)), ("changed", (6, 7, 8, 9))):
        pools[environment] = {}
        for sensor in ("invensense", "nexus", "trivisio"):
            actuals, coords = [], []
            predictions = {name: [] for name in (*baselines, *models)}
            for trial in trials:
                prefix = f"{sensor}_{trial}"
                query, actual = a[prefix + "_xy"], a[prefix + "_actual"]
                actuals.append(actual)
                coords.append(query)
                values = {k: a[prefix + "_" + k][None, ...] for k in baselines}
                values.update({k: np.stack([m.predict(query) for m in ms]) for k, ms in models.items()})
                saved[prefix + "_actual"], saved[prefix + "_xy"] = actual, query
                for name, pred in values.items():
                    predictions[name].append(pred)
                    saved[prefix + "_" + name] = pred
                    hole = in_hole(query)
                    rows.append(dict(environment=environment, sensor=sensor, trial=trial,
                        model=name, n=len(query), rmse_by_seed=[vector_rmse(actual, p) for p in pred],
                        hole_n=int(hole.sum()), hole_rmse_by_seed=
                        [vector_rmse(actual[hole], p[hole]) for p in pred] if hole.any() else None))
            actual, query = np.concatenate(actuals), np.concatenate(coords)
            masks = dict(all=np.ones(len(query), dtype=bool), hole=in_hole(query), outside_hole=~in_hole(query))
            pools[environment][sensor] = {}
            for region, mask in masks.items():
                pools[environment][sensor][region] = dict(n=int(mask.sum()), models={})
                for name, parts in predictions.items():
                    pred = np.concatenate(parts, axis=1)
                    scores = [vector_rmse(actual[mask], p[mask]) for p in pred] if mask.any() else []
                    pools[environment][sensor][region]["models"][name] = (
                        stats(scores) if len(scores) > 1 else dict(mean=scores[0], sd=0.0) if scores else None)
    # Verify every reused pooled baseline agrees with the earlier published run.
    previous = json.loads((ROOT / "dipole_field/results.json").read_text())
    for env in pools:
        for sensor in pools[env]:
            for name in baselines:
                np.testing.assert_allclose(pools[env][sensor]["all"]["models"][name]["mean"],
                    previous["pooled"][env][sensor][name]["vector_rmse"], rtol=1e-12)
    np.savez_compressed(OUTPUT / "sensor_predictions.npz", **saved)
    result = dict(protocol=dict(train=[1, 2], validation=[3], stable=[4, 5], changed=[6, 7, 8, 9],
        n_train=len(x), n_validation=len(xv), seeds=SEEDS,
        selection="Mean source trial-3 vector RMSE across seeds; refit retains training trials only.",
        calibration="Unchanged trial-1-only alignment from original experiment.",
        limitations="Exploratory reuse of previously examined tests. Seed SD is algorithm variation, not a confidence interval.",
        input="Planar position only; three aligned magnetic targets; no propagation constraints.",
        training_rows=train_audit, validation_rows=val_audit), selection=selection,
        pooled=pools, per_trial=rows, data_commit=provenance["commit"])
    return result


def profile(x):
    return np.exp(-(x / .7)**2) * np.cos(4*x)


def wave_data(rng, n):
    # Sample (z,t) uniformly in a rectangle, retaining only the common phase domain.
    parts = []
    while sum(map(len, parts)) < n:
        q = np.column_stack((rng.uniform(-3, 4, n), rng.uniform(0, 1, n)))
        xi = q[:, 0] - q[:, 1]
        parts.append(q[(xi >= -3) & (xi <= 3)])
    return np.concatenate(parts)[:n]


def front_coefficients(model):
    """Exact additive stump expansion of a fitted depth-one boosted model."""
    base = model.estimators_[0]
    baseline = float(base.init_.constant_.ravel()[0])
    thresholds, jumps = [], []
    for tree in base.estimators_.ravel():
        tr = tree.tree_
        if tr.node_count == 1:
            baseline += base.learning_rate * float(tr.value[0].ravel()[0])
        else:
            assert tr.node_count == 3
            left, right = float(tr.value[1].ravel()[0]), float(tr.value[2].ravel()[0])
            baseline += base.learning_rate * left
            thresholds.append(float(tr.threshold[0]))
            jumps.append(base.learning_rate * (right-left))
    return baseline, np.array(thresholds), np.array(jumps)


def wave_experiment():
    # Dimensionless c=1. Outputs are E and C=cB, with independent channel noise.
    output, predictions = [], {}
    xi = np.linspace(-3, 3, 2401)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x, xv = wave_data(rng, 600), wave_data(rng, 300)
        clean = np.repeat(profile(x[:, 0]-x[:, 1])[:, None], 2, axis=1)
        cleanv = np.repeat(profile(xv[:, 0]-xv[:, 1])[:, None], 2, axis=1)
        sigma = .02 * np.sqrt(np.mean(clean**2))
        y = clean + rng.normal(0, sigma, clean.shape)
        yv = cleanv + rng.normal(0, sigma, cleanv.shape)
        candidates, chosen = {}, {}
        for name in ("random_forest", "boosted_stumps", "maxwell_stumps", "phase_random_forest"):
            grid = RF_GRID if name in ("random_forest", "phase_random_forest") else STUMP_GRID
            best = None
            for cfg in grid:
                if name in ("maxwell_stumps", "phase_random_forest"):
                    tx, vx = (x[:, 0]-x[:, 1])[:, None], (xv[:, 0]-xv[:, 1])[:, None]
                    ty = y.mean(axis=1, keepdims=True)
                else:
                    tx, vx, ty = x, xv, y
                kind = "random_forest" if name == "phase_random_forest" else "boosted_stumps" if name == "maxwell_stumps" else name
                m = estimator(kind, cfg, seed).fit(tx, ty.ravel() if name == "phase_random_forest" else ty)
                p = m.predict(vx)
                if p.ndim == 1:
                    p = p[:, None]
                if name in ("maxwell_stumps", "phase_random_forest"):
                    p = np.repeat(p, 2, axis=1)
                score = vector_rmse(yv, p)
                candidates.setdefault(name, []).append(dict(config=cfg, validation_rmse=score))
                if best is None or score < best[0]:
                    best = score, m, cfg
            chosen[name] = best
        print("Wave seed", seed, "settings frozen:", {k: v[2] for k, v in chosen.items()}, flush=True)
        tests = []
        for t in (0., .5, 1., 2., 4.):
            query = np.column_stack((xi+t, np.full(len(xi), t)))
            truth = np.repeat(profile(xi)[:, None], 2, axis=1)
            for name, (_, model, _) in chosen.items():
                if name in ("maxwell_stumps", "phase_random_forest"):
                    pred = np.repeat(model.predict(xi[:, None]).reshape(-1, 1), 2, axis=1)
                else:
                    pred = model.predict(query)
                tests.append(dict(time=t, model=name,
                    relative_l2=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),
                    electric_sup_error=float(np.max(np.abs(pred[:, 0]-truth[:, 0]))),
                    polarization_mismatch_rms=float(np.sqrt(np.mean((pred[:, 0]-pred[:, 1])**2)))))
                predictions[f"seed{seed}_{name}_t{t}"] = pred
        _, m, _ = chosen["maxwell_stumps"]
        baseline, thresholds, jumps = front_coefficients(m)
        # sklearn uses > rather than >= at a split; point values do not affect the weak solution.
        reconstructed = baseline + (xi[:, None] > thresholds) @ jumps
        np.testing.assert_allclose(reconstructed, m.predict(xi[:, None]).ravel(), atol=1e-12)
        # For U=(E,cB), c=1: residual delta coefficients are (-s*jE+jC, -s*jC+jE).
        residual = np.column_stack((-jumps+jumps, -jumps+jumps))
        wrong_speed_residual = np.column_stack((-.8*jumps+jumps, -.8*jumps+jumps))
        assert np.max(np.abs(residual)) == 0
        assert np.max(np.abs(wrong_speed_residual)) > 0
        output.append(dict(seed=seed, noise_sigma=float(sigma), candidates=candidates,
            selected={k: dict(config=v[2], validation_rmse=v[0]) for k, v in chosen.items()}, tests=tests,
            front_check=dict(n_stumps=len(jumps), max_abs_delta_coefficient=float(np.max(np.abs(residual))),
                wrong_speed_08_max_abs_delta_coefficient=float(np.max(np.abs(wrong_speed_residual)))),
            fitted_profile=dict(baseline=baseline, thresholds=thresholds.tolist(), jumps=jumps.tolist())))
        predictions[f"seed{seed}_train_zt"], predictions[f"seed{seed}_train_noisy_E_C"] = x, y
    predictions["phase_grid"] = xi
    np.savez_compressed(OUTPUT / "wave_predictions.npz", **predictions)
    summary = []
    for t, name in itertools.product((0., .5, 1., 2., 4.), ("random_forest", "boosted_stumps", "maxwell_stumps", "phase_random_forest")):
        entries = [next(r for r in run["tests"] if r["time"] == t and r["model"] == name) for run in output]
        summary.append(dict(time=t, model=name, **{key: stats([e[key] for e in entries]) for key in
            ("relative_l2", "electric_sup_error", "polarization_mismatch_rms")}))
    return dict(protocol=dict(c=1, target="exp(-(xi/.7)^2)*cos(4*xi), xi=z-t; E=cB=f(xi)",
        seeds=SEEDS, n_train=600, n_validation=300, training_time=[0, 1],
        train_phase=[-3, 3], noise="Independent Gaussian in E and cB; sigma=2% clean pooled component RMS.",
        selection="Per-seed noisy validation RMSE within training time interval; no future labels.",
        test="Noiseless dense phase grid [-3,3], translated to each test time; all target phases observed in training domain.",
        interpretation="Known direction and speed supplied to Maxwell stumps and phase random forest; both also enforce E=cB and average the two noisy targets. Controlled inductive-bias comparison, not discovery. Phase forest added as an attribution control after examining initial wave results.",
        equation_check="Exact distributional delta coefficients at every fitted constrained front; no finite-difference residual proxy."),
        runs=output, summary=summary)


def plot(sensor, wave):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    names = ("kernel", "random_forest", "boosted_stumps", "direct_dipoles", "dipoles")
    labels = ("Gaussian", "Random forest", "Boosted stumps", "Direct dipoles", "Kernel to dipoles")
    for env, shift in (("stable", -.19), ("changed", .19)):
        scores = sensor["pooled"][env]["invensense"]["all"]["models"]
        axes[0].bar(np.arange(len(names))+shift, [scores[n]["mean"] for n in names], width=.38,
                    yerr=[scores[n]["sd"] for n in names], label=env, capsize=3)
    axes[0].set(xticks=np.arange(len(names)), xticklabels=labels, ylabel="Vector RMSE (reference units)",
                title="Real sensor data: reference instrument")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].legend()
    for name in ("random_forest", "boosted_stumps", "maxwell_stumps", "phase_random_forest"):
        rows = [r for r in wave["summary"] if r["model"] == name]
        axes[1].errorbar([r["time"] for r in rows], [100*r["relative_l2"]["mean"] for r in rows],
            yerr=[100*r["relative_l2"]["sd"] for r in rows], marker="o", capsize=3, label=name.replace("_", " "))
    axes[1].axvspan(0, 1, alpha=.1, color="gray", label="Training time interval")
    axes[1].set(xlabel="Time (dimensionless)", ylabel="Relative field L2 error (%)",
                title="Controlled wave: same phases at later times")
    axes[1].legend(fontsize=8)
    fig.savefig(OUTPUT / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-only", action="store_true", help="Reuse saved sensor comparison and rerun wave controls.")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    if args.wave_only:
        sensor = json.loads((OUTPUT / "sensor_results.json").read_text())
    else:
        sensor = sensor_experiment()
        (OUTPUT / "sensor_results.json").write_text(json.dumps(sensor, indent=2, allow_nan=False)+"\n")
    wave = wave_experiment()
    (OUTPUT / "wave_results.json").write_text(json.dumps(wave, indent=2, allow_nan=False)+"\n")
    plot(sensor, wave)
    paths = (Path(__file__), ROOT/"dipole_field/predictions.npz", ROOT/"dipole_field/results.json",
             ROOT/"results/results.json", ROOT/"run_experiment.py")
    manifest = dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        elapsed_seconds=time.perf_counter()-started, wave_only=args.wave_only,
        inputs_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUTPUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("Complete:", OUTPUT, "elapsed", manifest["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()

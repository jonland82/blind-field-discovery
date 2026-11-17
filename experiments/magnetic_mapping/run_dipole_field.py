"""Project a real-data kernel into a magnetostatic dipole family."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import hashlib
import json
import numpy as np
from threadpoolctl import threadpool_limits
from dipole_models import fit_path
from models import FieldMap
from fetch_data import SENSORS
from run_experiment import (ROOT, SOURCE, TRAIN_TRIALS, VALIDATION_TRIALS, TEST_TRIALS,
                            SHIFT_TRIALS, load_recording, spatial_split, verify_archive,
                            vector_rmse, in_hole)

OUTPUT = ROOT / "dipole_field"
COUNTS = (4, 7, 10)
DEPTHS = (.25, .5, 1.)
RIDGES = (1e-5, 1e-3, .1, 1.)


def run():
    provenance = verify_archive(ROOT.parents[1] / "data/aalto_magnetic")
    old = json.loads((ROOT / "results/results.json").read_text())
    recordings = {(SOURCE, t): load_recording(ROOT.parents[1] / "data/aalto_magnetic", SOURCE, t)
                  for t in TRAIN_TRIALS + VALIDATION_TRIALS}
    x, y, train_audit = spatial_split(recordings, TRAIN_TRIALS, old["protocol"]["train_per_trial_cap"])
    xv, yv, val_audit = spatial_split(recordings, VALIDATION_TRIALS, 600)
    config = old["model_selection"]["unrestricted"]["selected"]
    teacher = FieldMap("unrestricted", config["length_metres"], config["ridge"]).fit(x, y)
    teacher_val = vector_rmse(yv, teacher.predict(xv))
    np.testing.assert_allclose(teacher_val, old["model_selection"]["unrestricted"]["validation_vector_rmse"])
    candidates = []
    for count in COUNTS:
        gx, gy = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max(), count),
                             np.linspace(x[:, 1].min(), x[:, 1].max(), count))
        for depth in DEPTHS:
            sources = np.column_stack((gx.ravel(), gy.ravel(), np.full(count**2, -depth)))
            for ridge, model in fit_path(x, teacher.predict(x), sources, RIDGES):
                candidates.append((dict(grid_side=count, depth_metres=depth, ridge=ridge,
                    validation_vector_rmse=vector_rmse(yv, model.predict(xv))), model))
    selected, student = min(candidates, key=lambda item: item[0]["validation_vector_rmse"])
    direct = next(fit_path(x, y, student.sources, [selected["ridge"]]))[1]
    print("Frozen before test scoring:", selected, flush=True)
    # Only now open previously scored test acquisitions. No tuning on these labels.
    archive = np.load(ROOT / "results/predictions.npz")
    saved = dict(training_xy=x, training_actual=y, validation_xy=xv, validation_actual=yv)
    rows, pooled = [], {}
    for environment, trials in (("stable", TEST_TRIALS), ("changed", SHIFT_TRIALS)):
        pooled[environment] = {}
        for sensor in SENSORS:
            actuals, predictions = [], {k: [] for k in ("kernel", "dipoles", "direct_dipoles", "constant")}
            for trial in trials:
                prefix = f"{sensor}_{trial}"
                query, actual = archive[prefix+"_x"], archive[prefix+"_actual_aligned"]
                values = dict(kernel=teacher.predict(query), dipoles=student.predict(query),
                              direct_dipoles=direct.predict(query), constant=np.tile(y.mean(axis=0), (len(query), 1)))
                np.testing.assert_allclose(values["kernel"], archive[prefix+"_unrestricted"], atol=1e-10)
                saved[prefix+"_xy"], saved[prefix+"_actual"] = query, actual
                actuals.append(actual)
                for name, value in values.items():
                    predictions[name].append(value)
                    saved[prefix+"_"+name] = value
                    hole = in_hole(query)
                    rows.append(dict(sensor=sensor, trial=trial, environment=environment, model=name,
                        n=len(query), vector_rmse=vector_rmse(actual, value), hole_n=int(hole.sum()),
                        hole_rmse=vector_rmse(actual[hole], value[hole]) if hole.any() else None))
            actual = np.concatenate(actuals)
            pooled[environment][sensor] = {}
            for name, parts in predictions.items():
                value = np.concatenate(parts)
                pooled[environment][sensor][name] = dict(vector_rmse=vector_rmse(actual, value),
                    component_correlations=[float(np.corrcoef(actual[:, j], value[:, j])[0, 1])
                                            if np.std(value[:, j]) > 1e-10 else None for j in range(3)])
    result = dict(selected=selected, candidates=[c for c, _ in candidates], teacher_validation_rmse=teacher_val,
        field=student.to_dict(), direct_control=direct.to_dict(), pooled=pooled, evaluations=rows,
        provenance=provenance, training_rows=train_audit, validation_rows=val_audit,
        protocol=dict(train=[1, 2], validation=[3], stable_test=[4, 5], stress=[6, 7, 8, 9],
            grid_sides=COUNTS, depths_metres=DEPTHS, ridges=RIDGES,
            selection="Minimum measured source-validation vector RMSE; all choices frozen before test labels.",
            targets="Kernel predictions at actual training positions; no synthetic observations.",
            assumptions="Flat observation plane z=0; equivalent dipoles at assumed z=-depth; static source-free air above sheet.",
            calibration=old["device_alignment"], original_protocol=old["protocol"],
            limitations="Previously examined tests: exploratory reuse. Equivalent sources are nonunique, not identified room objects. Reference units, not SI moments."),
        code_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                     ("dipole_models.py", "run_dipole_field.py", "models.py", "run_experiment.py")})
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(OUTPUT/"predictions.npz", **saved)
    plot(saved, teacher, student, x)
    lines = ["# Real measurements reconstructed with magnetic dipole fields", "",
             "See [the derivation and protocol](THEORY.md). All errors are vector RMSE in reference recording units.", "",
             f"Selected by trial 3: {selected['grid_side']**2} equivalent dipoles, depth {selected['depth_metres']} m, ridge {selected['ridge']}. Teacher validation error {teacher_val:.4f}; dipoles {selected['validation_vector_rmse']:.4f}.", "",
             "| Environment | Instrument | Constant | Kernel | Kernel to dipoles | Direct-data dipoles |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for environment, sensors in pooled.items():
        for sensor, scores in sensors.items():
            lines.append(f"| {environment} | {sensor} | " + " | ".join(f"{scores[k]['vector_rmse']:.4f}" for k in ("constant", "kernel", "dipoles", "direct_dipoles"))+" |")
    lines += ["", "[Shape comparison](comparison.png). [Full coefficients, component correlations, per-trial and spatial-hole errors](results.json).",
              "", "Source-free Maxwell structure is supplied by the dipole basis. Measurement agreement tests the reconstruction, not independent discovery of Maxwell or identification of the actual sources."]
    (OUTPUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(pooled, indent=2))


def plot(saved, teacher, student, x):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    # Fixed first test acquisition, all components, no favorable window selection.
    prefix = "invensense_4"
    xy = saved[prefix+"_xy"]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    cut = np.column_stack((np.linspace(x[:, 0].min(), x[:, 0].max(), 400),
                           np.full(400, np.median(x[:, 1]))))
    kt, ds = teacher.predict(cut), student.predict(cut)
    for j in range(3):
        axes[j, 0].scatter(distance, saved[prefix+"_actual"][:, j], s=5, c="gray", alpha=.5, label="Measured")
        axes[j, 0].plot(distance, saved[prefix+"_kernel"][:, j], lw=1, label="Learned kernel")
        axes[j, 0].plot(distance, saved[prefix+"_dipoles"][:, j], lw=1, label="Magnetic dipole reconstruction")
        axes[j, 1].plot(cut[:, 0], kt[:, j], label="Learned kernel")
        axes[j, 1].plot(cut[:, 0], ds[:, j], label="Magnetic dipole reconstruction")
        axes[j, 0].set(ylabel=f"B{j+1} (reference units)", xlabel="Distance along trial-4 sampled path (m)")
        axes[j, 1].set(xlabel="x (m)", ylabel=f"B{j+1} (reference units)")
        for ax in axes[j]:
            ax.grid(alpha=.2)
    axes[0, 0].set_title("Held-out real measurements: full trial 4")
    axes[0, 1].set_title(f"Model shapes on fixed cut y={cut[0, 1]:.2f} m\nNo measured ground truth on this exact cut")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.savefig(OUTPUT/"comparison.png", dpi=160)
    fig.savefig(OUTPUT/"comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()

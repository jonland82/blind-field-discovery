"""Audit saved experimental scores, model selection, and front propagation."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_sensor_scores_and_selection_reproduce():
    result = json.loads((ROOT / "tree_comparison/sensor_results.json").read_text())
    data = np.load(ROOT / "tree_comparison/sensor_predictions.npz")
    for selection in result["selection"].values():
        assert selection["selected"]["validation_rmse_mean"] == min(
            r["validation_rmse_mean"] for r in selection["candidates"])
    for row in result["per_trial"]:
        prefix = f"{row['sensor']}_{row['trial']}"
        actual, pred = data[prefix+"_actual"], data[prefix+"_"+row["model"]]
        rmse = np.sqrt(np.mean(np.sum((pred-actual)**2, axis=2), axis=1))
        np.testing.assert_allclose(rmse, row["rmse_by_seed"], rtol=1e-12)


def test_fronts_reconstruct_predictions_and_preserve_error():
    result = json.loads((ROOT / "tree_comparison/wave_results.json").read_text())
    data = np.load(ROOT / "tree_comparison/wave_predictions.npz")
    xi = data["phase_grid"]
    truth = np.repeat((np.exp(-(xi/.7)**2)*np.cos(4*xi))[:, None], 2, axis=1)
    for run in result["runs"]:
        seed = run["seed"]
        f = run["fitted_profile"]
        reconstructed = f["baseline"] + (xi[:, None] > np.array(f["thresholds"])) @ np.array(f["jumps"])
        initial = data[f"seed{seed}_maxwell_stumps_t0.0"]
        np.testing.assert_allclose(initial[:, 0], reconstructed, atol=1e-12)
        for name in ("maxwell_stumps", "phase_random_forest"):
            initial = data[f"seed{seed}_{name}_t0.0"]
            for t in (0., .5, 1., 2., 4.):
                pred = data[f"seed{seed}_{name}_t{t}"]
                np.testing.assert_array_equal(pred, initial)
                np.testing.assert_array_equal(pred[:, 0], pred[:, 1])
        assert run["front_check"]["max_abs_delta_coefficient"] == 0
        assert run["front_check"]["wrong_speed_08_max_abs_delta_coefficient"] > 0
        for row in run["tests"]:
            pred = data[f"seed{seed}_{row['model']}_t{row['time']}"]
            np.testing.assert_allclose(np.linalg.norm(pred-truth)/np.linalg.norm(truth), row["relative_l2"])

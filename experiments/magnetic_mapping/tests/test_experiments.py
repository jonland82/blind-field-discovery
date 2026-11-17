"""Numerical and leakage checks. Artificial fixtures here are only software tests.

They are written to temporary directories and never used as experimental data
or mixed into the public measured-data results.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controlled_experiment as controlled
from models import FieldMap, potential_kernel, rbf, planar_curl
import run_experiment as pilot


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(limits=1):
        yield


def test_potential_covariance_matches_scalar_derivatives():
    x = np.array([[.3, -.2], [.8, 1.2]])
    z = np.array([[-.7, .6], [.2, -.4]])
    length, step = .9, 1e-4
    kernel = potential_kernel(x, z, length)
    for a in range(2):
        for b in range(2):
            da, db = np.eye(2)[a] * step, np.eye(2)[b] * step
            numeric = length**2 * (rbf(x+da, z+db, length) - rbf(x+da, z-db, length)
                                  - rbf(x-da, z+db, length) + rbf(x-da, z-db, length)) / (4 * step**2)
            np.testing.assert_allclose(kernel[a::2, b::2], numeric, rtol=2e-6, atol=2e-7)
    self_covariance = potential_kernel(x, x, length)
    np.testing.assert_allclose(self_covariance, self_covariance.T)
    assert np.linalg.eigvalsh(self_covariance).min() >= -1e-10


def test_potential_prediction_has_zero_planar_curl():
    rng = np.random.default_rng(42)
    x, y = rng.normal(size=(30, 2)), rng.normal(size=(30, 3))
    model = FieldMap("planar_potential", .7, .1).fit(x, y)
    curl = planar_curl(model, rng.normal(size=(20, 2)), step=1e-5)
    assert np.max(abs(curl)) < 1e-7


def test_spatial_and_trial_holdouts_do_not_enter_training():
    x = np.column_stack((np.linspace(-1, 4, 300), np.full(300, -1.5)))
    recordings = {(pilot.SOURCE, trial): pilot.Recording(
        pilot.SOURCE, trial, x.copy(), np.ones((300, 3))*trial,
        np.arange(300, dtype=float), np.arange(300), np.array([], dtype=int))
        for trial in (1, 2, 3, 4, 5)}
    train_x, train_y, audit = pilot.spatial_split(recordings, pilot.TRAIN_TRIALS, 50)
    assert not pilot.in_hole(train_x, pilot.BUFFER).any()
    assert set(train_y[:, 0]) == {1, 2}
    assert {row["trial"] for row in audit} == {1, 2}


def test_affine_alignment_recovers_scale_offset_and_axis_mixing():
    rng = np.random.default_rng(33)
    target = rng.normal(size=(100, 3))
    linear = np.array([[2, .1, .2], [-.3, 1.5, 0], [0, -.1, .7]])
    reference = target @ linear + np.array([4., -2., 1.])
    coefficients, _ = pilot.affine_fit(target, reference)
    np.testing.assert_allclose(pilot.apply_alignment(target, coefficients), reference, atol=1e-12)
    with pytest.raises(ValueError, match="spanning"):
        pilot.affine_fit(np.ones((100, 3)), reference)


def test_planned_calibration_has_training_pairs_and_no_heldout_sources():
    rows = controlled.planned_trials()
    assert len(rows) == 192
    assert len({r["trial_id"] for r in rows}) == 192
    train = {controlled.condition(r) for r in rows if r["split"] == "train"}
    validation = {controlled.condition(r) for r in rows if r["split"] == "validation"}
    geometry = {controlled.condition(r) for r in rows if r["split"] == "test_geometry"}
    assert not train & validation and not train & geometry
    calibration = [r for r in rows if r["split"] == "calibration"]
    assert len(calibration) == 12
    assert all(controlled.condition(r) in train and r["source_id"] != "M3" for r in calibration)
    assert not any(r["source_id"] == "M3" for r in rows if r["split"] in ("train", "validation", "calibration"))


def test_empty_collection_reports_pending_and_preserves_recordings(tmp_path):
    folder, output = tmp_path / "collection", tmp_path / "output"
    controlled.make_plan(folder)
    before = (folder / "samples.csv").read_bytes()
    controlled.make_plan(folder)
    assert (folder / "samples.csv").read_bytes() == before
    status = controlled.analyze(folder, output)
    assert status["status"] == "awaiting_measurements"
    assert status["complete_trials"] == 0
    assert not (output / "results.json").exists()


def artificial_recording_fixture(folder, test_offset=0.):
    """Known dipole and invertible sensor transform, for testing the analyzer only."""
    controlled.make_plan(folder)
    beta = np.array([7., -2., -12., .03, .4, -.3, .2])
    transform = np.array([[1.2, .1, 0.], [0., .8, -.05], [.02, 0., 1.1]])
    bias = np.array([2., -3., 1.])
    with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=controlled.SAMPLE_FIELDS)
        writer.writeheader()
        for row in controlled.planned_trials():
            measurement = controlled.dipole_design(row) @ beta
            if row["device_id"] == "B":
                measurement = (measurement - bias) @ np.linalg.inv(transform)
            if row["split"].startswith("test_"):
                measurement = measurement + test_offset
            for time in np.linspace(0, 5, 31):
                writer.writerow({"trial_id": row["trial_id"], "timestamp_s": time,
                                 "u0": measurement[0], "u1": measurement[1], "u2": measurement[2],
                                 "unit": "artificial_unit_test_fixture_only"})


def test_controlled_runner_known_geometry_and_no_test_label_leakage(tmp_path):
    folder = tmp_path / "fixture"
    artificial_recording_fixture(folder)
    result = controlled.analyze(folder, tmp_path / "first")
    assert result["status"] == "analyzed_measured_collection"
    assert result["device_B_alignment"]["independent_trial_means"] == 12
    assert max(row["vector_rmse"] for row in result["evaluations"]
               if row["model"] == "dipole_background_bias") < 1e-9
    artificial_recording_fixture(folder, test_offset=1000.)
    altered = controlled.analyze(folder, tmp_path / "altered")
    assert altered["model_selection"] == result["model_selection"]
    assert altered["device_B_alignment"] == result["device_B_alignment"]
    assert altered["dipole"] == result["dipole"]
    with np.load(tmp_path / "first" / "predictions.npz") as first, np.load(tmp_path / "altered" / "predictions.npz") as second:
        for name in ("unrestricted", "linear_geometry", "shuffled_labels", "dipole_background_bias", "training_mean"):
            np.testing.assert_array_equal(first[name], second[name])


def test_corrupt_or_nonfinite_collection_is_rejected(tmp_path):
    folder = tmp_path / "collection"
    controlled.make_plan(folder)
    trial = controlled.planned_trials()[0]["trial_id"]
    with (folder / "samples.csv").open("a", encoding="utf-8") as stream:
        stream.write(f"{trial},0,nan,1,2,test_units\n")
    with pytest.raises(ValueError, match="Non-finite"):
        controlled.load_collection(folder)


def test_saved_public_predictions_reproduce_reported_errors():
    root = Path(__file__).resolve().parents[1]
    if not (root / "results" / "results.json").exists():
        pytest.skip("Run the public-data pilot before checking its archived predictions")
    result = json.loads((root / "results" / "results.json").read_text())
    with np.load(root / "results" / "predictions.npz") as archive:
        for row in result["evaluations"]:
            prefix = f"{row['sensor']}_{row['trial']}"
            error = pilot.vector_rmse(archive[f"{prefix}_actual_aligned"], archive[f"{prefix}_{row['model']}"])
            assert error == pytest.approx(row["scores"]["all"]["vector_rmse"], rel=1e-12)
    assert result["synthetic_observations"] is False
    assert result["exact_two_phone_experiment_completed"] is False

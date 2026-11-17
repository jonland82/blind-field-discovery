"""Physics-coordinate and leakage checks; artificial fixtures are not evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiu_models import WorldMap, world_covariance
from models import rbf
import run_fiu


@pytest.fixture(autouse=True)
def limit_threads():
    with threadpool_limits(limits=1):
        yield


def test_world_potential_covariance_matches_scalar_derivatives():
    x = np.array([[.03, -.02, .05], [.08, .12, -.04]])
    z = np.array([[-.07, .06, .02], [.02, -.04, -.03]])
    length, step = .09, 1e-5
    covariance = world_covariance(x, z, length)
    for a in range(3):
        for b in range(3):
            da, db = np.eye(3)[a] * step, np.eye(3)[b] * step
            numeric = length**2 * (rbf(x+da, z+db, length) - rbf(x+da, z-db, length)
                                  - rbf(x-da, z+db, length) + rbf(x-da, z-db, length)) / (4*step**2)
            np.testing.assert_allclose(covariance[a::3, b::3], numeric, rtol=3e-6, atol=1e-7)
    self_covariance = world_covariance(x, x, length)
    np.testing.assert_allclose(self_covariance, self_covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(self_covariance).min() > -1e-10


@pytest.mark.parametrize("kind", ["vector", "potential"])
def test_shared_world_prediction_is_independent_of_global_axis_choice(kind):
    rng = np.random.default_rng(21)
    x, y = rng.normal(size=(25, 3)) * .1, rng.normal(size=(25, 3))
    rotations = Rotation.random(25, random_state=rng).as_matrix()
    query = rng.normal(size=(12, 3)) * .1
    query_rotation = Rotation.random(12, random_state=rng).as_matrix()
    basis_change = Rotation.random(random_state=rng).as_matrix()
    original = WorldMap(kind, .08, .03).fit(x, rotations, y)
    changed = WorldMap(kind, .08, .03).fit(x @ basis_change.T, basis_change @ rotations, y)
    np.testing.assert_allclose(original.predict(query, query_rotation),
                               changed.predict(query @ basis_change.T, basis_change @ query_rotation), atol=1e-10)


def test_known_vector_and_sensor_bias_are_recovered_from_independent_rotations():
    rng = np.random.default_rng(29)
    rotations = Rotation.random(60, random_state=rng).as_matrix()
    x = rng.normal(size=(60, 3)) * .1
    field, bias = np.array([.1, -.25, .3]), np.array([.02, -.01, .03])
    measured = np.einsum("nji,j->ni", rotations, field) + bias
    model = WorldMap("vector", .06, .01).fit(x, rotations, measured)
    np.testing.assert_allclose(model.background, field, atol=1e-12)
    np.testing.assert_allclose(model.bias, bias, atol=1e-12)
    new_rotation = Rotation.random(10, random_state=rng).as_matrix()
    expected = np.einsum("nji,j->ni", new_rotation, field) + bias
    np.testing.assert_allclose(model.predict(x[:10], new_rotation), expected, atol=1e-12)


def test_loader_uses_camera_not_magnetic_orientation_and_preserves_first_observation(tmp_path):
    folder = tmp_path / "recordings"
    folder.mkdir()
    raw = np.zeros((120, 23))
    raw[:, 0] = np.arange(120) * 8
    raw[:, 1:4] = [.1, .2, .3]
    raw[:, 7] = 1.  # Optical quaternion w.
    raw[:, 8:12] = np.nan  # Magnetic filter quaternion is deliberately unusable.
    raw[:, 18:21] = [.2, .3, .4]
    raw[:, 22] = 1.
    raw[2, 22] = 0.
    raw[3, 4:8] = 0.
    np.savetxt(folder / "rec01.csv", raw, delimiter=",", header=",".join(f"c{i}" for i in range(23)), comments="")
    recording = run_fiu.load_recording(tmp_path, 1)
    assert recording.original_rows[0] == 0
    assert recording.t[0] == 0.
    assert len(recording.x) == 118
    assert recording.qc["untracked_rows"] == 1
    assert recording.qc["invalid_quaternion_rows"] == 1
    assert recording.qc["nonfinite_relevant_rows"] == 0


def test_whole_recording_holdout_and_no_test_label_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(run_fiu, "TRAIN", (1, 2))
    monkeypatch.setattr(run_fiu, "VALID", (3,))
    monkeypatch.setattr(run_fiu, "TEST", (4, 5))
    monkeypatch.setattr(run_fiu, "LENGTHS", (.06,))
    monkeypatch.setattr(run_fiu, "RIDGES", (.01,))
    monkeypatch.setattr(run_fiu, "ORIENTATION_LENGTHS", (1.5,))
    monkeypatch.setattr(run_fiu, "CONVENTIONS", ("as_recorded",))
    monkeypatch.setattr(run_fiu, "verify", lambda path: {"commit": "temporary-test-fixture", "files": []})
    monkeypatch.setattr(run_fiu, "plot", lambda result, output: None)
    offset = [0.]
    reads = []

    def load_fixture(data, number):
        reads.append(number)
        rng = np.random.default_rng(number)
        x = rng.normal(size=(120, 3)) * .04
        rotation = Rotation.random(120, random_state=rng).as_matrix()
        y = np.einsum("nji,j->ni", rotation, [.1, -.2, .3])
        if number in run_fiu.TEST:
            y += offset[0]
        return run_fiu.Recording(number, x, rotation, y, np.arange(120), np.arange(120),
                                 {"recording": number, "fixture": True})

    monkeypatch.setattr(run_fiu, "load_recording", load_fixture)
    first = run_fiu.run(tmp_path, tmp_path / "first", train_per_recording=10, test_per_recording=20)
    offset[0] = 100.
    second = run_fiu.run(tmp_path, tmp_path / "second", train_per_recording=10, test_per_recording=20)
    assert reads == [1, 2, 3, 4, 5] * 2
    assert first["model_selection"] == second["model_selection"]
    assert first["world_model_nuisance_fits"] == second["world_model_nuisance_fits"]
    assert {r["recording"] for r in first["training_rows"]} == {1, 2}
    assert {r["recording"] for r in first["validation_rows"]} == {3}
    assert {r["recording"] for r in first["test_rows"]} == {4, 5}
    with np.load(tmp_path / "first/predictions.npz") as a, np.load(tmp_path / "second/predictions.npz") as b:
        for number in (4, 5):
            for name in ("constant", "pose_learner", "shared_vector", "static_potential", "shuffled_pose"):
                np.testing.assert_array_equal(a[f"rec{number:02d}_{name}"], b[f"rec{number:02d}_{name}"])


def test_archived_public_metrics_match_saved_predictions():
    output = Path(__file__).resolve().parents[1] / "public_transfer"
    if not (output / "results.json").exists():
        pytest.skip("Run the public optical-pose study before checking its artifacts")
    result = json.loads((output / "results.json").read_text())
    assert result["synthetic_observations"] is False
    assert result["requires_new_measurements"] is False
    assert result["protocol"]["onboard_magnetic_orientation_used"] is False
    with np.load(output / "predictions.npz") as archive:
        for row in result["evaluations"]:
            prefix = f"rec{row['recording']:02d}"
            error = run_fiu.vector_rmse(archive[f"{prefix}_actual"], archive[f"{prefix}_{row['model']}"])
            assert error == pytest.approx(row["vector_rmse"], rel=1e-12)

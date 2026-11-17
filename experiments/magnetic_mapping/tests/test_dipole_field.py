"""Independent physical identities and reproducibility checks."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dipole_models import DipoleField, design


def test_axial_and_equatorial_dipole():
    field = DipoleField([[0, 0, 0]], [0, 0, 0], [[0, 0, 2]])
    np.testing.assert_allclose(field.predict([[0, 0, 2], [2, 0, 0]]), [[0, 0, .5], [0, 0, -.25]])


def test_full_three_dimensional_curl_and_divergence():
    field = DipoleField([[.2, -.3, -1], [1, 1, -.5]], [2, 3, 4], [[1, 2, 3], [-1, 1, 2]])
    points = np.array([[0., 0., 0.], [.5, -.2, .3], [-.2, .7, .1]])
    step = 1e-5
    jac = np.stack([(field.predict(points+step*axis)-field.predict(points-step*axis))/(2*step)
                    for axis in np.eye(3)], axis=-1)
    np.testing.assert_allclose(np.trace(jac, axis1=1, axis2=2), 0, atol=2e-8)
    np.testing.assert_allclose(jac-jac.transpose(0, 2, 1), 0, atol=2e-8)


def test_saved_coefficients_and_scores_reproduce():
    folder = Path(__file__).resolve().parents[1]/"dipole_field"
    result = json.loads((folder/"results.json").read_text())
    spec = result["field"]
    field = DipoleField(spec["sources_metres"], spec["background"], spec["effective_moments"])
    data = np.load(folder/"predictions.npz")
    for row in result["evaluations"]:
        prefix = f"{row['sensor']}_{row['trial']}"
        predicted = data[prefix+"_"+row["model"]]
        actual = data[prefix+"_actual"]
        np.testing.assert_allclose(np.sqrt(np.mean(np.sum((predicted-actual)**2, axis=1))), row["vector_rmse"])
        if row["model"] == "dipoles":
            np.testing.assert_allclose(field.predict(data[prefix+"_xy"]), predicted, atol=1e-10)

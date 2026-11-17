"""Scientific controls, independent calculus and saved-result reproduction."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


learner = load("hidden_wire_learner_tests", "learner.py")
world = load("hidden_wire_world_tests", "world.py")


def finite_difference(function, x, step=1e-5):
    return np.stack([(function(x+step*axis)-function(x-step*axis))/(2*step)
                     for axis in np.eye(3)], axis=-1)


def test_oracle_known_values_calculus_and_loop_integral():
    # Axis/equator values are evaluated independently of fitting.
    np.testing.assert_allclose(world.wire_field(np.array([[1., 0., 2.], [0., -2., -3.]])),
                               [[0., .4, 0.], [-.3, 0., 0.]])
    x = np.array([[.3, -.7, 2.], [-1., .6, -1.]])
    np.testing.assert_allclose(world.wire_jacobian(x), finite_difference(world.wire_field, x),
                               rtol=1e-7, atol=1e-8)
    theta = np.linspace(0, 2*np.pi, 1200, endpoint=False)
    for radius in (.2, 1., 3.):
        points = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.full(len(theta), 2.)))
        tangent = radius*np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(len(theta))))
        circulation = np.sum(world.wire_field(points)*tangent)*2*np.pi/len(theta)
        np.testing.assert_allclose(circulation, .8*np.pi, rtol=1e-12)


def test_kernel_and_rational_calculus_without_singularity_clipping():
    rng = np.random.default_rng(13)
    x, query = world.sample_inputs(rng, 90), world.sample_inputs(rng, 30)
    with threadpool_limits(limits=1):
        teacher = learner.GaussianTeacher(.6, .01).fit(x, world.wire_field(x))
        expression, _ = learner.discover(x, world.wire_field(x), query, world.wire_field(query))
    np.testing.assert_allclose(teacher.jacobian(query), finite_difference(teacher.predict, query),
                               rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(expression.jacobian(query), finite_difference(expression.predict, query),
                               rtol=2e-6, atol=1e-8)
    assert not np.isfinite(expression.predict(np.array([[0., 0., 2.]]))).all()


def test_fitting_has_no_oracle_access_and_does_not_require_origin_or_fixed_strength():
    rng = np.random.default_rng(17)
    x, xv = world.sample_inputs(rng, 240), world.sample_inputs(rng, 160)
    center = (.35, -.25)
    y = world.wire_field(x, center=center, coefficient=.37)
    yv = world.wire_field(xv, center=center, coefficient=.37)
    with threadpool_limits(limits=1), patch.object(world, "wire_field", side_effect=AssertionError("Oracle leakage")):
        teacher, _ = learner.fit_teacher(x, y, xv, yv)
        # Test generic algebra on direct targets; kernel distortion is separately
        # measured in the main experiment, not concealed behind this control.
        expression, _ = learner.discover(x, y, xv, yv)
        assert np.isfinite(teacher.predict(xv)).all()
    np.testing.assert_allclose(expression.predict(xv), yv, rtol=1e-6, atol=1e-7)
    tree = ast.parse((ROOT/"learner.py").read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("world" in name for name in imported)


def test_nonmagnetic_control_recovers_polynomial_instead_of_forcing_wire():
    result = json.loads((ROOT/"results/results.json").read_text())
    spec = result["negative_control"]["distilled"]["selected"]
    model = learner.Expression.from_dict(spec)
    assert np.count_nonzero(model.denominator) == 1 and model.denominator[0] == 1
    rng = np.random.default_rng(21)
    x = world.sample_inputs(rng, 100, radius=(2.3, 4.), current=(5., 8.))
    np.testing.assert_allclose(model.predict(x), world.polynomial_field(x), rtol=2e-6, atol=2e-6)


def test_saved_models_scores_and_frozen_selection_reproduce():
    result = json.loads((ROOT/"results/results.json").read_text())
    data = np.load(ROOT/"results/predictions.npz")
    assert len(result["runs"]) == 15
    for row in result["runs"]:
        prefix = f"{row['case']}_{row['seed']}_"
        for name in ("distilled", "direct"):
            model = learner.Expression.from_dict(row[name]["selected"])
            for domain, scores in row["evaluations"].items():
                query = data[prefix+domain+"_x"]
                saved = data[prefix+domain+"_"+name]
                np.testing.assert_allclose(model.predict(query), saved, atol=1e-12)
                truth = data[prefix+domain+"_oracle"]
                relative = np.linalg.norm(saved-truth)/np.linalg.norm(truth)
                np.testing.assert_allclose(relative, scores[name]["oracle_relative_rmse"], rtol=1e-12)
            chosen = row[name]["selected"]["coefficient_count"]
            eligible = [c for c in row[name]["candidates"] if c["validation_mse"] is not None
                        and c["validation_mse"] <= row[name]["selection_mse_tolerance"]]
            assert chosen == min(c["coefficient_count"] for c in eligible)


def test_recorded_source_protocol_and_sensor_hashes_match():
    import hashlib
    result = json.loads((ROOT/"results/results.json").read_text())
    assert hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest() == result["protocol_sha256"]
    for name, expected in result["implementation_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    for name, expected in result["input_sha256"].items():
        assert hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest() == expected

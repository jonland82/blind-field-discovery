"""Analytic calculus and trial-leakage checks; generated data are fixtures only."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from symbolic_models import AnalyticField, ScalarKernel
import run_symbolic_field as experiment


def test_kernel_derivative_matches_finite_difference():
    s = np.linspace(-2, 2, 50)
    model = ScalarKernel(.7, .01).fit(s, np.sin(s)+2)
    query, step = np.linspace(-1.8, 1.8, 31), 1e-5
    derivative = (model.predict(query+step)-model.predict(query-step))/(2*step)
    np.testing.assert_allclose(model.derivative(query), derivative, rtol=1e-6, atol=1e-7)


def test_riccati_analytic_solution_matches_independent_numerical_solver():
    # Hyperbolic, trigonometric, rational, exponential, and linear cases.
    coefficients = ([1.2, .3, -.8], [1., .1, .2], [0., 0., .3], [1., -.7, 0.], [.3, 0., 0.])
    query = np.linspace(0., .8, 61)
    for c in coefficients:
        model = AnalyticField("riccati", c, 0., 5.4, 5., 2., support=(0, 1, 2))
        z0 = (5.4-5.)/2.
        solution = solve_ivp(lambda t,z: [c[0]+c[1]*z[0]+c[2]*z[0]**2],
                            (0., .8), [z0], t_eval=query, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(model.predict(query), 5.+2.*solution.y[0], rtol=1e-9, atol=1e-9)
        step = 1e-5
        numerical = (model.predict(query+step)-model.predict(query-step))/(2*step)
        np.testing.assert_allclose(model.derivative(query), numerical, rtol=1e-6, atol=1e-7)


def test_solved_equation_has_tanh_branch_and_pole_is_visible():
    model = AnalyticField("riccati", [2., 0., -.5], .3, 3., 3., 4., support=(0, 2))
    s = np.linspace(-2., 2., 70)
    np.testing.assert_allclose(model.predict(s), 3.+8.*np.tanh(s-.3), rtol=1e-12)
    solved = model.to_dict()["tanh_equivalent"]
    np.testing.assert_allclose([solved["background"],solved["amplitude"],solved["center_metres"],
                               solved["width_metres"]], [3.,8.,.3,1.], rtol=1e-12)
    pole = AnalyticField("riccati", [0., 0., 1.], 0., 1., 0., 1., support=(2,))
    assert np.isnan(pole.predict(np.array([1.]))[0])


def test_geometry_band_and_interval_boundaries_do_not_duplicate_samples():
    x = np.column_stack(([-1., 0., .5, 1., 2.], [0., 0., .2, 0., 0.]))
    rec = SimpleNamespace(x=x, y=np.tile([3.,4.,0.], (5,1)), original_rows=np.arange(5),
                          trial=1, sensor=experiment.SOURCE)
    lower = experiment.band_sample(rec, 0., (-1.,1.,False), 100)
    upper = experiment.band_sample(rec, 0., (1.,2.,True), 100)
    assert lower["original_rows"].tolist() == [0,1]
    assert upper["original_rows"].tolist() == [3,4]
    np.testing.assert_allclose(lower["b"], 5.)


def test_test_readings_cannot_change_fitted_equations_or_predictions(tmp_path, monkeypatch):
    """Artificial fixture outputs stay in pytest's temporary directory."""
    offset = [0.]
    def fake_recording(data, sensor, trial):
        s = np.linspace(-1,1,80)
        field = 50.+10.*np.tanh(1.5*s) + (offset[0] if trial>=4 else 0.)
        return SimpleNamespace(x=np.column_stack((s,np.zeros_like(s))),
            y=np.column_stack((np.zeros_like(s),np.zeros_like(s),field)),
            trial=trial, sensor=sensor, original_rows=np.arange(len(s)))
    monkeypatch.setattr(experiment, "verify_archive", lambda data: {"fixture_only": True})
    monkeypatch.setattr(experiment, "load_recording", fake_recording)
    monkeypatch.setattr(experiment, "SENSORS", (experiment.SOURCE,))
    monkeypatch.setattr(experiment, "LENGTHS", (.2,.7))
    monkeypatch.setattr(experiment, "RIDGES", (.01,.1))
    monkeypatch.setattr(experiment, "write_report", lambda result: None)
    monkeypatch.setattr(experiment, "plot_results", lambda *args: None)
    with threadpool_limits(limits=1):
        monkeypatch.setattr(experiment, "OUTPUT", tmp_path/"first")
        first = experiment.run()
        offset[0] = 1000.
        monkeypatch.setattr(experiment, "OUTPUT", tmp_path/"altered")
        second = experiment.run()
    assert first["regions"] == second["regions"]
    assert first["protocol"] == second["protocol"]
    with np.load(tmp_path/"first/predictions.npz") as a, np.load(tmp_path/"altered/predictions.npz") as b:
        for name in a.files:
            if name.endswith(("_kernel", "_analytic_selected", "_best_riccati_diagnostic")):
                np.testing.assert_allclose(a[name], b[name], equal_nan=True)
        assert not np.array_equal(a["whole_band_invensense_4_actual"], b["whole_band_invensense_4_actual"])


def test_saved_real_predictions_reproduce_reported_errors():
    root = Path(__file__).resolve().parents[1]/"symbolic_field"
    report = json.loads((root/"results.json").read_text(encoding="utf-8"))
    assert not report["synthetic_physical_observations"]
    assert set(report["regions"]) == {"whole_band", "interval_1", "interval_2", "interval_3", "interval_4"}
    with np.load(root/"predictions.npz") as data:
        for row in report["evaluations"]:
            prefix = f"{row['region']}_{row['sensor']}_{row['trial']}"
            prediction = data[prefix+"_"+row["model"]]
            actual = data[prefix+"_actual"]
            assert len(actual) == row["n"]
            if row["rmse"] is not None:
                np.testing.assert_allclose(experiment.rmse(actual, prediction), row["rmse"], rtol=1e-12)

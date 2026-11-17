"""Checks for the scientific workflow and analytic kernel calculus."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("kernel_workflow_demo", Path(__file__).parents[1] / "run_demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_kernel_derivative_and_integral_match_numerical_calculus():
    centers = np.linspace(-2, 2, 20)
    values = np.column_stack((np.sin(centers), np.cos(centers)))
    fit = demo.KernelFit(.7, 1e-7).fit(centers, values)
    x, step = np.linspace(-1.8, 1.8, 19), 1e-5
    difference = (fit.predict(x+step) - fit.predict(x-step)) / (2*step)
    np.testing.assert_allclose(fit.derivative(x), difference, rtol=1e-6, atol=1e-7)
    integral_difference = (fit.integral(x+step, -2.) - fit.integral(x-step, -2.)) / (2*step)
    np.testing.assert_allclose(integral_difference, fit.predict(x), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fit.integral(np.array([-2.]), -2.), 0., atol=1e-14)


def test_learner_cannot_access_oracle_and_recovers_hidden_relation():
    samples = demo.make_samples(0., 20261116)
    with patch.object(demo, "hidden_functions", side_effect=AssertionError("Oracle leakage")):
        fit, selected, _, accepted = demo.learn(samples)
        symbolic = demo.recover_symbolic(fit, samples, selected)
    _, _, support, coefficients = selected
    assert accepted and support == (2,)
    np.testing.assert_allclose(coefficients, [85/36], rtol=1e-5)
    assert symbolic["selected_powers"] == [0, 2] and symbolic["closed_form_recovered"]
    np.testing.assert_allclose([symbolic["amplitude"], symbolic["length"]], [1.7, .9], rtol=1e-5)


def test_known_truth_recovery_and_integrated_prediction_with_noise():
    report, arrays = demo.run_case(.001, 20261116)
    assert report["equation_terms"] == ["J"]
    assert report["coefficient_relative_error"] < .001
    assert report["field_relative_rmse"] < .005
    assert report["derivative_relative_rmse"] < .03
    assert report["integrated_law_field_relative_rmse"] < .005
    # These checks use the known generator only as the verification oracle.
    np.testing.assert_allclose(arrays["integrated_law_b"][0], arrays["anchor_b"], atol=1e-12)
    symbolic = report["symbolic_recovery"]
    assert symbolic["field_relative_rmse"] < .001
    assert symbolic["source_relative_rmse"] < .002
    assert symbolic["derivative_relative_rmse"] < .002
    assert symbolic["extrapolated_field_relative_rmse"] < .001


def test_symbolic_recovery_handles_shifted_and_rescaled_function():
    x = np.linspace(-3., 3., 80)
    amplitude, length, center, source = 2.3, 1.2, .35, .6
    shape = 1/np.cosh((x-center)/length)**2
    y = np.column_stack((amplitude*np.tanh((x-center)/length), source*shape))
    fit = demo.KernelFit(.6, 1e-10).fit(x, y)
    samples = {"train_x": x, "discovery_x": np.linspace(-2.5, 2.5, 100),
               "law_validation_x": np.linspace(-2.49, 2.49, 97)}
    a = amplitude/(length*source)
    coupling = (1, 0., (2,), np.array([a]))
    symbolic = demo.recover_symbolic(fit, samples, coupling)
    assert symbolic["closed_form_recovered"]
    np.testing.assert_allclose([symbolic["amplitude"], symbolic["length"], symbolic["center"],
                               symbolic["source_amplitude"]],
                              [amplitude, length, center, source], rtol=1e-4, atol=1e-5)
    query, step = np.linspace(-2, 2, 31), 1e-5
    b, j, derivative = demo.evaluate_symbolic(query, symbolic)
    numerical = (demo.evaluate_symbolic(query+step, symbolic)[0]
                 -demo.evaluate_symbolic(query-step, symbolic)[0])/(2*step)
    np.testing.assert_allclose(derivative, numerical, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(derivative, symbolic["p"]-symbolic["q"]*b*b, atol=1e-12)
    np.testing.assert_allclose(derivative, a*j, atol=1e-12)


def test_field_only_search_does_not_force_tanh_for_a_linear_profile():
    x = np.linspace(-2, 2, 60)
    fit = demo.KernelFit(2., 1e-12).fit(x, np.column_stack((x, np.ones_like(x))))
    samples = {"train_x": x, "discovery_x": np.linspace(-1, 1, 71),
               "law_validation_x": np.linspace(-.99, .99, 59)}
    symbolic = demo.recover_symbolic(fit, samples, (1, 0., (2,), np.array([1.])))
    assert symbolic["selected_powers"] == [0]
    assert not symbolic["closed_form_recovered"]

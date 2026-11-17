"""Learned and geometric models of a measured inter-receiver response kernel."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

C = 299792458.0  # A supplied SI propagation speed, not estimated from this dataset.


def rbf(x, z, length):
    delta = (x[:, None, :] - z[None, :, :]) / length
    return np.exp(-.5 * np.sum(delta**2, axis=-1))


def fit_constant(reference, actual):
    """Per-frequency least-squares receiver filter, independent of geometry."""
    return np.sum(reference[:, None, :].conj() * actual, axis=0) / np.sum(abs(reference)**2, axis=0)


def fit_green_gain(propagation, reference, actual):
    design = propagation * reference[:, None, :]
    return np.sum(design.conj() * actual, axis=0) / np.sum(abs(design)**2, axis=0)


class KernelMap:
    def __init__(self, length, ridge):
        self.length, self.ridge = length, ridge

    def fit(self, x, quotient, reference):
        self.x = x.copy()
        self.shape = quotient.shape[1:]
        labels = quotient.reshape(len(x), -1)
        weights = np.mean(abs(reference)**2, axis=1)
        weights = np.clip(weights / np.mean(weights), .05, 20.)
        self.mean = np.sum(weights[:, None] * labels, axis=0) / weights.sum()
        matrix = rbf(x, x, self.length)
        matrix.flat[::len(x)+1] += self.ridge / weights + 1e-10
        self.coefficients = cho_solve(cho_factor(matrix, lower=True, check_finite=False),
                                      labels-self.mean, check_finite=False)
        return self

    def predict(self, x):
        values = rbf(x, self.x, self.length) @ self.coefficients + self.mean
        return values.reshape((len(x),) + self.shape)


def antenna_positions(array, horizontal_sign=1, vertical_sign=1):
    """Use published centers, directions, spacings and channel assignments.

    Axis-sign ambiguity in the published row/column layout is selected using
    validation data only, along with the complex phase convention.
    """
    assignments = np.asarray(array["assignments"])
    normal = np.asarray(array["direction"], dtype=float)
    normal /= np.linalg.norm(normal)
    horizontal = np.cross([0., 0., 1.], normal)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    result = {}
    rows, columns = assignments.shape
    for row in range(rows):
        for column in range(columns):
            result[int(assignments[row, column])] = (
                np.asarray(array["location"])
                + horizontal_sign * (column-(columns-1)/2) * array["spacingX"] * horizontal
                + vertical_sign * ((rows-1)/2-row) * array["spacingY"] * vertical)
    return result


def green_quotient(x, array, reference_port, output_ports, frequencies, configuration):
    positions = antenna_positions(array, configuration["horizontal_sign"], configuration["vertical_sign"])
    reference_distance = np.linalg.norm(x-np.asarray(positions[reference_port]), axis=1)
    output_distance = np.linalg.norm(x[:, None, :]-np.array([positions[p] for p in output_ports])[None, :, :], axis=2)
    if min(reference_distance.min(), output_distance.min()) <= 0:
        raise ValueError("Source position coincides with an antenna")
    delay_difference = (output_distance-reference_distance[:, None]) / C
    phase = configuration["phase_sign"] * 2j * np.pi * delay_difference[:, :, None] * frequencies[None, None, :]
    return reference_distance[:, None, None] / output_distance[:, :, None] * np.exp(phase)


def normalized_error(actual, prediction):
    denominator = np.sum(abs(actual)**2)
    if denominator <= 0:
        raise ValueError("Cannot score a zero-energy observation set")
    return float(np.sum(abs(actual-prediction)**2) / denominator)

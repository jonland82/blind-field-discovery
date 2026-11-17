"""Two matched kernel learners: independent components and a planar potential.

The potential model imposes only d_x B_y - d_y B_x = 0. It does not impose
three-dimensional magnetic divergence or the complete Maxwell system.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def rbf(x, z, length):
    delta = (np.asarray(x)[:, None, :] - np.asarray(z)[None, :, :]) / length
    return np.exp(-0.5 * np.sum(delta * delta, axis=-1))


def potential_kernel(x, z, length):
    """Covariance of a planar gradient, from scalar covariance length**2 * RBF."""
    delta = (np.asarray(x)[:, None, :] - np.asarray(z)[None, :, :]) / length
    kernel = rbf(x, z, length)
    blocks = (np.eye(2) - delta[..., :, None] * delta[..., None, :]) * kernel[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(2 * len(x), 2 * len(z))


def solve(kernel, targets, ridge):
    regularized = kernel.copy()
    regularized.flat[::len(kernel) + 1] += ridge + 1e-10
    return cho_solve(cho_factor(regularized, lower=True, check_finite=False),
                     targets, check_finite=False)


class LinearMap:
    """Simple empirical baseline, with a fitted intercept and optional ridge."""
    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, x, y):
        self.center = np.mean(x, axis=0)
        self.mean = np.mean(y, axis=0)
        centered = np.asarray(x) - self.center
        self.weights = solve(centered.T @ centered, centered.T @ (y - self.mean), self.ridge)
        return self

    def predict(self, x):
        return (np.asarray(x) - self.center) @ self.weights + self.mean


class FieldMap:
    def __init__(self, kind, length=1.0, ridge=0.01):
        if kind not in ("unrestricted", "planar_potential"):
            raise ValueError(f"Unknown field model: {kind}")
        if length <= 0 or ridge <= 0:
            raise ValueError("Length and ridge must be positive")
        self.kind, self.length, self.ridge = kind, length, ridge

    def fit(self, x, y):
        self.x = np.asarray(x, dtype=float).copy()
        self.mean = np.mean(y, axis=0)
        centered = np.asarray(y) - self.mean
        kernel = rbf(self.x, self.x, self.length)
        if self.kind == "unrestricted":
            self.weights = solve(kernel, centered, self.ridge)
        else:
            self.xy_weights = solve(potential_kernel(self.x, self.x, self.length),
                                    centered[:, :2].reshape(-1), self.ridge)
            self.z_weights = solve(kernel, centered[:, 2], self.ridge)
        return self

    def predict(self, x, chunk=400):
        x = np.asarray(x)
        predictions = []
        for start in range(0, len(x), chunk):
            query = x[start:start + chunk]
            kernel = rbf(query, self.x, self.length)
            if self.kind == "unrestricted":
                prediction = kernel @ self.weights
            else:
                xy = (potential_kernel(query, self.x, self.length) @ self.xy_weights).reshape(-1, 2)
                prediction = np.column_stack((xy, kernel @ self.z_weights))
            predictions.append(prediction + self.mean)
        return np.concatenate(predictions) if predictions else np.empty((0, 3))


def planar_curl(model, x, step=1e-3):
    """Diagnostic of an interpolated model, not independent measured derivatives."""
    dx = np.array([step, 0.0])
    dy = np.array([0.0, step])
    derivative_x = (model.predict(x + dx) - model.predict(x - dx)) / (2 * step)
    derivative_y = (model.predict(x + dy) - model.predict(x - dy)) / (2 * step)
    return derivative_x[:, 1] - derivative_y[:, 0]

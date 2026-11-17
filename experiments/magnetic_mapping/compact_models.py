"""Explicit empirical field functions distilled from a measured-data kernel map."""
from itertools import product
import numpy as np

from models import rbf


class CompactField:
    def __init__(self, center, scale, powers, coefficients, centers=None, length=None):
        self.center = np.asarray(center)
        self.scale = np.asarray(scale)
        self.powers = np.asarray(powers, dtype=int)
        self.coefficients = np.asarray(coefficients)
        self.centers = np.empty((0, 2)) if centers is None else np.asarray(centers)
        self.length = length

    def features(self, x):
        z = (np.asarray(x) - self.center) / self.scale
        polynomial = np.prod(z[:, None, :] ** self.powers[None, :, :], axis=2)
        if len(self.centers):
            return np.column_stack((polynomial, rbf(np.asarray(x), self.centers, self.length)))
        return polynomial

    def predict(self, x):
        return self.features(x) @ self.coefficients

    def gradient(self, x):
        """d output channel / d coordinate; shape (queries, 3, 2)."""
        x = np.asarray(x)
        z = (x-self.center)/self.scale
        gradient = np.zeros((len(x), self.coefficients.shape[1], 2))
        for axis in (0, 1):
            powers = self.powers.copy()
            factors = powers[:, axis].copy()
            powers[:, axis] = np.maximum(powers[:, axis]-1, 0)
            differentiated = np.prod(z[:, None, :] ** powers[None, :, :], axis=2)
            differentiated *= factors / self.scale[axis]
            gradient[:, :, axis] = differentiated @ self.coefficients[:len(self.powers)]
            if len(self.centers):
                delta = x[:, None, axis] - self.centers[None, :, axis]
                differentiated_kernel = -delta/self.length**2 * rbf(x, self.centers, self.length)
                gradient[:, :, axis] += differentiated_kernel @ self.coefficients[len(self.powers):]
        return gradient

    def to_dict(self):
        return {"coordinate_center_metres": self.center.tolist(),
                "coordinate_scale_metres": self.scale.tolist(),
                "polynomial_powers": self.powers.tolist(),
                "coefficients_by_basis_then_channel": self.coefficients.tolist(),
                "gaussian_centers_metres": self.centers.tolist(),
                "gaussian_length_metres": self.length,
                "basis_count": len(self.coefficients),
                "scalar_field_weight_count": int(self.coefficients.size)}


def fit_polynomial(x, targets, degree):
    center, scale = np.mean(x, axis=0), np.std(x, axis=0)
    scale = np.maximum(scale, 1e-12)
    powers = [(a, b) for total in range(degree+1) for a, b in product(range(total+1), repeat=2)
              if a+b == total]
    model = CompactField(center, scale, powers, np.zeros((len(powers), targets.shape[1])))
    model.coefficients = np.linalg.lstsq(model.features(x), targets, rcond=None)[0]
    return model


def fit_sparse_path(x, targets, length, counts=(1, 2, 4, 8, 16, 32, 64)):
    """Shared-center multichannel matching pursuit, all geometry from training."""
    base = fit_polynomial(x, targets, 1)
    design = base.features(x)
    dictionary = rbf(x, x, length)
    norm = np.maximum(np.linalg.norm(dictionary, axis=0), 1e-12)
    selected, models = [], {}
    for step in range(1, max(counts)+1):
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        residual = targets-design@coefficients
        scores = np.sum((dictionary.T@residual / norm[:, None])**2, axis=1)
        if selected:
            scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
        design = np.column_stack((base.features(x), dictionary[:, selected]))
        if step in counts:
            models[step] = CompactField(base.center, base.scale, base.powers,
                np.linalg.lstsq(design, targets, rcond=None)[0], x[selected].copy(), length)
    return models


def teacher_gradient(teacher, x):
    delta = np.asarray(x)[:, None, :]-teacher.x[None, :, :]
    kernel = rbf(x, teacher.x, teacher.length)
    return np.einsum("nmd,mc->ncd", -delta/teacher.length**2*kernel[:, :, None], teacher.weights)

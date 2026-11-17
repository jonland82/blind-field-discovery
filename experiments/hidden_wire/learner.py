"""Data-only Gaussian teacher and generic polynomial/rational extraction.

There are no field equations, radial features, physical constants or oracle
imports here. General algebra and the finite search grammar are supplied.
"""
from itertools import product
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares


LENGTHS = (.35, .6, 1., 1.8, 3.)
RIDGES = (1e-8, 1e-6, 1e-4, .01, .1)
THRESHOLDS = (0., .003, .01, .03, .1, .2)


def mse(predicted, actual):
    if not np.isfinite(predicted).all():
        return float("inf")
    return float(np.mean(np.sum((predicted-actual)**2, axis=1)))


class GaussianTeacher:
    def __init__(self, length, ridge):
        self.length, self.ridge = length, ridge

    def fit(self, x, y):
        self.center, self.scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-12)
        self.x = (x-self.center)/self.scale
        self.mean = y.mean(axis=0)
        matrix = self.kernel(x)
        matrix.flat[::len(x)+1] += self.ridge
        self.weights = cho_solve(cho_factor(matrix, lower=True), y-self.mean)
        return self

    def kernel(self, x):
        delta = ((np.asarray(x)-self.center)/self.scale)[:, None, :]-self.x
        return np.exp(-.5*np.sum(delta*delta, axis=2)/self.length**2)

    def predict(self, x):
        return self.mean+self.kernel(x)@self.weights

    def jacobian(self, x):
        delta = ((np.asarray(x)-self.center)/self.scale)[:, None, :]-self.x
        dk = -delta*self.kernel(x)[:, :, None]/(self.length**2*self.scale)
        return np.einsum("nki,kj->nji", dk, self.weights)


def fit_teacher(x, y, xv, yv):
    candidates = []
    for length in LENGTHS:
        for ridge in RIDGES:
            teacher = GaussianTeacher(length, ridge).fit(x, y)
            candidates.append((mse(teacher.predict(xv), yv), teacher))
    score, teacher = min(candidates, key=lambda row: row[0])
    return teacher, dict(length=teacher.length, ridge=teacher.ridge,
                         validation_vector_rmse=float(np.sqrt(score)),
                         candidates=[dict(length=t.length, ridge=t.ridge, validation_mse=s)
                                     for s, t in candidates])


def powers(degree):
    return np.array(sorted((p for p in product(range(degree+1), repeat=3) if sum(p) <= degree),
                           key=lambda p: (sum(p), p)), dtype=int)


def features(x, exponents):
    return np.prod(np.asarray(x)[:, None, :]**exponents[None, :, :], axis=2)


def feature_jacobian(x, exponents):
    result = np.zeros((len(x), len(exponents), 3))
    for dim in range(3):
        active = exponents[:, dim] > 0
        reduced = exponents[active].copy()
        reduced[:, dim] -= 1
        result[:, active, dim] = features(x, reduced)*exponents[active, dim]
    return result


@dataclass
class Expression:
    exponents: np.ndarray
    denominator: np.ndarray
    numerators: np.ndarray  # term, component
    kind: str

    def predict(self, x):
        basis = features(x, self.exponents)
        q = basis@self.denominator
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return (basis@self.numerators)/q[:, None]

    def jacobian(self, x):
        basis, derivative = features(x, self.exponents), feature_jacobian(x, self.exponents)
        p, q = basis@self.numerators, basis@self.denominator
        dp = np.einsum("nki,kj->nji", derivative, self.numerators)
        dq = np.einsum("nki,k->ni", derivative, self.denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            return dp/q[:, None, None]-p[:, :, None]*dq[:, None, :]/q[:, None, None]**2

    def complexity(self):
        # One denominator coefficient fixes the arbitrary rational scale.
        return int(np.count_nonzero(self.numerators)+np.count_nonzero(self.denominator)-1)

    def to_dict(self):
        return dict(kind=self.kind, exponents=self.exponents.tolist(),
                    denominator=self.denominator.tolist(), numerators=self.numerators.tolist(),
                    coefficient_count=self.complexity(), formula=self.formula())

    @classmethod
    def from_dict(cls, data):
        return cls(np.array(data["exponents"]), np.array(data["denominator"]),
                   np.array(data["numerators"]), data["kind"])

    def formula(self):
        def poly(coefficients):
            parts = []
            for coefficient, exponent in zip(coefficients, self.exponents):
                if coefficient == 0:
                    continue
                term = "*".join(name+(f"^{power}" if power != 1 else "")
                                for name, power in zip(("x", "y", "I"), exponent) if power)
                parts.append(f"{coefficient:+.9g}"+("*"+term if term else ""))
            return " ".join(parts) or "0"
        denominator = poly(self.denominator)
        return [f"({poly(self.numerators[:, j])}) / ({denominator})" for j in range(3)]


def rational_candidate(x, targets, degree, threshold):
    exponents = powers(degree)
    basis = features(x, exponents)
    n, k = basis.shape
    # Stack all components with a shared unknown denominator. Homogeneous TLS
    # permits a zero constant denominator coefficient (e.g. Q=x^2+y^2).
    design = np.zeros((3*n, 4*k))
    for j in range(3):
        design[j*n:(j+1)*n, :k] = targets[:, j, None]*basis
        design[j*n:(j+1)*n, (j+1)*k:(j+2)*k] = -basis
    scales = np.maximum(np.linalg.norm(design, axis=0), 1e-14)
    normalized = design/scales
    active = np.ones(4*k, dtype=bool)
    for _ in range(8):
        _, _, vt = np.linalg.svd(normalized[:, active], full_matrices=False)
        weights = np.zeros(4*k)
        weights[active] = vt[-1]
        keep = np.abs(weights) >= threshold*np.max(np.abs(weights))
        if not keep[:k].any():
            return None
        if np.array_equal(keep, active):
            break
        active = keep
    # Refit after the final support update as well.
    _, _, vt = np.linalg.svd(normalized[:, active], full_matrices=False)
    coefficients = np.zeros(4*k)
    coefficients[active] = vt[-1]/scales[active]
    pivot = int(np.argmax(np.abs(coefficients[:k])))
    if abs(coefficients[pivot]) < 1e-14:
        return None
    coefficients /= coefficients[pivot]
    free = np.flatnonzero(active & (np.arange(4*k) != pivot))

    def unpack(values):
        c = np.zeros(4*k)
        c[pivot], c[free] = 1., values
        return c[:k], c[k:].reshape(3, k).T

    def residual(values):
        q, p = unpack(values)
        denominator = basis@q
        # Only the optimizer's residual is capped near a pole. The final model
        # is evaluated unmodified; nonfinite validation predictions lose.
        safe = np.where(np.abs(denominator) < 1e-10,
                        np.where(denominator < 0, -1e-10, 1e-10), denominator)
        return ((basis@p)/safe[:, None]-targets).ravel()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = least_squares(residual, coefficients[free], max_nfev=150,
                                ftol=1e-9, xtol=1e-9, gtol=1e-9)
        q, p = unpack(fit.x)
    except (ValueError, FloatingPointError):
        return None
    return Expression(exponents, q, p, f"rational_degree_{degree}_threshold_{threshold:g}")


def discover(x, targets, xv, observed_validation):
    """Same search/selection for teacher targets and direct measured targets."""
    candidates = []
    for degree in range(5):
        exponent = powers(degree)
        p = np.linalg.lstsq(features(x, exponent), targets, rcond=None)[0]
        q = np.zeros(len(exponent))
        q[0] = 1.
        model = Expression(exponent, q, p, f"polynomial_degree_{degree}")
        candidates.append((mse(model.predict(xv), observed_validation), model))
    for degree in (1, 2):
        for threshold in THRESHOLDS:
            model = rational_candidate(x, targets, degree, threshold)
            if model is not None:
                candidates.append((mse(model.predict(xv), observed_validation), model))
    best = min(score for score, _ in candidates)
    tolerance = 1.05*best+1e-10*float(np.mean(np.sum(targets**2, axis=1)))
    score, model = min((row for row in candidates if row[0] <= tolerance),
                       key=lambda row: (row[1].complexity(), row[0]))
    audit = [dict(kind=m.kind, coefficient_count=m.complexity(),
                  validation_mse=s if np.isfinite(s) else None) for s, m in candidates]
    return model, dict(selected=model.to_dict(), validation_vector_rmse=float(np.sqrt(score)),
                       selection_mse_tolerance=tolerance, candidates=audit)

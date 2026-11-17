"""Kernel differentiation, small ODE discovery, and analytic field prediction."""
from itertools import combinations
import numpy as np
from scipy.linalg import cho_factor, cho_solve


class ScalarKernel:
    def __init__(self, length, ridge):
        self.length, self.ridge = length, ridge

    def fit(self, s, b):
        self.s, self.mean = np.asarray(s).copy(), float(np.mean(b))
        matrix = self.kernel(s)
        matrix.flat[::len(s)+1] += self.ridge+1e-10
        self.weights = cho_solve(cho_factor(matrix, lower=True), b-self.mean)
        return self

    def kernel(self, s):
        return np.exp(-.5*((np.asarray(s)[:, None]-self.s)/self.length)**2)

    def predict(self, s):
        return self.mean+self.kernel(s)@self.weights

    def derivative(self, s):
        delta = np.asarray(s)[:, None]-self.s
        return (-delta/self.length**2*self.kernel(s))@self.weights


class AnalyticField:
    def __init__(self, kind, coefficients, reference_s, reference_b, field_mean, field_scale,
                 coordinate_scale=1., support=()):
        self.kind = kind
        self.coefficients = np.asarray(coefficients)
        self.reference_s, self.reference_b = reference_s, reference_b
        self.field_mean, self.field_scale = field_mean, field_scale
        self.coordinate_scale, self.support = coordinate_scale, tuple(support)

    def predict(self, s):
        s = np.asarray(s)
        if self.kind == "polynomial":
            return np.polynomial.polynomial.polyval(s-self.reference_s, self.coefficients)
        # z'=c0+c1*z+c2*z^2. Solve via a 2x2 linear system, z=u/v.
        c0, c1, c2 = self.coefficients
        z0 = (self.reference_b-self.field_mean)/self.field_scale
        t = s-self.reference_s
        discriminant = c1*c1-4*c0*c2
        if discriminant > 1e-12:
            rate = .5*np.sqrt(discriminant)
            with np.errstate(over="ignore", invalid="ignore"):
                h, g = np.cosh(rate*t), np.sinh(rate*t)/rate
        elif discriminant < -1e-12:
            rate = .5*np.sqrt(-discriminant)
            h, g = np.cos(rate*t), np.sin(rate*t)/rate
        else:
            h, g = np.ones_like(t), t
        numerator = h*z0+g*(.5*c1*z0+c0)
        denominator = h+g*(-c2*z0-.5*c1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = numerator/denominator
        z = np.where(abs(denominator)>1e-9, z, np.nan)
        return self.field_mean+self.field_scale*z

    def derivative(self, s, b=None):
        if self.kind == "polynomial":
            return np.polynomial.polynomial.polyval(np.asarray(s)-self.reference_s,
                np.polynomial.polynomial.polyder(self.coefficients))
        if b is None:
            b = self.predict(s)
        z = (np.asarray(b)-self.field_mean)/self.field_scale
        return self.field_scale*np.polynomial.polynomial.polyval(z, self.coefficients)

    def to_dict(self):
        result = {"kind": self.kind, "coefficients": self.coefficients.tolist(),
            "reference_s_metres": float(self.reference_s), "reference_b": float(self.reference_b),
            "field_mean": float(self.field_mean), "field_scale": float(self.field_scale),
            "coordinate_scale_metres": float(self.coordinate_scale),
            "selected_derivative_powers": list(self.support),
            "derivative_terms_plus_reference": len(self.support)+1}
        if self.kind == "riccati":
            c0, c1, c2 = self.coefficients
            delta = float(c1*c1-4*c0*c2)
            result.update({"discriminant": delta, "solution_family":
                "linear" if abs(c1)+abs(c2)<1e-12 else "exponential" if abs(c2)<1e-12
                else "hyperbolic-function ratio" if delta>1e-12
                else "trigonometric-function ratio" if delta < -1e-12 else "rational"})
            if abs(c2)>1e-12 and delta>1e-12:
                midpoint, amplitude = -c1/(2*c2), -np.sqrt(delta)/(2*c2)
                z0 = (self.reference_b-self.field_mean)/self.field_scale
                relative = (z0-midpoint)/amplitude
                if abs(relative)<1:
                    rate = .5*np.sqrt(delta)
                    result["tanh_equivalent"] = {
                        "background": float(self.field_mean+self.field_scale*midpoint),
                        "amplitude": float(self.field_scale*amplitude),
                        "width_metres": float(1/rate),
                        "center_metres": float(self.reference_s-np.arctanh(relative)/rate)}
        return result


def extract_candidates(kernel, s, observed_b, validation_s):
    """Fit derivative equations only to the kernel at actual training positions."""
    reference_s = float(np.median(s))
    reference_b = float(kernel.predict(np.array([reference_s]))[0])
    mean = float(np.mean(observed_b))
    scale = max(float(np.std(observed_b)), 1e-9)
    coordinate_scale = max(float(np.std(s)), 1e-9)
    b = kernel.predict(s)
    target = kernel.derivative(s)/scale
    zv = (kernel.predict(validation_s)-mean)/scale
    candidates = {}
    design = np.vander((b-mean)/scale, 3, increasing=True)
    for size in (1, 2, 3):
        for support in combinations(range(3), size):
            coefficients = np.zeros(3)
            coefficients[list(support)] = np.linalg.lstsq(design[:, support], target, rcond=None)[0]
            name = "riccati_"+"_".join(map(str, support))
            candidates[name] = AnalyticField("riccati", coefficients, reference_s, reference_b,
                                             mean, scale, support=support)
    # z'=sum c_k ((s-sr)/sx)^k, integrated analytically to a polynomial field.
    for derivative_degree in range(4):
        t = (s-reference_s)/coordinate_scale
        coefficients = np.linalg.lstsq(np.vander(t, derivative_degree+1, increasing=True),
                                       target, rcond=None)[0]
        powers = np.arange(1, derivative_degree+2)
        integrated = np.r_[reference_b, scale*coefficients/(powers*coordinate_scale**(powers-1))]
        candidates[f"coordinate_derivative_degree_{derivative_degree}"] = AnalyticField(
            "polynomial", integrated, reference_s, reference_b, mean, scale,
            coordinate_scale, tuple(range(derivative_degree+1)))
    diagnostics = []
    target_v = kernel.derivative(validation_s)
    for name, model in candidates.items():
        # Local derivative fit, evaluated at the teacher's B rather than rolled-out B.
        predicted_derivative = model.derivative(validation_s, mean+scale*zv)
        error = float(np.linalg.norm(predicted_derivative-target_v)/max(np.linalg.norm(target_v), 1e-12))
        diagnostics.append({"name": name, "derivative_terms_plus_reference": len(model.support)+1,
                            "validation_kernel_derivative_relative_residual": error})
    return candidates, diagnostics

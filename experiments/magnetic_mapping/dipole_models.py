"""Equivalent magnetic dipoles; moments absorb mu0/(4*pi) and field units."""
import numpy as np


def design(points, sources):
    points = np.asarray(points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    r = points[:, None, :] - np.asarray(sources)[None, :, :]
    radius = np.linalg.norm(r, axis=2)
    if np.any(radius == 0):
        raise ValueError("Cannot evaluate at a point source")
    tensor = (3*r[..., :, None]*r[..., None, :]/radius[..., None, None]**5
              - np.eye(3)/radius[..., None, None]**3)
    return tensor.transpose(0, 2, 1, 3).reshape(3*len(points), -1)


class DipoleField:
    def __init__(self, sources, background, moments):
        self.sources = np.asarray(sources)
        self.background = np.asarray(background)
        self.moments = np.asarray(moments)

    def predict(self, points):
        return (design(points, self.sources) @ self.moments.ravel()).reshape(-1, 3) + self.background

    def to_dict(self):
        return dict(sources_metres=self.sources.tolist(), background=self.background.tolist(),
                    effective_moments=self.moments.tolist())


def fit_path(x, targets, sources, ridges):
    """Unpenalized constant vector; ridge on column-scaled dipole coefficients."""
    d = design(x, sources).reshape(len(x), 3, -1)
    mean_design = d.mean(axis=0)
    a = (d-mean_design).reshape(3*len(x), -1)
    scales = np.sqrt(np.sum(a*a, axis=0))
    scales = np.maximum(scales, 1e-12)
    u, singular, vt = np.linalg.svd(a/scales, full_matrices=False)
    target_mean = targets.mean(axis=0)
    projected = u.T @ (targets-target_mean).ravel()
    for ridge in ridges:
        weights = (vt.T @ (singular/(singular**2+ridge)*projected))/scales
        yield ridge, DipoleField(sources, target_mean-mean_design@weights, weights.reshape(-1, 3))

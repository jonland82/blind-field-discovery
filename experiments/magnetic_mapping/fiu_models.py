"""Competing descriptions of real magnetic readings under optical pose changes."""
import numpy as np

from models import rbf, solve


def world_covariance(x, z, length):
    """Covariance of a 3D gradient of a scalar RBF potential."""
    delta = (x[:, None, :] - z[None, :, :]) / length
    blocks = (np.eye(3) - delta[..., :, None] * delta[..., None, :]) * rbf(x, z, length)[..., None, None]
    return blocks.transpose(0, 2, 1, 3).reshape(3 * len(x), 3 * len(z))


def orientation_matrix(rotations, convention):
    if convention == "as_recorded":
        return rotations
    if convention == "transposed":
        return rotations.transpose(0, 2, 1)
    raise ValueError("Unknown optical rotation convention")


class PoseMap:
    """Three independent kernel maps of position and optical orientation.

    The covariance does not impose a vector transformation on the outputs.
    """
    def __init__(self, length, orientation_length, ridge):
        self.length, self.orientation_length, self.ridge = length, orientation_length, ridge

    def features(self, x, rotations):
        return np.column_stack((x / self.length,
                                rotations.reshape(-1, 9) / (np.sqrt(2) * self.orientation_length)))

    def fit(self, x, rotations, y):
        self.training = self.features(x, rotations)
        self.mean = y.mean(axis=0)
        self.weights = solve(rbf(self.training, self.training, 1.), y - self.mean, self.ridge)
        return self

    def predict(self, x, rotations):
        features = self.features(x, rotations)
        return rbf(features, self.training, 1.) @ self.weights + self.mean


class WorldMap:
    """A common world field viewed through optical rotations, plus fixed bias.

    The vector transformation is an explicit hypothesis, not a discovered law.
    'potential' additionally imposes curl-free world-field covariance. Neither
    construction imposes magnetic divergence or the complete Maxwell system.
    """
    def __init__(self, kind, length, ridge, convention="as_recorded"):
        if kind not in ("vector", "potential"):
            raise ValueError("Unknown world-field description")
        self.kind, self.length, self.ridge, self.convention = kind, length, ridge, convention

    def fit(self, x, rotations, y):
        rotations = orientation_matrix(rotations, self.convention)
        world_to_sensor = rotations.transpose(0, 2, 1)
        design = np.concatenate((world_to_sensor, np.broadcast_to(np.eye(3), (len(x), 3, 3))), axis=2)
        coefficient, _, rank, _ = np.linalg.lstsq(design.reshape(-1, 6), y.reshape(-1), rcond=None)
        if rank < 6:
            raise ValueError("Optical orientations do not identify a background vector and fixed sensor bias")
        self.background, self.bias = coefficient[:3], coefficient[3:]
        self.x = np.asarray(x).copy()
        residual = np.einsum("nij,nj->ni", rotations, y - self.bias) - self.background
        if self.kind == "vector":
            self.weights = solve(rbf(x, x, self.length), residual, self.ridge)
        else:
            self.weights = solve(world_covariance(x, x, self.length), residual.reshape(-1), self.ridge)
        return self

    def world_predict(self, x):
        if self.kind == "vector":
            residual = rbf(x, self.x, self.length) @ self.weights
        else:
            residual = (world_covariance(x, self.x, self.length) @ self.weights).reshape(-1, 3)
        return self.background + residual

    def predict(self, x, rotations):
        rotations = orientation_matrix(rotations, self.convention)
        return np.einsum("nji,nj->ni", rotations, self.world_predict(x)) + self.bias

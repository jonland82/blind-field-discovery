"""Privileged simulator and oracle. Never imported by learner.py."""
import numpy as np


def wire_field(inputs, center=(0., 0.), coefficient=.2):
    """Infinite z-directed wire, metres/amperes -> microtesla."""
    x, y, current = np.asarray(inputs).T
    x, y = x-center[0], y-center[1]
    squared_radius = x*x+y*y
    if np.any(squared_radius == 0):
        raise ValueError("The ideal wire is singular on its axis")
    return coefficient*current[:, None]*np.column_stack((-y, x, np.zeros_like(x)))/squared_radius[:, None]


def wire_jacobian(inputs):
    x, y, current = np.asarray(inputs).T
    r2 = x*x+y*y
    result = np.zeros((len(x), 3, 3))  # point, output, input
    result[:, 0, 0] = .4*current*x*y/r2**2
    result[:, 0, 1] = .2*current*(y*y-x*x)/r2**2
    result[:, 1, 0] = .2*current*(y*y-x*x)/r2**2
    result[:, 1, 1] = -.4*current*x*y/r2**2
    result[:, :, 2] = wire_field(np.column_stack((x, y, np.ones_like(x))))
    return result


def sample_inputs(rng, n, radius=(.25, 2.), current=(.5, 4.), angle=(-np.pi, np.pi)):
    r = rng.uniform(*radius, n)
    theta = rng.uniform(*angle, n)
    amps = rng.uniform(*current, n)*rng.choice([-1., 1.], n)
    return np.column_stack((r*np.cos(theta), r*np.sin(theta), amps))


def polynomial_field(inputs):
    """Nonmagnetic control: no prescribed circular or inverse-distance law."""
    x, y, current = np.asarray(inputs).T
    return np.column_stack((.7+.4*x-.2*y*current, -.3+.5*y+.1*x*current,
                            .15*current+.2*x*y))

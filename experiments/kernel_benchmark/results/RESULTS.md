# Synthetic kernel-to-equation verification

These results verify a numerical workflow using known artificial functions. They are not physical measurements.

The oracle is B(x) = 1.7 tanh(x/0.9), J(x) = 0.8 sech²(x/0.9). Their exact relation is B′ = (85/36) J.

The learner receives only sample arrays, a Gaussian kernel family, and a generic candidate library. Oracle formulas and derivatives are used only for generation and final scoring.

| Metric | Noiseless | 0.1% scale noise |
| --- | ---: | ---: |
| Field relative RMS error | 0.001176% | 0.08535% |
| Derivative relative RMS error | 0.04387% | 1.207% |
| Coefficient relative error | 1.519e-06% | 0.02208% |
| Integrated equation field relative RMS error | 1.258e-05% | 0.2289% |
| Extrapolated field relative RMS error (x = 3.05–4) | 19.96% | 23.98% |
| Extrapolated derivative relative RMS error (x = 3.05–4) | 2.79e+04% | 2.44e+04% |

**Noiseless recovered equation:** B′ = 2.36111108 (J).
Selected Gaussian length 0.5, ridge 1e-12.

**Noisy recovered equation:** B′ = 2.36163252 (J).
Selected Gaussian length 0.8, ridge 1e-06.

## Symbolic recovery of the original function family

A second search excludes J and x, using the autonomous field-only library {1, B, B², B³, B⁴}, with the same support-size rule and 1% residual tolerance. This supplies a search class, not the equation coefficients. No tanh or sech functions occur in the regression library.

Both runs select B′ = p − q B². Solving this equation on its bounded branch gives B(x) = A tanh((x − x_c)/L), with A = sqrt(p/q) and L = 1/sqrt(pq). One interior kernel value at the median training position fixes x_c. Together with B′ = a J, this gives J(x) = (p/a) sech²((x − x_c)/L).

The ODE solution rule is supplied analytic mathematics. The polynomial support, coefficients, scales, center and compatible source amplitude come from the samples. Branch and center selection finish before oracle scoring.

| Quantity | Noiseless | Noisy | Hidden truth |
| --- | ---: | ---: | ---: |
| p | 1.88888871 | 1.8901351 | 1.88888889 |
| q | 0.653594619 | 0.654414874 | 0.653594771 |
| Field amplitude A | 1.70000012 | 1.69949481 | 1.7 |
| Transition width L | 0.900000148 | 0.899139333 | 0.9 |
| Center x_c | 4.65379271e-08 | -0.00064392317 | 0 |
| Source amplitude | 0.799999935 | 0.800351063 | 0.8 |

| Symbolic prediction relative RMS error | Noiseless | Noisy |
| --- | ---: | ---: |
| Field | 5.674e-06% | 0.04462% |
| Source | 1.149e-05% | 0.08857% |
| Derivative | 1.157e-05% | 0.09033% |
| Extrapolated field (x = 3.05–4) | 6.727e-06% | 0.02891% |
| Extrapolated derivative (x = 3.05–4) | 0.00012% | 0.7672% |

Errors compare against the hidden analytic oracle on 901 unseen positions in [-3, 3]. Noise is Gaussian with standard deviations 0.001 × 1.7 for B and 0.001 × 0.8 for J; it is not relative to each local value.

The integrated prediction uses only the selected equation, the learned J function and one observed boundary value B(-3). Neither the true B curve nor its derivative is supplied to that solver.

Agreement is numerical on the tested interval, not exact equality of functions or evidence that physical laws were independently discovered. Extrapolation is reported separately. Only one seed and one generator family were tested.

[Explanation and formulas](../../README.md) · [Machine-readable report](results.json) · [Comparison plot](comparison.png)

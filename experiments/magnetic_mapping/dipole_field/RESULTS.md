# Real measurements reconstructed with magnetic dipole fields

See [the derivation and protocol](THEORY.md). All errors are vector RMSE in reference recording units.

Selected by trial 3: 100 equivalent dipoles, depth 1.0 m, ridge 0.001. Teacher validation error 8.6255; dipoles 6.1767.

| Environment | Instrument | Constant | Kernel | Kernel to dipoles | Direct-data dipoles |
| --- | --- | ---: | ---: | ---: | ---: |
| stable | invensense | 16.4819 | 8.0898 | 5.2248 | 5.3123 |
| stable | nexus | 16.3157 | 10.3424 | 8.8300 | 8.8467 |
| stable | trivisio | 16.4199 | 8.0266 | 5.5559 | 5.7067 |
| changed | invensense | 17.6227 | 8.6403 | 7.3288 | 7.3561 |
| changed | nexus | 27.5251 | 24.1574 | 23.8064 | 23.7877 |
| changed | trivisio | 17.2404 | 8.6873 | 7.4347 | 7.4962 |

[Shape comparison](comparison.png). [Full coefficients, component correlations, per-trial and spatial-hole errors](results.json).

Source-free Maxwell structure is supplied by the dipole basis. Measurement agreement tests the reconstruction, not independent discovery of Maxwell or identification of the actual sources.

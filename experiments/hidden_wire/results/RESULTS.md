# Hidden-wire experiment: results

Synthetic workflow verification. Read the [fixed protocol](../PROTOCOL.md) and [theory and comparison](../README.md).

Errors below are percentages of clean vector-field RMS, averaged over all three fixed seeds. Each fit was selected on noisy validation observations before test scoring.

| Case | Kernel: same domain | Kernel to expression | Direct expression | Distilled: full angles | Distilled: outward | Distilled: new currents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noiseless | 20.4176% | 0.3529% | 0.0000% | 0.3540% | 0.3513% | 0.3509% |
| main | 20.4393% | 0.3827% | 0.0557% | 0.3840% | 0.3795% | 0.3798% |
| noisy | 21.1762% | 0.6030% | 0.2783% | 0.6009% | 0.5977% | 0.5928% |
| sparse | 51.1045% | 6.2307% | 0.1047% | 6.1793% | 6.3795% | 6.3276% |
| limited_angle | 3.6235% | 0.1577% | 0.0385% | 0.9609% | 0.9056% | 0.9152% |

## Every recovered expression

x,y in metres, I in amperes, B in microtesla. These are displayed to nine significant figures; scores use full saved precision. No rounding to the generating law enters predictions.

### noiseless, seed 20261116

Selected rational_degree_2_threshold_0.003; 3 free nonzero coefficients.

```text
Bx = (-0.199138528*y*I) / (+1*y^2 +1.00333367*x^2)
By = (+0.199536025*x*I) / (+1*y^2 +1.00333367*x^2)
Bz = (0) / (+1*y^2 +1.00333367*x^2)
```

### noiseless, seed 20261117

Selected rational_degree_2_threshold_0.003; 3 free nonzero coefficients.

```text
Bx = (-0.199986668*y*I) / (+1*y^2 +0.999904986*x^2)
By = (+0.199978513*x*I) / (+1*y^2 +0.999904986*x^2)
Bz = (0) / (+1*y^2 +0.999904986*x^2)
```

### noiseless, seed 20261118

Selected rational_degree_2_threshold_0.003; 3 free nonzero coefficients.

```text
Bx = (-0.198745762*y*I) / (+1*y^2 +0.999411133*x^2)
By = (+0.19898907*x*I) / (+1*y^2 +0.999411133*x^2)
Bz = (0) / (+1*y^2 +0.999411133*x^2)
```

### main, seed 20261116

Selected rational_degree_2_threshold_0.003; 3 free nonzero coefficients.

```text
Bx = (-0.198992252*y*I) / (+1*y^2 +1.00305995*x^2)
By = (+0.199601092*x*I) / (+1*y^2 +1.00305995*x^2)
Bz = (0) / (+1*y^2 +1.00305995*x^2)
```

### main, seed 20261117

Selected rational_degree_2_threshold_0.003; 3 free nonzero coefficients.

```text
Bx = (-0.199873893*y*I) / (+1*y^2 +0.999647415*x^2)
By = (+0.20005534*x*I) / (+1*y^2 +0.999647415*x^2)
Bz = (0) / (+1*y^2 +0.999647415*x^2)
```

### main, seed 20261118

Selected rational_degree_2_threshold_0.01; 3 free nonzero coefficients.

```text
Bx = (-0.199033194*y*I) / (+1.00171191*y^2 +1*x^2)
By = (+0.199048321*x*I) / (+1.00171191*y^2 +1*x^2)
Bz = (0) / (+1.00171191*y^2 +1*x^2)
```

### noisy, seed 20261116

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.198406346*y*I) / (+1*y^2 +1.00194597*x^2)
By = (+0.199858191*x*I) / (+1*y^2 +1.00194597*x^2)
Bz = (0) / (+1*y^2 +1.00194597*x^2)
```

### noisy, seed 20261117

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.19834921*y*I) / (+1*y^2 +0.991124376*x^2)
By = (+0.198618948*x*I) / (+1*y^2 +0.991124376*x^2)
Bz = (0) / (+1*y^2 +0.991124376*x^2)
```

### noisy, seed 20261118

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.199718176*y*I) / (+1.00622886*y^2 +1*x^2)
By = (+0.19881673*x*I) / (+1.00622886*y^2 +1*x^2)
Bz = (0) / (+1.00622886*y^2 +1*x^2)
```

### sparse, seed 20261116

Selected rational_degree_2_threshold_0.01; 3 free nonzero coefficients.

```text
Bx = (-0.20141811*y*I) / (+1*y^2 +1.09676009*x^2)
By = (+0.209623774*x*I) / (+1*y^2 +1.09676009*x^2)
Bz = (0) / (+1*y^2 +1.09676009*x^2)
```

### sparse, seed 20261117

Selected rational_degree_2_threshold_0.1; 3 free nonzero coefficients.

```text
Bx = (-0.183808579*y*I) / (+1*y^2 +0.859247014*x^2)
By = (+0.167475835*x*I) / (+1*y^2 +0.859247014*x^2)
Bz = (0) / (+1*y^2 +0.859247014*x^2)
```

### sparse, seed 20261118

Selected rational_degree_2_threshold_0.1; 3 free nonzero coefficients.

```text
Bx = (-0.177745541*y*I) / (+0.960109459*y^2 +1*x^2)
By = (+0.177287991*x*I) / (+0.960109459*y^2 +1*x^2)
Bz = (0) / (+0.960109459*y^2 +1*x^2)
```

### limited_angle, seed 20261116

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.204258038*y*I) / (+1*y^2 +1.02915708*x^2)
By = (+0.20538486*x*I) / (+1*y^2 +1.02915708*x^2)
Bz = (0) / (+1*y^2 +1.02915708*x^2)
```

### limited_angle, seed 20261117

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.19937944*y*I) / (+0.993235286*y^2 +1*x^2)
By = (+0.199856746*x*I) / (+0.993235286*y^2 +1*x^2)
Bz = (0) / (+0.993235286*y^2 +1*x^2)
```

### limited_angle, seed 20261118

Selected rational_degree_2_threshold_0.03; 3 free nonzero coefficients.

```text
Bx = (-0.199167358*y*I) / (+0.975420921*y^2 +1*x^2)
By = (+0.199669943*x*I) / (+0.975420921*y^2 +1*x^2)
Bz = (0) / (+0.975420921*y^2 +1*x^2)
```

## Matched workflow audit

Synthetic entries use the fixed first main seed, not a selected best seed. Aalto entries reproduce the frozen reference-instrument results. Absolute errors have different units and noise levels; this table compares stages, not difficulty.

| Stage | Hidden wire | Measured Aalto sensor workflow |
| --- | --- | --- |
| Training / validation / stable test observations | 480 / 180 / 800 | 500 / 600 / 3000 |
| Kernel inputs / outputs | x,y,I / Bx,By,Bz | x,y / Bx,By,Bz; upstream orientation correction |
| Explicit representation | General polynomial and rational search | Supplied 3D magnetic dipole response, assumed source sheet |
| Expression-fit targets | Teacher at training positions | Teacher at training positions |
| Kernel training measurement RMSE | 0.022420 microtesla | 2.191329 reference units |
| Kernel validation measurement RMSE | 0.175261 microtesla | 8.625504 reference units |
| Extracted field validation measurement RMSE | 0.008664 microtesla | 6.176653 reference units |
| Direct field validation measurement RMSE | 0.007888 microtesla | 6.071583 reference units |
| Expression vs teacher at training inputs | 3.033% | 5.731% |
| Expression vs teacher at stable test inputs | 20.515% | 12.197% |
| Kernel: stable test normalized error | 19.817% of clean field RMS | 16.677% of observed field RMS |
| Extracted field: stable test normalized error | 0.516% of clean field RMS | 10.771% of observed field RMS |
| Direct fit: stable test normalized error | 0.061% of clean field RMS | 10.951% of observed field RMS |
| Analytic truth available? | Yes, used only in generation and final scoring | No room-field formula or true source locations |
| Changed environment | Wire moves; held-out geometry excluded from inputs | Recorded toolbox/shelf changes; source geometry unavailable |
| Independent physical-structure discovery? | Sparse algebra selected within a supplied grammar | Magnetic structure supplied by dipoles |

The teacher-to-expression discrepancy need not shrink on new positions: the expression can disagree with the teacher and agree better with observations. The reconstruction is therefore an additional modeling assumption, not an exact symbolic rewrite of the kernel.

## Fixed diagnostics for the first main seed

Post-fit model diagnostics, not independently observed derivatives.

| Model | Spatial derivative error | Planar divergence / derivative norm | Planar curl / derivative norm |
| --- | ---: | ---: | ---: |
| kernel | 0.635903 | 0.210727 | 0.524101 |
| distilled | 0.00528349 | 1.76101e-07 | 0.00142354 |
| direct | 0.000626789 | 0.000697143 | 0.000513258 |

## Changed environment

The wire moves without updating the inputs or retraining. Each entry averages all main-case seeds.

| Model | Error against moved wire |
| --- | ---: |
| kernel | 86.991% |
| distilled | 89.869% |
| direct | 89.999% |

## Nonmagnetic control

Distilled selection: rational_degree_2_threshold_0.003, with constant denominator (an ordinary polynomial). Outward-radius plus unseen-current relative error: 4.85627e-06%. See the JSON for the full expressions and direct-fit control.

[Plots](workflow.png), [all fits and scores](results.json), [sample and prediction arrays](predictions.npz).

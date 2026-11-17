# A compact magnetic-field function extracted from real-data learning

**Question:** can the kernel map learned from public magnetic measurements be expressed as a smaller, explicit function without losing its predictive usefulness?

This is an empirical field map for one room and measurement frame. No Maxwell constraints, dipole formula, current labels, or synthetic physical fields enter fitting.

## Selected function

The selected function is **affine_plus_64_gaussians_length_0.35**, with **67 basis terms per channel**, compared with 501 for the original kernel map (including its mean).
It did not meet the predeclared source-validation requirement of at most 5% higher RMSE than the teacher.

Coordinates $x,y$ are supplied positions in metres. Define normalized coordinates

$$
u=(x-(1.5304872434))/(1.62340676617),\qquad v=(y-(-0.9347748064))/(1.32748404387).
$$

The three measured field channels are represented by

$$
\mathbf B_{\mathrm{compact}}(x,y)=\sum_{p,q}\mathbf c_{pq}u^pv^q+\sum_{m=1}^{M}\mathbf w_m\exp\!\left[-\frac{(x-a_m)^2+(y-b_m)^2}{2\ell^2}\right].
$$

Coefficients below are in the reference recording's numerical field units; no unverified SI conversion is made. All three channels share the displayed spatial basis.

| Polynomial powers (p,q) | Channel 1 | Channel 2 | Channel 3 |
| --- | ---: | ---: | ---: |
| (0, 0) | -23.6302678946 | -8.28056700494 | -43.9169789085 |
| (0, 1) | 2.33096475784 | -4.25568588333 | 1.71327683349 |
| (1, 0) | 3.61813517564 | 6.72630515959 | -0.246250465694 |

Here M = 64 and $\ell=0.35$ m.

| m | Center x (m) | Center y (m) | Weight channel 1 | Weight channel 2 | Weight channel 3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.36749 | 0.12165 | -6.91003710068 | -3.10972868774 | 32.0035081762 |
| 2 | -0.053657 | 0.77381 | 12.6510986361 | 6.16907909574 | 25.0410588606 |
| 3 | 0.20251 | -0.059975 | 4.59931591777 | -4.94434597739 | -31.214924907 |
| 4 | -0.64217 | -2.3915 | 22.5459028705 | 15.3720080558 | -5.39424442662 |
| 5 | -0.98287 | -0.45036 | 44.9878588241 | 5.44576288835 | -23.4829724657 |
| 6 | -0.55134 | -1.6054 | 14.2370468615 | -3.8783591679 | 44.1111784255 |
| 7 | -0.89644 | 0.6832 | 9.13231704344 | 12.3744855027 | -52.9195866105 |
| 8 | 3.5031 | -1.6728 | -5.30438732031 | -21.0643005034 | -33.8712191881 |
| 9 | -0.59491 | -0.98104 | -11.6626506327 | 17.8350017202 | 25.3556674469 |
| 10 | 0.16942 | -2.1126 | -8.57670044198 | 19.0713885662 | -10.7181401542 |
| 11 | 0.37148 | 1.0988 | -4.42253023595 | 29.6781946189 | -30.495452827 |
| 12 | 3.5734 | -2.8019 | 6.67488101826 | -8.4235807398 | 21.3542025799 |
| 13 | -0.094782 | -2.7263 | -15.3408398576 | 22.8066275101 | 18.6588273857 |
| 14 | -0.33858 | 1.0827 | 15.2056634744 | 64.333193441 | -25.6304031575 |
| 15 | 2.3812 | 0.11389 | -11.1873906929 | 4.10563526914 | -21.8969512972 |
| 16 | 2.6745 | -1.8751 | -9.74866709716 | 5.87570704849 | 13.7917958862 |
| 17 | 0.87536 | -1.5353 | 0.884660608737 | 23.7599975709 | 22.4331140627 |
| 18 | 1.6862 | -0.37435 | 11.2204882897 | 10.53941177 | 7.00809378058 |
| 19 | 3.4732 | -2.1496 | 10.4567393223 | 24.7963233805 | -6.91168909043 |
| 20 | 0.79741 | 0.77461 | -3.09162676095 | 18.8926228405 | 7.22420682026 |
| 21 | -0.048365 | -0.82635 | 3.82845338281 | -4.56570259924 | 8.66675394747 |
| 22 | 2.9612 | -0.46048 | 17.367323773 | 4.15237357183 | 18.6461244572 |
| 23 | 2.997 | -2.7074 | -11.2079615382 | -4.96136432903 | 6.10160435547 |
| 24 | -0.84455 | -1.8283 | -29.7959553702 | -11.3873758031 | -34.9742758599 |
| 25 | 3.5531 | 1.024 | 4.97263793094 | 5.48676662673 | 9.48734369693 |
| 26 | 1.6501 | 0.56366 | 6.84032619076 | 1.57601829199 | 9.60276005377 |
| 27 | 0.70009 | -2.3456 | 3.39627895841 | 8.49558190814 | 1.59321646672 |
| 28 | 0.61482 | -0.5541 | 15.3500283427 | 15.7124962068 | 10.7194598783 |
| 29 | 1.7565 | -2.4863 | 7.81731010605 | 3.7460443755 | 7.16302815029 |
| 30 | 2.7143 | -1.2722 | 0.281751799383 | 6.39928961823 | -9.52714079217 |
| 31 | 2.8435 | 1.0658 | 1.53352680285 | 12.2695099853 | -2.66350259036 |
| 32 | 2.9491 | 0.27904 | -4.81686081124 | -0.16653195298 | 7.37931129927 |
| 33 | 3.5646 | -0.11544 | 8.46311605562 | -1.19104756761 | -9.047128152 |
| 34 | 1.7837 | 1.1227 | 7.2768087474 | 11.8039496374 | -9.56113545626 |
| 35 | 1.2299 | -1.5167 | 24.6123074984 | -16.0536764046 | -2.00927028318 |
| 36 | 0.48709 | -2.8584 | 11.5119722448 | -2.81534385985 | 6.54706287991 |
| 37 | 2.4203 | -2.6198 | 7.96509251615 | -2.20734034743 | -4.68940926872 |
| 38 | -0.99844 | 0.26045 | -8.0571930143 | 24.8448374481 | 4.2007363835 |
| 39 | 2.109 | -0.84942 | 13.2459253141 | 18.2477434077 | -23.4159198695 |
| 40 | -0.66023 | -0.23573 | -56.7668477479 | -20.1208749976 | -12.3933098982 |
| 41 | 4.133 | 0.1252 | -8.06102450968 | 9.41997483238 | 1.00182943559 |
| 42 | 1.6669 | -3.1417 | 5.79608481357 | -6.06084045281 | 4.53339277872 |
| 43 | 3.5453 | 0.49043 | 2.01380059018 | -6.55531602706 | -4.57699638452 |
| 44 | 3.024 | -0.88707 | -12.34506396 | -5.04729209759 | -20.3951493082 |
| 45 | 3.1012 | -1.3954 | 24.0234812504 | 7.57851896179 | 18.1126062156 |
| 46 | -0.19796 | -0.26535 | 32.2320780687 | 15.8885784506 | 33.385888079 |
| 47 | 0.62775 | 0.40368 | 13.5282381339 | 2.35620942533 | 3.37729680615 |
| 48 | 1.1273 | 0.094381 | -5.55569763072 | 3.47163911502 | -1.85249542771 |
| 49 | 1.2451 | -0.87573 | -3.23471078765 | 9.89882146129 | -16.9863903513 |
| 50 | -0.21945 | -2.2247 | -6.52279622123 | -43.0809531848 | -13.9180577104 |
| 51 | -0.074695 | -1.5822 | 11.4127217639 | -1.16228152111 | -11.6316960155 |
| 52 | 4.1394 | 0.73447 | 3.69299794687 | -2.3485280394 | -7.11448635059 |
| 53 | 1.9531 | 0.063807 | 14.5792054819 | 3.01021775441 | -2.8227951089 |
| 54 | 2.9269 | -3.2881 | -2.23034292037 | -2.87093839136 | 4.77492963885 |
| 55 | -0.66361 | 1.0333 | -27.2599194232 | -9.52468130233 | 25.9357678822 |
| 56 | 0.58179 | -1.8801 | -6.45684363495 | -19.4189838674 | 10.4312497726 |
| 57 | 2.5019 | -0.61668 | -7.06638478089 | -15.3455082986 | 7.77700343063 |
| 58 | 3.6069 | -0.85412 | -2.14014190338 | -5.59589726349 | 0.401840125754 |
| 59 | -0.35045 | 0.4837 | -17.7517573328 | 13.5652891152 | 33.3322686943 |
| 60 | -0.42216 | -2.6015 | 1.95480240593 | 19.3925087072 | -32.2538790741 |
| 61 | 0.21426 | 0.42245 | 10.7421914023 | -2.90452431327 | 24.4249633671 |
| 62 | -0.90138 | -0.7362 | -33.707213205 | 17.9265070644 | -20.0011841602 |
| 63 | 1.4382 | 1.12 | 3.99405450312 | 4.13983081021 | 4.75353285751 |
| 64 | 1.6908 | -0.73583 | 2.77836269181 | -17.1018902696 | 19.1891936325 |

## Predictions against measured recordings

Vector RMSE pools all three components and both stable test trials. The teacher and compact map predict exactly the same observations. Other instruments use the earlier trial-1-only frozen affine alignment.

| Instrument | Constant | Quadratic from teacher | Original kernel | Compact distilled | Matched direct-data fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| invensense | 16.4819 | 16.0119 | 8.0898 | 8.7254 | 8.7588 |
| nexus | 16.3157 | 15.4760 | 10.3424 | 10.8704 | 10.9734 |
| trivisio | 16.4199 | 16.0263 | 8.0266 | 8.4581 | 8.4907 |

## Frozen-map changed-environment stress test

| Instrument | Constant | Original kernel | Compact distilled |
| --- | ---: | ---: | ---: |
| invensense | 17.6227 | 8.6403 | 9.1056 |
| nexus | 27.5251 | 24.1574 | 24.6215 |
| trivisio | 17.2404 | 8.6873 | 9.0616 |

## What can be differentiated

For a Gaussian term $g_m$, its spatial derivatives are analytic:

$$
\partial_x g_m=-\frac{x-a_m}{\ell^2}g_m,\qquad\partial_y g_m=-\frac{y-b_m}{\ell^2}g_m.
$$

Polynomial terms differentiate in the usual way, with the coordinate-scale factors above.
The compact map's gradient differs from the teacher's by 37.7% in relative Frobenius norm at validation positions. This is model-to-model disagreement, not error against a measured true derivative.

A short function can preserve values while changing derivatives. These recordings supply no independent current density, electric field, or normal spatial derivative. We cannot identify $B'=aJ$, full curl, divergence, or Maxwell merely by differentiating this map.

## Protocol and limits

- Public Aalto data, pinned and hash-verified; train Invensense trials 1–2, select the teacher and student on source trial 3, evaluate trials 4–5 and stress trials 6–9.
- The earlier spatial rectangle and buffer remain excluded from training and validation. Per-trial and spatial-hole results are preserved in JSON.
- Student fitting uses teacher predictions only at actual training positions. Those targets are derived model outputs, not additional independent observations. Validation and evaluation use actual measured labels.
- Candidate polynomials have degree 0–5; sparse affine-plus-Gaussian maps use widths 0.2, 0.35, 0.7, 1.4 m and 1, 2, 4, 8, 16, 32, 64 shared centers. Geometry and coefficient fitting use training only. The matched direct-data control reuses the selected shape and fits coefficients to measured training labels.
- Pick the fewest basis terms within 5% of the original teacher's source-validation RMSE, breaking ties by error. If no candidate qualifies, disclose failure and report the minimum-error fallback. The tolerance is a modeling choice, not a confidence interval.
- These test recordings have already been examined in prior experiments. This is exploratory reuse, not a fresh confirmatory test. No test scores tune this run's candidates or settings.
- Known coordinates, upstream calibration/orientation correction, the basis family and static-map assumption are supplied. A compact empirical field function does not establish a unique source arrangement or a universal law.

## How much further to a theory?

This step asks whether a measured-data predictor admits a smaller explicit representation. Next, quantify coefficient and derivative stability across training subsets, then replicate the frozen extraction procedure on an independent public magnetic dataset. A source-response equation requires independently measured sources and suitable geometry; this dataset cannot supply them. A new dataset is needed to establish that stronger claim.

[Comparison plot](comparison.png) · [Machine-readable results and full coefficients](results.json).

Reproduce from the project root: `python experiments/magnetic_mapping/run_compact_field.py`.

Source: Solin et al. (2018), [Aalto magnetic recordings](https://github.com/AaltoML/magnetic-data), [paper](https://doi.org/10.1109/TRO.2018.2830326). The upstream data are already calibrated and orientation-corrected. Attribution and original provenance are retained; no new redistribution licence is asserted.

# From real magnetic measurements to explicit field functions

**This experiment runs the kernel → derivative equation → analytic solution workflow on public measured magnetic recordings.** It is a simplified one-dimensional field-strength profile, not a recovery of Maxwell or source/current equations.

## Observed result

The original 1D kernel predicts held-out source-instrument readings with RMSE **2.879**, versus **8.100** for a constant prediction. No single global Riccati equation captures the multi-peaked profile: its analytic prediction has RMSE **16.992**.

The four fixed local intervals give pooled RMSE **2.936** for their analytic candidates, **2.636** for their kernels, and **7.664** for local constants. **0 of 5 region extractions pass all validation requirements.** Good value prediction alone has not established a reliable derivative equation. The candidates remain useful explicit approximations, with acceptance failures retained.

The 1D simplification also loses some cross-instrument predictive performance: the phone's whole-band kernel does not beat its constant baseline. The full tables retain these failures and changed-environment results.

## Starting question and data

Can a room's measured magnetic field, first learned as a kernel, be expressed by a short differential equation whose solved function predicts other recordings?

Use the public Aalto magnetometer data. The scalar B is the norm of all three channels, s is the published x position, and observations are restricted to y = -0.892475 ± 0.15 m. This band is fixed by training geometry alone. Its finite width means B(s) is an approximation to a spatial profile, not a measurement along an exact mathematical line.

Source trials 1–2 train, trial 3 selects, trials 4–5 evaluate, and trials 6–9 stress the frozen functions after environmental changes. Four equal-width local intervals are fixed from the training band's x extent; every interval and the whole band are reported. No magnetic values select a favorable location. Other sensors use the earlier trial-1-only frozen affine calibration.

**Exploratory scope:** these acquisition groups have previously been scored in other project experiments. This run freezes its choices before opening their labels, but they are not a new confirmatory test. This separate 1D protocol does not retain the old 2D spatial hole.

## Formalism

$$
\widehat B(s)=\overline B+\sum_i\alpha_i e^{-(s-s_i)^2/(2\ell^2)},\qquad\widehat B'(s)=-\sum_i\alpha_i\frac{s-s_i}{\ell^2}e^{-(s-s_i)^2/(2\ell^2)}.
$$

Normalize the fitted field as z=(B−m)/σ, where m and σ come from training measurements. Fit all supports of the candidate equation

$$
z'=c_0+c_1z+c_2z^2.
$$

Each candidate is solved analytically using z=u/v and

$$
\frac{d}{ds}\begin{pmatrix}u\\v\end{pmatrix}=\begin{pmatrix}c_1&c_0\\-c_2&0\end{pmatrix}\begin{pmatrix}u\\v\end{pmatrix}.
$$

Its matrix exponential yields an explicit ratio of hyperbolic functions, trigonometric functions, or their linear/rational limits. This includes the toy example's tanh branch without fitting tanh features. An interior kernel value at the median training s fixes the integration constant.

Explicitly, set $t=s-s_r$, $z_r=(B_r-m)/\sigma$, $\Delta=c_1^2-4c_0c_2$, and let $h=\cosh(\sqrt{\Delta}t/2)$, $g=2\sinh(\sqrt{\Delta}t/2)/\sqrt{\Delta}$ for $\Delta>0$. Then

$$
B(s)=m+\sigma\frac{h z_r+g(c_1z_r/2+c_0)}{h+g(-c_2z_r-c_1/2)}.
$$

For negative discriminant use cosine and sine with $\sqrt{-\Delta}$; at zero discriminant use $h=1$, $g=t$. Poles/nonfinite predictions remain failures rather than being silently dropped.

Also search coordinate-dependent derivative polynomials of degree 0–3. Integrating them gives a field polynomial of degree 1–4. Their derivative coefficients are fitted to the kernel derivative, not directly to the measured field values; the integration reference comes from the kernel.

Validation requires derivative residual ≤10%, measured prediction RMSE ≤110% of the kernel RMSE, and ≥5% improvement over the training-mean baseline. Select the fewest derivative terms plus reference among passing candidates, breaking ties by measured validation error. Failure is reported if no candidate passes; a best-value analytic fallback remains visible but is not accepted as a recovered equation. The original toy's stricter 1% derivative diagnostic is also preserved.

## Region-by-region extraction

| Region | x interval (m) | Train / validate | Status | Function | Kernel validation RMSE | Analytic validation RMSE | Derivative residual |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| whole_band | -0.892 to 4.255 | 400 / 361 | no_analytic_candidate_passed | riccati_0_1_2 | 1.8543 | 11.9445 | 130.5% |
| interval_1 | -0.892 to 0.395 | 364 / 91 | no_analytic_candidate_passed | coordinate_derivative_degree_2 | 2.1838 | 12.1706 | 68.4% |
| interval_2 | 0.395 to 1.681 | 283 / 92 | no_analytic_candidate_passed | coordinate_derivative_degree_3 | 0.9089 | 2.5130 | 23.7% |
| interval_3 | 1.681 to 2.968 | 400 / 94 | no_analytic_candidate_passed | riccati_0_1_2 | 1.0116 | 1.7786 | 64.4% |
| interval_4 | 2.968 to 4.255 | 282 / 84 | no_analytic_candidate_passed | riccati_2 | 1.3784 | 1.3855 | 125.2% |

## Scores against actual held-out measurements

RMSE is for scalar field magnitude in reference numerical recording units, not an asserted SI unit. Local pooled scores combine all four intervals, including unsuccessful extractions. The local constant baseline has a separately fitted mean for each interval. Analytical fallback scores must be read together with their region's acceptance status.

| Scope | Instrument | Environment | N | Constant | Kernel | Selected analytic | Best Riccati diagnostic |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| whole_band | invensense | stable | 1088 | 8.1000 | 2.8790 | 16.9925 | 16.9925 |
| whole_band | invensense | changed | 2839 | 6.4455 | 3.5529 | 11.6737 | 11.6737 |
| whole_band | nexus | stable | 1154 | 7.8568 | 7.9409 | 13.7318 | 13.7318 |
| whole_band | nexus | changed | 2824 | 18.0710 | 19.2872 | 14.2002 | 14.2002 |
| whole_band | trivisio | stable | 1897 | 7.2626 | 3.4370 | 17.0264 | 17.0264 |
| whole_band | trivisio | changed | 3572 | 6.1711 | 2.6933 | 12.0428 | 12.0428 |
| local_intervals_pooled | invensense | stable | 1088 | 7.6637 | 2.6363 | 2.9362 | 7.1576 |
| local_intervals_pooled | invensense | changed | 3357 | 6.1803 | 3.6584 | 3.3826 | 5.1796 |
| local_intervals_pooled | nexus | stable | 1154 | 6.2668 | 7.5037 | 6.3251 | 7.1623 |
| local_intervals_pooled | nexus | changed | 3134 | 18.0470 | 20.3918 | 20.1050 | 18.3516 |
| local_intervals_pooled | trivisio | stable | 2442 | 7.8179 | 3.4149 | 3.5184 | 7.1542 |
| local_intervals_pooled | trivisio | changed | 5470 | 5.4335 | 2.9630 | 2.8460 | 5.1109 |

## The actual selected functions

### whole_band

Status: **no_analytic_candidate_passed**. Reference position $s_r=1.69295$ m.

$$
z'=(0.587381620904)+(-1.31477783929)z+(-0.494450203532)z^2,\qquadz=(B-(48.5293654433))/(10.3660624134).
$$

Solution family: hyperbolic-function ratio; at s = 1.69295, B = 40.8826934779.

Equivalently, the solved candidate is

$$
B_{\mathrm{analytic}}(s)=(34.7473212206)+(17.8212260755)\tanh\!\left[\frac{s-(1.2707032498)}{1.17639638098}\right].
$$

This is an analytic re-expression of the fitted equation, not a tanh fit directly to measured labels. Its validation status above still applies.

### interval_1

Status: **no_analytic_candidate_passed**. Reference position $s_r=-0.29712$ m.

$$
B_{\mathrm{analytic}}(s)=(40.965747431)(s-(-0.29712))^0+(-48.1779500539)(s-(-0.29712))^1+(50.946178891)(s-(-0.29712))^2+(50.1527049482)(s-(-0.29712))^3.
$$

Its derivative is the fitted coordinate-polynomial equation; its closed-form field is obtained by exact integration. This is a local spatial approximation, not a source law.

### interval_2

Status: **no_analytic_candidate_passed**. Reference position $s_r=0.76556$ m.

$$
B_{\mathrm{analytic}}(s)=(40.6680301244)(s-(0.76556))^0+(45.5145616364)(s-(0.76556))^1+(23.2374784648)(s-(0.76556))^2+(-234.690056803)(s-(0.76556))^3+(161.375147234)(s-(0.76556))^4.
$$

Its derivative is the fitted coordinate-polynomial equation; its closed-form field is obtained by exact integration. This is a local spatial approximation, not a source law.

### interval_3

Status: **no_analytic_candidate_passed**. Reference position $s_r=2.07285$ m.

$$
z'=(4.23945891062)+(-1.78944652548)z+(-1.20469543767)z^2,\qquadz=(B-(51.1891737592))/(6.80335472637).
$$

Solution family: hyperbolic-function ratio; at s = 2.07285, B = 52.828209918.

Equivalently, the solved candidate is

$$
B_{\mathrm{analytic}}(s)=(46.1363450126)+(13.7264504585)\tanh\!\left[\frac{s-(1.85364564222)}{0.411422087171}\right].
$$

This is an analytic re-expression of the fitted equation, not a tanh fit directly to measured labels. Its validation status above still applies.

### interval_4

Status: **no_analytic_candidate_passed**. Reference position $s_r=3.55255$ m.

$$
z'=(0)+(0)z+(-1.38445823594)z^2,\qquadz=(B-(52.1512798743))/(2.67176862842).
$$

Solution family: rational; at s = 3.55255, B = 53.2190207054.

## What this establishes and the remaining gap

An accepted function is a compact empirical description within its band/interval under fixed validation rules. Agreement of its solved function with measured values supports that scope. It is not unique: other basis families or source arrangements may fit the same finite observations.

A successful local polynomial does not demonstrate that the source-free autonomous toy equation applies. Conversely, a failed global Riccati fit says that this scalar autonomous family cannot represent the observed room profile well under this protocol; it does not refute Maxwell.

Derivative labels are outputs of a fitted kernel. The 0.3 m band has transverse variation; sensor noise, smoothing, position errors and drift also affect them. B here is field magnitude, so its derivative is not a curl component. Current density, electric field and out-of-plane gradients are unobserved. No missing source is manufactured from Maxwell.

A nonconstant solution of a smooth scalar autonomous first-order equation cannot repeatedly turn up and down: at each B value, the equation assigns only one slope. The multi-peaked learned whole-band profile is therefore structurally incompatible with a single Riccati/tanh transition. This concerns the fitted profile and tested representation, not an independent physical derivative measurement.

Next: test whether coefficients and derivatives remain stable under changed training subsets and band width, and replicate the fixed procedure on an independent public dataset. A genuine source-response law needs independent source measurements or a suitable controlled-source public dataset. Compact fit, derivative agreement, and a universal field law are different claims.

Whole-band and local pooled sample counts can differ because each region has its own fixed cap. Compare competing models within the same table row; cross-scope differences are not matched-sample error estimates.

[Plots](comparison.png) · [Full audit, model equations and all per-trial results](results.json).

Reproduce: `python experiments/magnetic_mapping/run_symbolic_field.py` from the workspace root.

Source: Solin et al. (2018), [public Aalto magnetic recordings](https://github.com/AaltoML/magnetic-data), [paper](https://doi.org/10.1109/TRO.2018.2830326). Hashes, pinned version and upstream preprocessing are retained in results.json. The authors calibrated and orientation-corrected these measurements; no raw-variable discovery is claimed.

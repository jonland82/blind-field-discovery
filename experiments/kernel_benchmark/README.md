# A known magnetic-like function, learned as a kernel

**Purpose:** verify the numerical workflow on a known synthetic example. The generator supplies a hidden analytic function; the learner receives sampled values; the hidden truth is revealed for final scoring. This is a software/method check, separate from discovering physics in measured data.

Run from the workspace root:

```powershell
python experiments/kernel_benchmark/run_demo.py
python -m pytest experiments/kernel_benchmark/tests -q
```

Dependencies are NumPy, SciPy, Matplotlib and pytest, already used elsewhere in the project. [Results](results/RESULTS.md) · [Comparison plot](results/comparison.png) · [Implementation](run_demo.py).

The [three-page PDF note](../../papers/01-kernel-to-equation/kernel_to_equation.pdf) presents the question, formalism, experiment and symbolic recovery with supporting plots. [LaTeX source and build instructions](../../papers/01-kernel-to-equation/README.md).

## 1. The hidden truth

Work in normalized, dimensionless coordinates and channel units. Choose

$$
B(x)=B_0\tanh(x/L),\qquad
J(x)=J_0\operatorname{sech}^2(x/L),
$$

with $B_0=1.7$, $L=0.9$, and $J_0=0.8$. Here $\operatorname{sech}(z)=1/\cosh(z)$. Differentiation gives

$$
B'(x)=aJ(x),\qquad
a=\frac{B_0}{LJ_0}=\frac{85}{36}\approx2.361111111.
$$

This resembles a static current layer: for $\mathbf B=(0,B(x),0)$ and $\mathbf J=(0,0,J(x))$, the magnetic curl equation reduces to $B'=\mu_0J$ in SI units. Our dimensionless coefficient $a$ includes the chosen normalization; it is **not an estimate of the SI permeability**.

The generator deliberately builds a consistent pair. That is appropriate for checking whether the algorithm recovers a known relationship. It supplies no independent physical evidence for that relationship.

## 2. What the learner sees

For each of two runs (noiseless and noisy), it receives:

- 80 sampled positions and paired channel values for kernel training;
- 60 other positions and channel values to select smoothing parameters;
- 100 positions with source-channel values for fitting candidate equations;
- 100 other positions with source-channel values for equation validation.

The positions lie in $[-3,3]$; equation discovery and validation use $[-2.8,2.8]$ to reduce boundary derivative effects. All sets are generated with a fixed seed. Noise standard deviations are $0.001B_0$ and $0.001J_0$ in the noisy run. These are fixed channel scales, not noise relative to the local value.

The `learn(samples)` function receives no generator parameters, analytic formulas, true derivatives, or test labels. It is supplied with a Gaussian kernel family, coordinates, the choice to examine the first channel's derivative, and a candidate expression library. These are explicit modeling choices, not learned facts. Channel names B and J describe the generator and are not semantic inputs to regression.

## 3. Learn and differentiate the kernel representation

Fit both channels with

$$
\widehat B(x)=\overline B+\sum_i\alpha_i k_\ell(x,x_i),\qquad
\widehat J(x)=\overline J+\sum_i\beta_i k_\ell(x,x_i),
$$

$$
k_\ell(x,x_i)=\exp\left[-\frac{(x-x_i)^2}{2\ell^2}\right],\qquad
\begin{pmatrix}\boldsymbol\alpha&\boldsymbol\beta\end{pmatrix}
=(K+\lambda I)^{-1}(Y-\mathbf1\overline Y).
$$

$Y$ contains the observed channel values, $K$ contains pairwise kernel values, $\ell$ is a smoothing length, and $\lambda$ regularizes the solve. Validation selects $\ell$ and $\lambda$ by channel-standardized prediction error, without seeing oracle derivatives. The candidate grid is specified in the implementation before fitting.

The field derivative is available analytically:

$$
\widehat B'(x)=-\sum_i\alpha_i\frac{x-x_i}{\ell^2}k_\ell(x,x_i).
$$

Good value predictions alone do not guarantee good derivatives, so both are scored separately against the oracle after learning.

## 4. Recover a compact relationship

Fit candidate expressions for $\widehat B'$ from the library

$$
\{1,B,J,x,xB,xJ,B^2,BJ,J^2\}.
$$

At discovery and equation-validation positions, $B$ is the learned field and $J$ is a separately sampled source-channel value. Search all supports with one to three terms. Select the smallest support whose validation residual is at most 1% of derivative RMS; among equal-size supports choose the smallest residual. If none meets that predeclared numerical tolerance, report failure and the best residual. The tolerance is not a statistical confidence interval.

Both runs select the single term $J$, yielding $\widehat B'\approx\widehat aJ$. The expected coefficient is not supplied to fitting or selection.

## 5. The kernel version of the recovered equation

Integrating the selected equation using the learned source gives a second field representation:

$$
\widetilde B(x)=B_{\mathrm{anchor}}+\widehat a
\left[\overline J(x-x_0)+\sum_i\beta_i\kappa_\ell(x,x_i;x_0)\right],
$$

$$
\kappa_\ell(x,x_i;x_0)
=\int_{x_0}^{x}k_\ell(u,x_i)\,du
=\ell\sqrt{\frac\pi2}\left[
\operatorname{erf}\left(\frac{x-x_i}{\sqrt2\ell}\right)
-\operatorname{erf}\left(\frac{x_0-x_i}{\sqrt2\ell}\right)\right].
$$

Here $\operatorname{erf}$ is the Gaussian error function, $x_0=-3$, and $B_{\mathrm{anchor}}$ is one separately observed boundary value, subject to the same field noise. No true field curve is given to this predictor.

**This representation satisfies $\widetilde B'=\widehat a\widehat J$ exactly as a mathematical identity**, up to floating-point arithmetic. Agreement of $\widetilde B$ with the hidden analytic $B$ is a separate, empirical numerical check. The independently fitted $\widehat B$ and the integrated $\widetilde B$ need not be identical.

This is the small-scale bridge we wanted: samples produce kernel weights, those weights permit differentiation and equation selection, and the selected equation produces an explicit integral-kernel predictor with a boundary value.

## 6. Recover the original analytic function family

The integral-kernel representation alone does not recover the hidden tanh/sech-squared form. A second, field-only search uses $\{1,B,B^2,B^3,B^4\}$, excluding the source and explicit coordinate. With the same one-to-three-term support search and 1% validation-residual threshold, both runs select

$$
B'=p-qB^2,\qquad p,q>0.
$$

No tanh or sech functions are regression candidates. The autonomous polynomial search class is supplied; the selected powers and coefficients come from the learned kernel and its derivative.

Separating variables on the branch $|B|<A$ gives

$$
\frac{1}{\sqrt{pq}}\operatorname{artanh}(B/A)=x-x_c,\qquad
A=\sqrt{p/q},\qquad L=1/\sqrt{pq}.
$$

Thus the solved analytic functions are

$$
B_{\mathrm{sym}}(x)=A\tanh\!\left(\frac{x-x_c}{L}\right),\qquad
J_{\mathrm{sym}}(x)=\frac{p}{\widehat a}\operatorname{sech}^2\!\left(\frac{x-x_c}{L}\right).
$$

At the median training position $x_r$, take $b_r=\widehat B(x_r)$. Both runs have $|b_r|<A$, selecting the bounded branch, and $x_c=x_r-L\operatorname{artanh}(b_r/A)$. This interior kernel reference is distinct from the observed boundary value in the integral predictor. All selection and reference calculations finish before oracle scoring.

Solving the quadratic equation uses a supplied analytic rule, rather than a general-purpose symbolic solver. It is not an additional learned law. If the accepted support, signs, or reference do not support this branch, the implementation reports that it did not recover the intended closed form.

The noisy run recovers $A\approx1.699495$, $L\approx0.899139$, and $x_c\approx-0.000644$, versus the hidden values 1.7, 0.9, and 0. Its field/source relative RMS errors on the test interval are about 0.045% and 0.089%. Field extrapolation error at $x=3.05$ to $4$ drops from approximately 24% for the Gaussian fit to 0.029% for the solved function. See [the full numerical report](results/RESULTS.md) and [symbolic-recovery plot](../../papers/01-kernel-to-equation/symbolic_recovery.png).

## 7. What this verifies

After all selections, reveal the true functions and compare on 901 unseen positions in $[-3,3]$: field values, source values, derivatives, the recovered coefficient, and the integrated equation's field predictions. Also report extrapolation at $x=3.05$ to $4$ separately. Raw samples, learned weights and scored predictions are saved in the two NPZ archives.

Agreement is approximate over a tested interval. A finite Gaussian expansion is not generally identical everywhere to a hyperbolic tangent. Boundary derivatives and extrapolation are appreciably less reliable, and one seed/function family is not a robustness study.

Equation representations are also not unique. This same profile satisfies

$$
B'=\frac{B_0}{L}-\frac{B^2}{B_0L}.
$$

Selecting the simpler one-term $aJ$ relation within the declared library does not establish a unique underlying theory. Here it checks recovery of a deliberately hidden, known relationship. The earlier [measured radio experiment](../radio_response/README.md) addresses a separate question: which representations predict actual measurements.

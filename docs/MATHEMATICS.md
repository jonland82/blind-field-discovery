# How far does this project get toward Maxwell's equations?

Latest real-data extension: [kernel-to-dipole derivation and experiment](../experiments/magnetic_mapping/dipole_field/THEORY.md). Projecting the learned three-component field onto a sum of theoretical magnetic dipoles gives a source-free magnetostatic field whose measured test error improves from 8.09 to 5.22. The assumed source family is supplied physics; the fitted equivalent moments are inferred from the kernel and do not uniquely identify the room's physical sources.

Mathematical walkthrough of [the implementation](../experiments/maxwell_control/run_discovery.py) and [the saved results](../experiments/maxwell_control/results.json), 16 November 2026.

For the later connection between learned similarity kernels and electromagnetic response kernels, see [the public-data radio theory and experiment](../experiments/radio_response/THEORY_AND_EXPERIMENT.md) and [its measured results](../experiments/radio_response/results/RESULTS.md). The derivations below describe the original synthetic equation-discovery experiment.

For a smaller worked example, see [the synthetic kernel-to-equation demonstration](../experiments/kernel_benchmark/README.md): hide a known magnetic-like function, learn samples with Gaussian kernels, recover its derivative relationship, and construct an explicit integral-kernel predictor. [Results](../experiments/kernel_benchmark/results/RESULTS.md) check the numerical workflow against the hidden function, separately from physical discovery.

The [three-page note](../papers/01-kernel-to-equation/kernel_to_equation.pdf) completes that example by inferring a field-only differential equation and solving it to recover the original tanh/sech-squared family. The [real magnetic-profile follow-up](../experiments/magnetic_mapping/symbolic_field/RESULTS.md) applies the same chain to public measurements: it finds useful local analytic approximations, but no equation passes the joint derivative/prediction checks in this run.

**The synthetic experiment recovers the term structure of all eight scalar Maxwell equations, plus charge conservation, in dimensionless units. The real phone recording has not yielded a learned electromagnetic equation in this implementation.** The synthetic success establishes that the sparse learner can extract relations from a favorable dataset whose generator already obeys Maxwell. It does not establish Maxwell from independent physical measurements.

This document separates equations actually fitted by the program, properties built into the generator, mathematical consequences of the recovered structure, and what remains unmeasured.

## 1. The target: what counts as recovering Maxwell?

Let $\mathbf E=(E_x,E_y,E_z)$ be the electric field, $\mathbf B=(B_x,B_y,B_z)$ the magnetic field, $\rho$ the charge density, and $\mathbf J=(J_x,J_y,J_z)$ the current density. Each is a function of position $\mathbf x=(x,y,z)$ and time $t$.

In SI units, the microscopic Maxwell equations with sources are

$$
\nabla\cdot\mathbf E=\frac{\rho}{\epsilon_0},
\qquad
\nabla\cdot\mathbf B=0,
$$

$$
\nabla\times\mathbf E=-\partial_t\mathbf B,
\qquad
\nabla\times\mathbf B=\mu_0\mathbf J+\mu_0\epsilon_0\partial_t\mathbf E.
$$

These are four vector-calculus statements but **eight scalar equations**: two divergence equations and three components of each curl equation. Here “with sources” allows nonzero $\rho$ and $\mathbf J$; it does not mean a source-free vacuum wave experiment. For background on these equations and their wave consequences, see [TU Delft's electromagnetic optics text](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap1_Basics/Basics_2022_01Clean.html).

The project's generator uses dimensionless variables with the corresponding permittivity and permeability set to one. Its target system is therefore

$$
\boxed{\nabla\cdot\mathbf E=\rho,\qquad\nabla\cdot\mathbf B=0,}
$$

$$
\boxed{\partial_t\mathbf B=-\nabla\times\mathbf E,\qquad
\partial_t\mathbf E=\nabla\times\mathbf B-\mathbf J.}
$$

Recovering these structures requires selecting the right channels, coordinate derivatives, relative signs, and coefficient ratios. An equation can be multiplied by any nonzero constant without changing its meaning, so absolute coefficient normalization is arbitrary.

It does not require discovering names such as “electric field.” Those names are attached afterward using the known generator mapping. Nor does recovering these field equations discover the Lorentz force law, a model for the motion of charges, or constitutive laws for materials.

## 2. The synthetic observations

The learner sees ten scalar channels $u_0,\ldots,u_9$. Their eventual interpretation is

| Channels | Meaning revealed after fitting |
| --- | --- |
| $u_0,u_1,u_2$ | $E_x,E_y,E_z$ |
| $u_3,u_4,u_5$ | $B_x,B_y,B_z$ |
| $u_6$ | $\rho$ |
| $u_7,u_8,u_9$ | $J_x,J_y,J_z$ |

The coordinate axes are supplied as $(x,y,z,t)$. In the saved feature names, `d0`, `d1`, `d2`, and `d3` mean differentiation with respect to those four coordinates, respectively.

Each experiment has 512 random center points in $[-\pi,\pi]^4$. There are 12 training experiments, four validation experiments, and four test experiments:

$$
n_{\rm train}=6144,\qquad n_{\rm valid}=2048,\qquad n_{\rm test}=2048.
$$

The test experiments have new random seeds and source configurations, while using the same family of generators, noise level, and sampling scheme. This tests transfer within that synthetic family.

### 2.1 Numerical derivatives and their cost

Write $q=(x,y,z,t)$ and let $\mathbf e_a$ be coordinate direction $a$. Each derivative is estimated from additional observations:

$$
\widehat{\partial_a u_j}(q)
=\frac{\widetilde u_j(q+h\mathbf e_a)-\widetilde u_j(q-h\mathbf e_a)}{2h},
\qquad h=0.02.
$$

The observed value is $\widetilde u_j=u_j+\eta_j$, where pointwise Gaussian noise has standard deviation $10^{-4}$ times that channel's RMS within the experiment. Perturbations at the plus and minus stencil locations are independent.

Thus a center point requires the center observation plus eight displaced observations, each containing all ten channels. The reported center-point counts do not include these extra stencil observations. In particular, the synthetic experiment has local samples along all three spatial directions and along time. The phone route does not provide that sampling pattern.

For a smooth noiseless function, Taylor expansion gives

$$
\frac{u(q+h\mathbf e_a)-u(q-h\mathbf e_a)}{2h}
=\partial_a u(q)+\frac{h^2}{6}\partial_a^3u(q)+O(h^4).
$$

If each stencil observation has noise variance $\sigma_j^2$, the derivative noise has variance

$$
\operatorname{Var}\!\left[\frac{\eta_+-\eta_-}{2h}\right]
=\frac{\sigma_j^2}{2h^2}.
$$

At $h=0.02$, its standard deviation is approximately $0.00354$ times the channel RMS **per coordinate unit**. This is not a universal relative derivative error: that depends on the magnitude of the true derivative. It explains why 0.01% pointwise measurement noise need not produce 0.01% equation residuals.

For a Fourier component $e^{i\kappa q_a}$, central differencing replaces the exact multiplier $i\kappa$ by

$$
i\frac{\sin(\kappa h)}{h}
=i\kappa\left(1-\frac{\kappa^2h^2}{6}+O((\kappa h)^4)\right).
$$

With $|\kappa|\leq 3$ in this generator, the leading attenuation reaches approximately $0.06\%$. Small fitted coefficient shifts are therefore expected even before considering noise. This gives an error scale, not a complete error budget or a statistical explanation of each coefficient.

## 3. The mathematical search space

The candidate feature vector is

$$
\theta(q)=\left(
1,\ \{u_j\}_{j=0}^{9},\
\{\widehat{\partial_a u_j}\}_{j=0,\ldots,9;\ a=0,\ldots,3},\
\{u_ju_k\}_{0\leq j\leq k\leq9}
\right).
$$

Its length is

$$
p=1+10+4\times10+\frac{10\times11}{2}=106.
$$

Stacking observations gives a matrix $\Theta\in\mathbb R^{n\times106}$. The learner searches for sparse, nonzero coefficient vectors $c$ satisfying

$$
\Theta c\approx0.
$$

For example, electric Gauss law would appear as

$$
u_6-\partial_xu_0-\partial_yu_1-\partial_zu_2\approx0.
$$

The feature list contains its ingredients separately; it does not contain a preassembled $\nabla\cdot\mathbf E$ column. Likewise, curl components must emerge from selecting and combining individual derivatives.

“Blind” has a specific scope here. The learner is not passed electromagnetic labels, equation supports, divergence/curl operators, or physical coefficients. It is passed a coordinate system, measured channels already expressed in useful physical variables, a polynomial/derivative library, and a preference for sparse equations. It is not asked to recover the variables from arbitrary nonlinear sensor encodings.

The library also limits what can be discovered. It contains first derivatives and raw quadratic products, but no second derivatives, products involving derivatives, nonlocal terms, or arbitrary functions. Consequently, a second-order wave equation cannot be directly selected by this implementation.

## 4. From a feature matrix to sparse equations

### 4.1 Normalize the columns

For feature $k$, compute its training RMS

$$
s_k=\max\left(\sqrt{\frac{1}{n}\sum_{i=1}^n\Theta_{ik}^2},\ 10^{-10}\right),
\qquad D=\operatorname{diag}(s_1,\ldots,s_p).
$$

The standardized matrix is $X=\Theta D^{-1}$. If $Xa\approx0$, then the coefficient vector in the original feature units is $c=D^{-1}a$.

This normalization matters because sparsity is enforced on standardized coefficients. Without scaling, merely changing a channel's units could strongly alter the optimization penalty.

### 4.2 Find approximate null directions by SVD

Compute

$$
X=U\Sigma V^{\mathsf T},\qquad
\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_{106}\geq0.
$$

A small singular value indicates a combination of columns that nearly cancels. The implementation finds the largest ratio $\sigma_k/\sigma_{k+1}$. If that ratio is below 10, it reports no approximate nullspace. Otherwise, it uses the right singular vectors after that gap.

In the saved full-channel run,

$$
\frac{\sigma_{97}}{\sigma_{98}}=139.5220629997156,
\qquad d=106-97=9.
$$

Let $N\in\mathbb R^{106\times9}$ contain those nine right singular vectors. Candidate standardized relations have the form $a=N\alpha$.

These are nine **approximate** null directions. With noisy data, the matrix need not have an exact nine-dimensional kernel. The gap rule is an algorithmic heuristic, not a proof of the number of physical laws.

### 4.3 Prefer sparse combinations

An SVD basis is generally dense: a singular vector may mix several simple equations. To find sparse combinations, the code solves, for each eligible anchor index $j$,

$$
\min_{\alpha}\|N\alpha\|_1
\quad\text{subject to}\quad
(N\alpha)_j=1.
$$

The anchor excludes the otherwise trivial solution $a=0$ and fixes a scale. In linear-programming form, with auxiliary variables $z\in\mathbb R^{106}$, this is

$$
\min_{\alpha,z}\sum_{k=1}^{106}z_k,
\qquad
-z\leq N\alpha\leq z,\quad z\geq0,\quad(N\alpha)_j=1.
$$

Anchors whose row of $N$ has norm below $10^{-3}$ are skipped. Searching anchors treats candidate terms generically; it does not designate a particular time derivative as the dependent variable.

This shares the broad sparse-library idea of [Brunton, Proctor, and Kutz's SINDy work](https://arxiv.org/abs/1509.03580) and [Rudy and colleagues' PDE discovery work](https://arxiv.org/abs/1609.06401). The specific implementation here searches for implicit homogeneous relations through an SVD subspace and anchored linear programs.

### 4.4 Select terms, refit, and validate

Given a standardized candidate $a$, retain the support

$$
S=\{k:|a_k|>0.025\max_\ell|a_\ell|\}.
$$

Only supports with $2\leq|S|\leq8$ are considered. On each support, the smallest right singular vector of $X_S$ solves

$$
\min_{\|b\|_2=1}\|X_Sb\|_2.
$$

Convert back to original feature units, then normalize the first listed coefficient to $+1$. The refit is performed on the selected support; it need not remain exactly inside the original nine-dimensional subspace.

For a dataset with $m$ rows, the relative cancellation residual is

$$
R(c)=
\frac{\sqrt{\frac1m\sum_{i=1}^{m}\left(\sum_{k\in S}\Theta_{ik}c_k\right)^2}}
{\sqrt{\frac1m\sum_{i=1}^{m}\sum_{k\in S}(\Theta_{ik}c_k)^2}}.
$$

The denominator measures the size of the individual equation terms before they cancel. This is not a percentage prediction error for a field trajectory, nor a probability that the equation is correct.

The implementation rejects validation residuals above $0.02$, deduplicates identical supports, and finally evaluates the accepted equations on the held-out test experiments. It does not require all accepted supports to be mutually independent.

## 5. The nine equations actually recovered

Below, coefficients are rounded to eight decimal places from `results.json`. Derivative notation stands for the numerically estimated features used during fitting. Each left-hand side is approximately zero on the observations.

### 5.1 Electric Gauss law

$$
\rho
-1.00046632\,\partial_xE_x
-1.00048054\,\partial_yE_y
-1.00048364\,\partial_zE_z
\approx0.
$$

Its support and signs match $\rho-\nabla\cdot\mathbf E=0$.

### 5.2 Ampere-Maxwell law: three components

$$
J_x+1.00042597\,\partial_tE_x
+1.00054253\,\partial_zB_y
-1.00051490\,\partial_yB_z
\approx0,
$$

$$
J_y+1.00046643\,\partial_tE_y
-1.00054766\,\partial_zB_x
+1.00055785\,\partial_xB_z
\approx0,
$$

$$
J_z+1.00037874\,\partial_tE_z
+1.00045206\,\partial_yB_x
-1.00044358\,\partial_xB_y
\approx0.
$$

Replacing the fitted coefficients by their ideal values gives

$$
\mathbf J+\partial_t\mathbf E-\nabla\times\mathbf B=0.
$$

The displacement-current terms $\partial_t\mathbf E$ are selected along with the current and magnetic curl terms. The recovery therefore includes electromagnetic dynamics, beyond electrostatics and magnetostatics.

### 5.3 Faraday's law: three components

The saved equations occur in the component order $z,y,x$:

$$
\partial_yE_x-1.00005193\,\partial_xE_y
-1.00000340\,\partial_tB_z\approx0,
$$

$$
\partial_zE_x-0.99993912\,\partial_xE_z
+0.99981932\,\partial_tB_y\approx0,
$$

$$
\partial_zE_y-0.99994557\,\partial_yE_z
-0.99983811\,\partial_tB_x\approx0.
$$

After harmless equation sign changes where needed, these match

$$
\partial_t\mathbf B+\nabla\times\mathbf E=0.
$$

### 5.4 Magnetic Gauss law

$$
\partial_xB_x+0.99999800\,\partial_yB_y
+1.00008182\,\partial_zB_z\approx0.
$$

This matches $\nabla\cdot\mathbf B=0$.

### 5.5 Charge continuity

$$
\partial_t\rho
+1.00020347\,\partial_xJ_x
+1.00012762\,\partial_yJ_y
+1.00016427\,\partial_zJ_z\approx0.
$$

This matches $\partial_t\rho+\nabla\cdot\mathbf J=0$.

### 5.6 Accuracy on held-out experiments

| Recovered relation | Test cancellation residual | Largest coefficient deviation from ideal magnitude |
| --- | ---: | ---: |
| Electric Gauss law | 0.100867% | 0.048364% |
| Ampere-Maxwell, $x$ | 0.120608% | 0.054253% |
| Ampere-Maxwell, $y$ | 0.125170% | 0.055785% |
| Ampere-Maxwell, $z$ | 0.123890% | 0.045206% |
| Faraday, $z$ | 0.180733% | 0.005193% |
| Faraday, $y$ | 0.183270% | 0.018068% |
| Faraday, $x$ | 0.197147% | 0.016189% |
| Magnetic Gauss law | 0.204054% | 0.008182% |
| Charge continuity | 0.169046% | 0.020347% |

All nine supports match the ideal relations. No constant, raw quadratic product, or unrelated channel term survives in these equations. The largest coefficient deviation is approximately $0.0558\%$ under the program's normalization.

The fitted coefficients are close to the ideal values, but are not exactly equal across axes. The compact vector Maxwell system describes their recovered structure and ideal limit; the raw fitted equations are approximate and need not obey exact rotational symmetry or all compatibility identities. No joint symmetry-constrained fit or confidence intervals are implemented.

## 6. Why nine relations do not mean nine independent physical laws

Take the divergence of ideal Ampere-Maxwell:

$$
\nabla\cdot\mathbf J+\partial_t(\nabla\cdot\mathbf E)
-\nabla\cdot(\nabla\times\mathbf B)=0.
$$

For sufficiently smooth fields, $\nabla\cdot(\nabla\times\mathbf B)=0$. Substituting $\nabla\cdot\mathbf E=\rho$ gives

$$
\boxed{\partial_t\rho+\nabla\cdot\mathbf J=0.}
$$

Continuity is therefore a **differential consequence** of Maxwell. Its recovery is an additional consistency check on the source channels, not an additional fundamental law.

There is no contradiction with finding nine approximate null directions in the feature matrix. A constant linear combination of the original eight scalar coefficient rows cannot create derivatives of $\rho$ or $\mathbf J$ that are absent from those rows. Deriving continuity requires differentiating equations, not just adding them with constant weights. In the 106-dimensional feature space, the nine ideal signed coefficient rows have rank nine; as differential equations, they are not nine independent axioms.

The divergence equations also act as constraints on the dynamics. Faraday implies

$$
\partial_t(\nabla\cdot\mathbf B)=0.
$$

Ampere-Maxwell together with continuity implies

$$
\partial_t(\nabla\cdot\mathbf E-\rho)=0.
$$

Thus, if both Gauss constraints hold initially, ideal evolution preserves them. This explains why counting scalar equations alone is not a count of independent dynamical degrees of freedom.

## 7. Where Maxwell enters the generator

The generator builds a scalar potential $\phi$ and vector potential $\mathbf A$ from 24 random Fourier modes. Schematically,

$$
\mathbf A(\mathbf x,t)=\frac{1}{\sqrt{24}}\operatorname{Re}
\sum_{m=1}^{24}\mathbf a_m e^{i(\mathbf k_m\cdot\mathbf x-\omega_mt)},
$$

$$
\phi(\mathbf x,t)=\frac{1}{\sqrt{24}}\operatorname{Re}
\sum_{m=1}^{24}b_m e^{i(\mathbf k_m\cdot\mathbf x-\omega_mt)}.
$$

Wave-vector components are integers from $-3$ to $3$, with zero vectors replaced; temporal frequencies lie between $-3$ and $3$. There is no imposed source-free dispersion relation $\omega_m^2=|\mathbf k_m|^2$.

The fields and sources are defined by

$$
\mathbf E=-\nabla\phi-\partial_t\mathbf A,
\qquad\mathbf B=\nabla\times\mathbf A,
$$

$$
\rho=\nabla\cdot\mathbf E,
\qquad\mathbf J=\nabla\times\mathbf B-\partial_t\mathbf E.
$$

For an individual Fourier mode, the complex amplitudes in the code are

$$
\mathbf E_m=-i\mathbf k_mb_m+i\omega_m\mathbf a_m,
\qquad\mathbf B_m=i\mathbf k_m\times\mathbf a_m,
$$

$$
\rho_m=i\mathbf k_m\cdot\mathbf E_m,
\qquad\mathbf J_m=i\mathbf k_m\times\mathbf B_m+i\omega_m\mathbf E_m.
$$

This is analytic construction inside the generator. The learner receives noisy pointwise values and finite-difference features, not those analytic formulas.

Two Maxwell equations follow identically from potentials:

$$
\nabla\cdot\mathbf B=\nabla\cdot(\nabla\times\mathbf A)=0,
$$

$$
\nabla\times\mathbf E+\partial_t\mathbf B
=-\nabla\times\nabla\phi
-\partial_t(\nabla\times\mathbf A)
+\partial_t(\nabla\times\mathbf A)=0.
$$

The remaining two hold by the definitions of $\rho$ and $\mathbf J$. Continuity then follows as above. **All nine target relations are present by construction.** Sources are derived from the fields; they are not independent measurements from separately calibrated charge/current sensors.

This is a useful positive control: the learner must identify the sparse relationships without seeing the definitions. It cannot establish that nature follows those relationships because the experimenter explicitly arranged that it would.

## 8. Consequences we can derive, but the code has not separately tested

### 8.1 Electromagnetic wave structure

Starting from the ideal recovered system,

$$
\partial_t^2\mathbf B
=-\nabla\times\partial_t\mathbf E
=-\nabla\times(\nabla\times\mathbf B-\mathbf J).
$$

Using $\nabla\times(\nabla\times\mathbf B)=\nabla(\nabla\cdot\mathbf B)-\nabla^2\mathbf B$ gives

$$
\boxed{(\partial_t^2-\nabla^2)\mathbf B=\nabla\times\mathbf J.}
$$

Similarly,

$$
\boxed{(\partial_t^2-\nabla^2)\mathbf E=-\nabla\rho-\partial_t\mathbf J.}
$$

In a source-free region, these reduce to wave equations with dimensionless speed one. Thus the recovered ideal structure contains wave propagation. The program does not directly fit these second-order equations or run a forward solver to demonstrate predicted propagation.

### 8.2 Electromagnetic energy balance

Dot Ampere-Maxwell with $\mathbf E$, dot Faraday with $\mathbf B$, and add:

$$
\partial_t\frac{|\mathbf E|^2+|\mathbf B|^2}{2}
=\mathbf E\cdot(\nabla\times\mathbf B)
-\mathbf B\cdot(\nabla\times\mathbf E)-\mathbf J\cdot\mathbf E.
$$

The identity $\nabla\cdot(\mathbf E\times\mathbf B)=\mathbf B\cdot(\nabla\times\mathbf E)-\mathbf E\cdot(\nabla\times\mathbf B)$ yields

$$
\boxed{\partial_t\left(\frac{|\mathbf E|^2+|\mathbf B|^2}{2}\right)
+\nabla\cdot(\mathbf E\times\mathbf B)=-\mathbf J\cdot\mathbf E.}
$$

This is the dimensionless Poynting balance. It is a mathematical implication of the ideal equations, not an additional measured or fitted result in this repository. The current library lacks the derivative/product combinations needed to select this balance directly.

### 8.3 What the unit coefficients do not measure

To see how units can produce the project's target equations, choose a length scale $L$ and electric-field scale $E_0$, and define

$$
c_0=\frac{1}{\sqrt{\mu_0\epsilon_0}},\qquad
t_0=\frac{L}{c_0},\qquad B_0=\frac{E_0}{c_0},
$$

$$
\rho_0=\frac{\epsilon_0E_0}{L},\qquad J_0=\rho_0c_0.
$$

Write $\mathbf x=L\mathbf x'$, $t=t_0t'$, $\mathbf E=E_0\mathbf E'$, and similarly scale $\mathbf B,\rho,\mathbf J$. Substitution into SI Maxwell gives the dimensionless equations of Section 1. Dropping primes produces the generator's notation.

The benchmark begins with dimensionless observations. Its recovered coefficients near $\pm1$ therefore do not independently estimate $\epsilon_0$, $\mu_0$, or the physical speed of light in metres per second. That would require dimensional calibration and independent experimental evidence.

## 9. What the controls establish

| Data supplied to the same learner | Candidate count | Accepted relations | Interpretation |
| --- | ---: | ---: | --- |
| All ten synthetic channels | 106 | 9 | All eight scalar Maxwell structures plus continuity |
| Three synthetic magnetic channels only | 22 | 1 | Magnetic Gauss law only |
| Independently shuffled full-library columns | 106 | 0 | No accepted relation after joint alignment is destroyed |

For three magnetic channels, the library size is $1+3+12+6=22$. These observations still sample a synthetic three-dimensional volume and time; they are much richer geometrically than a phone route. Recovering $\nabla\cdot\mathbf B\approx0$ under these conditions does not show that it can be recovered from a moving phone.

Magnetic data alone cannot form the coupled first-order curl equations in this library: the needed $\mathbf E$ and $\mathbf J$ terms are missing. This result concerns the present library and benchmark; it is not a theorem excluding every possible inference from magnetic data under additional assumptions.

Shuffling preserves each candidate column's marginal distribution while disrupting cross-column relations. Because the derivatives and products are shuffled independently too, the result is generally not a realizable field dataset. It checks whether this learner accepts identities after alignment is destroyed. One such control does not establish a general false-positive rate or robustness guarantee.

## 10. Why the phone recording does not reach Maxwell

### 10.1 What is actually analyzed

The real-data branch loads `data/advio-12/iphone/arkit.csv`, takes its three position columns, and computes their principal components. It does **not** load real electromagnetic channels into `discover`, estimate their derivatives, or report a field equation fitted to the phone recording.

The code establishes a route-geometry diagnostic. It does not claim an empirical test of magnetic Gauss law from the phone data. Electric field, charge density, and current density observations are absent from the real-data inputs used here.

### 10.2 A moving measurement gives a directional combination

Even supposing a perfectly calibrated magnetic measurement along a known route were available, write

$$
\mathbf b(t)=\mathbf B(\mathbf r(t),t).
$$

The chain rule gives

$$
\frac{d\mathbf b}{dt}
=\partial_t\mathbf B+(\dot{\mathbf r}\cdot\nabla)\mathbf B.
$$

For each field component, this is one directional combination of four coordinate derivatives. It does not separate intrinsic time variation from spatial variation encountered by the moving sensor.

A stationarity assumption $\partial_t\mathbf B=0$ would remove one unknown contribution, but would still leave only a directional spatial derivative at each route point. Further repeated sampling, geometry, or model assumptions would be needed to reconstruct a spatial gradient. These assumptions would need to be stated as inputs to any discovery claim.

### 10.3 The PCA diagnostic

Let $P$ be the centered matrix of measured positions, with singular values $s_1\geq s_2\geq s_3$. Its variance fractions are

$$
f_i=\frac{s_i^2}{s_1^2+s_2^2+s_3^2}.
$$

The saved values are

$$
(f_1,f_2,f_3)
=(0.8500292715,\ 0.1499401807,\ 0.0000305478).
$$

Thus $99.996945\%$ of position variance lies within the best-fit plane. RMS displacement normal to that plane is $0.104718$ m, compared with $17.468211$ m along the largest principal direction.

This measures the route's geometry. It is not an error estimate for a fitted electromagnetic derivative, and nonzero off-plane variance does not prove adequate local three-dimensional sampling. Conversely, near-planarity alone is not a proof that every derivative is strictly unobservable.

### 10.4 An explicit ambiguity in unmeasured directions

Rotate coordinates so an idealized observation plane is $z=0$. Consider two otherwise arbitrary smooth field extensions,

$$
\widetilde{\mathbf B}=\mathbf B+(0,0,\lambda z).
$$

On the plane, $\widetilde{\mathbf B}=\mathbf B$, so every planar magnetic observation agrees. But

$$
\nabla\cdot\widetilde{\mathbf B}=\nabla\cdot\mathbf B+\lambda.
$$

Likewise, the alternative

$$
\widehat{\mathbf B}=\mathbf B+(\lambda z,0,0)
$$

agrees with $\mathbf B$ on the plane while

$$
\nabla\times\widehat{\mathbf B}=\nabla\times\mathbf B+(0,\lambda,0).
$$

Planar measurements alone therefore cannot decide these divergence and curl components among unrestricted smooth field extensions. Rejecting an extension because it violates Maxwell would assume the law being tested. The actual route is not an exact plane; these constructions explain the ambiguity, rather than proving exact non-identifiability for every feature of the measured route.

Missing source and electric-field measurements provide a separate obstacle to recovering the complete coupled system, even if the route geometry were improved.

## 11. How much has been achieved, and what would move it further?

| Level of claim | Status in this project |
| --- | --- |
| Select all eight scalar Maxwell term structures from unnamed channels | Achieved on the supplied synthetic benchmark |
| Estimate their dimensionless relative coefficients accurately | Achieved here, with deviations below about 0.056% |
| Recover charge conservation | Achieved synthetically; a differential consequence of Maxwell |
| Transfer to new source configurations within the generator family | Supported by held-out residuals around 0.10–0.20% |
| Identify the physical meanings of arbitrary sensor outputs | Not demonstrated; the mapping is revealed from generator knowledge |
| Establish dimensional constants or measure a physical wave speed | Not demonstrated |
| Predict field evolution with a learned PDE solver | Not implemented or tested |
| Learn a Maxwell relation from the actual phone recording | Not demonstrated; only real position geometry is analyzed |
| Independently establish electromagnetic laws from experiment | Not demonstrated |

There is no useful single percentage of “Maxwell discovered.” **Equation structure is fully recovered in the synthetic control; independent experimental identification remains open.**

Useful next steps have different purposes:

1. **Test numerical reliability.** Sweep noise, differencing step, sample density, and random seeds. Report how often each support is recovered, coefficient variability, and failure thresholds. The present results describe one fixed protocol, not a robustness study.
2. **Test a broader hypothesis space.** Vary the feature library and channel scaling, and use alternatives that deliberately violate a chosen law. This would probe whether the result depends strongly on the selected representation or only on an especially favorable generator.
3. **Test predictive dynamics.** Convert a learned coupled system into a solver and evaluate future fields on independent initial/boundary-value problems with supplied source evolution. Monitor both field errors and divergence-constraint drift. The current cancellation metric does not substitute for this test.
4. **Make the real observation problem identifiable.** Collect calibrated measurements with enough spatial and temporal diversity to estimate the needed derivatives, plus independent electric-field and source information for the full system. A direct magnetic-divergence study could be a narrower first experimental target, but needs suitable volume sampling and a sensor-error analysis.

For a claim about the complete system, deriving missing $\rho$ or $\mathbf J$ from assumed Maxwell relations and then “recovering” those same relations would reproduce the logic of the synthetic control. A real experimental test needs independent observables or explicitly stated additional assumptions.

## 12. Reading and checking the evidence

- [Implementation](../experiments/maxwell_control/run_discovery.py): `make_control` defines the synthetic physics; `library` supplies candidate terms; `discover` selects relations; `evaluate` computes held-out residuals; `interpret` attaches physical labels after fitting.
- [Saved results](../experiments/maxwell_control/results.json): full coefficients, supports, split sizes, nullspace diagnostics, controls, and route statistics.
- [Saved synthetic matrices](../experiments/maxwell_control/synthetic_control.npz): the training, validation, and test feature matrices, feature names, and training singular values. The archive stores candidate matrices rather than the raw pointwise stencil observations.
- [Saved route geometry](../experiments/maxwell_control/real_route_geometry.npz): measured positions, principal directions, and variance fractions.
- [Project overview and reproduction instructions](../experiments/maxwell_control/README.md).

The numerical table above was checked against the saved coefficients and residuals, including recomputing test cancellation residuals from the archived feature matrix. Those checks verify the reported artifact values; they do not add new physical experiments.

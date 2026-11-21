# Decision stumps as exact Maxwell fronts

*Working note: origin, claims, proofs to write out, extensions, open questions.*
*Origin: §5 of "Equivalent Descriptions of Light" (J. R. Landers, Nov 2026), where a
continuous average of threshold stumps reproduces a sinusoidal plane wave and a
finite sample is described as "a stepwise approximation." The claim here is
stronger: every stump is itself an exact (weak) Maxwell solution, so a finite
forest is an exact field and only the initial profile is approximated.*

---

## 1. Setup

Vacuum, no sources, plane waves along $z$, one linear polarization:
$E = E_x(z,t)\,\hat x$, $B = B_y(z,t)\,\hat y$. Maxwell's equations reduce to

$$\partial_t E_x + c^2 \partial_z B_y = 0, \qquad \partial_t B_y + \partial_z E_x = 0,$$

with $\nabla\cdot E = \nabla\cdot B = 0$ automatically.

This is a linear hyperbolic system $\partial_t U + A\,\partial_z U = 0$ with
$U = (E_x, B_y)$ and $A = \begin{pmatrix} 0 & c^2 \\ 1 & 0 \end{pmatrix}$, eigenvalues $\pm c$.

**Riemann invariants:** $w_\pm = E_x \pm c B_y$ satisfy $\partial_t w_\pm \pm c\,\partial_z w_\pm = 0$.
So $w_+$ is carried right at speed $c$ and $w_-$ left. A purely right-moving wave has $w_- \equiv 0$,
i.e. $cB_y = E_x$.

## 2. Lemma 1: any profile is exact

For any profile $f$, the field

$$E = \hat x\, f(z - ct), \qquad B = \hat y\, f(z - ct)/c$$

solves vacuum Maxwell:
- classically if $f \in C^1$ (check: $\partial_t E_x = -c f'$, $-c^2\partial_z B_y = -c f'$; $\partial_t B_y = -f'$, $-\partial_z E_x = -f'$);
- weakly (distributionally) if $f$ is merely of bounded variation (BV), including step functions.

This is d'Alembert's solution, restricted to one direction.

## 3. Lemma 2: a stump is a one-wave Riemann problem

Take $f(\xi) = \mathbf 1\{\xi \ge a\}$, a single decision stump in the phase $\xi = z - ct$.

- **Jump (Rankine–Hugoniot) conditions** for a discontinuity moving at speed $s$:
  $s[E_x] = c^2[B_y]$ and $s[B_y] = [E_x]$, so $s^2 = c^2$. For $s = +c$: $[B_y] = [E_x]/c$,
  equivalently $[B] = \hat z \times [E]/c$, with $\hat z \cdot [E] = 0$ (transverse jump).
- The stump satisfies these exactly. As initial data it is a **Riemann problem**
  (piecewise-constant states with one jump) whose solution has only the right-moving wave,
  because the jump lies entirely in $w_+$ and $[w_-] = 0$.

**One-line version:** each tree is a solved Riemann problem.

## 4. Theorem: forest representation of every unidirectional wave

For $f$ absolutely continuous (or BV, via Stieltjes):

$$f(\xi) = f(\xi_0) + \int \mathbf 1\{\xi \ge a\}\, df(a)
\;\;\big(= f(\xi_0) + \textstyle\int f'(a)\,\mathbf 1\{\xi \ge a\}\,da \text{ for AC } f\big).$$

**Forest form:** draw thresholds $a \sim p$ and give each stump the weight $w(a) = f'(a)/p(a)$, so
$T_a(\xi) = w(a)\,\mathbf 1\{\xi \ge a\}$ and $\mathbb E_a[T_a(\xi)] = f(\xi) - f(\xi_0)$.

- The Light paper's case: $p$ uniform on $[0, 2\pi)$, $f = \sin$, weight $2\pi\cos a$.
- **Periodic / wrapped phase:** continuity of the average at the wrap point requires
  $\int_0^{2\pi} f'(a)\,da = 0$, i.e. zero-mean weights. Individual stumps jump at the wrap; the average doesn't.
- Atoms of $df$ (jumps in $f$) are just stumps with point-mass thresholds. So BV profiles, shocks
  included, are covered.

**Corollary (the reframe).** By linearity, a finite forest
$\hat f_N = f(\xi_0) + \frac1N\sum_{k} T_{a_k}$ gives a field $(\hat x \hat f_N,\ \hat y \hat f_N/c)$ that is an
**exact weak Maxwell solution** for every $N$. The forest approximates the *initial profile*, not the
equations: there is zero equation residual, and the profile error is transported unchanged at speed $c$.
That is approximation *within the solution space*, as opposed to approximating the residual.

## 5. Error rates

- **Importance sampling:** the variance-minimizing density is $p \propto |f'|$, giving
  $|w| = \mathrm{TV}(f)$, constant.
- **Monotone $f$:** with $p = f'/\mathrm{TV}$, the forest is exactly $\mathrm{TV}(f)$ times the
  empirical CDF of the thresholds. The DKW inequality then gives a uniform bound:
  $\sup_\xi |\hat f_N - f| \le \mathrm{TV}(f)\sqrt{\log(2/\delta)/(2N)}$ with probability $\ge 1-\delta$.
- **General BV:** use the Jordan decomposition $f = f^+ - f^-$; two empirical CDFs, so roughly
  $(\mathrm{TV}^+ + \mathrm{TV}^-)/\sqrt N$. *To check: constants, and the periodic case.*
- Since the field is an exact solution, the same uniform bound holds for $E$ (and $cB$) at
  **all times**. The error never grows.

## 6. Spacetime geometry: null splits

Treat a stump as a split on spacetime coordinates $(x, ct)$: $\mathbf 1\{n\cdot x - ct \ge a\}$, with $|n| = 1$.
The split hyperplane has normal $(n, -1)$, which is **null** in Minkowski signature.

> **Maxwell fronts are decision stumps whose split hyperplane is lightlike.**

- An oblique decision tree in spacetime with only null splits (plus transverse polarization
  $e \perp n$, $b = n \times e / c$) is an exact vacuum Maxwell field.
- Timelike or spacelike splits are not solutions. Those trees' residuals measure "how far from light" they are.
- *Possible slogan:* the solution space of vacuum Maxwell is the closed span of null stumps.
  (Needs the 3D statement in §7.)

## 7. Extensions

1. **Both directions / general 1D:** two forests, one over $w_+$ (right-moving) and one over $w_-$
   (left-moving); second polarization $(E_y, B_x)$ likewise. This covers all 1D vacuum solutions.
2. **3D:** superpose over directions $n \in S^2$: one forest over (direction, threshold) pairs.
   Relevant machinery: the Lax–Phillips translation representation and Friedlander radiation fields
   (plane-wave/Radon decomposition of wave-equation solutions; in odd dimensions the profile involves
   derivatives of the Radon transform of the data). *To check: the exact 3D formula and which data
   classes get a BV-type profile.* Note: Fourier is one decomposition into plane waves; null stumps are another.
3. **Dispersive media / lattices:** fronts are no longer exact. On the LC ladder (Light paper §6,
   $\Omega^2 = \frac{4}{LC}\sin^2\frac{ka}{2}$), a launched stump develops a ringing wake behind the front.
   The stump is a *probe* of the underlying spacing: exact in the continuum, visibly broken on the lattice.
   This ties back to the "observable departures" theme.
4. **Godunov connection:** Godunov's method represents solutions as piecewise-constant cells with
   Riemann problems at the interfaces; for linear advection at CFL = 1 it is exact. The stump forest is
   a **gridless, randomized Godunov representation**: random thresholds instead of cell edges.
5. **Random forests / kernels:** infinite random-stump ensembles act as kernels (Breiman; Lin & Jeon 2006;
   Scornet 2016). Here the "kernel" is the Heaviside/Green's structure of transport along characteristics.
   Question: does greedy boosting (choosing thresholds by residual) correspond to anything physical,
   e.g. adaptive placement of fronts where $|f'|$ is large?

## 8. Open questions

- The 3D theorem: precise statement, function spaces, and whether "closed span of null stumps"
  is the right topology (energy norm?).
- Energy: each stump carries a constant energy density behind the front. Is the finite forest's energy
  (Poynting flux) close to the target's? It should be within the profile error, but check the $L^2$ vs sup norms.
- With sources ($\rho$, $J$): stumps no longer suffice; Duhamel adds fronts emitted from source events.
  Is that a forest whose thresholds are drawn from the source?
- Which measurements are exact versus approximate for a finite forest (e.g., arrival times of fronts
  are exact; local field values carry the DKW error)?
- Is there a learning angle: fitting a forest of null stumps to sensor data gives a model that
  satisfies Maxwell by construction. (Compare the dipole equivalent-source fit in
  "Magnetic Fields from Measurements", which builds magnetostatics into the basis.)

## 9. Possible paper shape

**Title ideas:** "Decision stumps are Maxwell fronts" / "Null stumps and the exact forest
representation of light."

**Abstract sketch:** A decision stump in the phase $z - ct$ is an exact weak solution of vacuum
Maxwell's equations, a one-wave Riemann problem. By linearity and a Stieltjes layer-cake identity,
every unidirectional plane wave of bounded variation is an expectation over such stumps, and every
finite forest is itself an exact field whose error lives only in the transported profile, uniformly
bounded via DKW. Geometrically, these are stumps whose split hyperplane in spacetime is null. We
discuss the 3D plane-wave extension, the link to Godunov's method, and how lattice dispersion breaks
exactness.

**Sections:** (1) 1D system and Riemann invariants; (2) stumps as exact fronts and jump conditions;
(3) the forest theorem plus the periodic zero-mean condition; (4) error bounds; (5) null-split geometry;
(6) 3D via translation representation; (7) dispersion on the LC ladder (numerical demo: launch a stump
on the ladder, show the wake); (8) discussion.

## 10. References to pull

- Godunov (1959), finite-difference method via Riemann problems; LeVeque, *Finite Volume Methods for Hyperbolic Problems* (Riemann problems, linear systems).
- Lax & Phillips, *Scattering Theory* (translation representation); Friedlander, radiation fields.
- Dvoretzky–Kiefer–Wolfowitz inequality (Massart's tight constant).
- Breiman (2001); Lin & Jeon (2006), random forests and adaptive nearest neighbors; Scornet (2016), random forests and kernel methods.
- Jackson, *Classical Electrodynamics* (jump/boundary conditions at moving surfaces).
- Bialynicki-Birula, Riemann–Silberstein vector (if the complex-field form from the Light paper comes back in).
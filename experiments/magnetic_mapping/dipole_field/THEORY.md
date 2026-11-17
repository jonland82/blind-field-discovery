# From a learned kernel to a magnetostatic field

Standalone follow-up note: [three-page PDF](../../../papers/02-kernel-to-magnetostatics/kernel_to_magnetostatics.pdf) · [LaTeX source and build instructions](../../../papers/02-kernel-to-magnetostatics/README.md). It matches the earlier *From samples to an equation* note.

**Question:** can the vector field learned from real magnetic measurements be reconstructed using magnetic-field theory, with similar shapes and useful predictions?

Yes, for these exploratory Aalto recordings: an equivalent dipole model distilled from the kernel improves held-out measured-field errors. This constructs a physically admissible approximation under stated assumptions. The actual room field and source distribution remain unknown.

## 1. Start with the learned field

The existing three-output Gaussian kernel predicts

$$
\widehat{\mathbf B}_K(x,y)=\overline{\mathbf B}+\sum_i\boldsymbol\alpha_i
\exp\left[-\frac{\|(x,y)-(x_i,y_i)\|^2}{2\ell^2}\right].
$$

Positions are in metres; field values retain the reference instrument's numerical units. We use all three components, rather than the magnitude-only one-dimensional profile from the previous experiment. Magnitude loses directional information needed to constrain a magnetic source model.

## 2. Supply the magnetostatic structure

In static, source-free air with constant permeability, the magnetic field satisfies

$$
\nabla\cdot\mathbf B=0,\qquad\nabla\times\mathbf B=0.
$$

Locally we may write $\mathbf B=-\nabla\Phi$, where $\nabla^2\Phi=0$. A useful family of such potentials is a uniform background plus point dipoles:

$$
\Phi(\mathbf r)=-\mathbf b_0\cdot\mathbf r+
\sum_{j=1}^{M}\frac{\mathbf q_j\cdot\mathbf R_j}{\|\mathbf R_j\|^3},
\qquad \mathbf R_j=\mathbf r-\mathbf s_j.
$$

Differentiating gives the reconstructed field explicitly:

$$
\boxed{\mathbf B_D(\mathbf r)=\mathbf b_0+
\sum_{j=1}^{M}\left[
\frac{3\mathbf R_j(\mathbf q_j\cdot\mathbf R_j)}{\|\mathbf R_j\|^5}
-\frac{\mathbf q_j}{\|\mathbf R_j\|^3}\right].}
$$

Here $\mathbf b_0$ is the background field; $\mathbf s_j$ is an assumed equivalent source position; and $\mathbf q_j$ is its fitted effective moment. With SI field calibration, $\mathbf q_j=\mu_0\mathbf m_j/(4\pi)$. We do not infer SI moments from unverified field units. Each term satisfies the two equations above away from its source. This is a three-dimensional field with a planar measurement restriction, not a claim that planar divergence equals full divergence.

For example, one vertical dipole below a horizontal line has

$$
B_z(x)=b_z+q\frac{2h^2-(x-a)^2}{[(x-a)^2+h^2]^{5/2}},\qquad
B_x(x)=b_x+3qh\frac{x-a}{[(x-a)^2+h^2]^{5/2}}.
$$

These theoretical shapes have peaks, troughs and sign changes. Superposition represents the more complicated room profile without forcing it into a single tanh curve.

The dipole formula and potential construction are standard theory: [NASA dipole field derivation](https://ccmc.gsfc.nasa.gov/static/files/Dipole.pdf); [Fitzpatrick, permanent ferromagnets](https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node60.html). The formula is exact for the chosen ideal dipoles; their use as equivalent sources is a modeling approximation.

## 3. Project the kernel onto this family

Set observation height to $z=0$ and put a regular grid of equivalent sources at $z=-h$. This is an assumed representation, not a measured floor depth. For fixed source positions the field is linear in the moments. Define the block matrix

$$
D_{ij}=\frac{3\mathbf R_{ij}\mathbf R_{ij}^{\mathsf T}}{\|\mathbf R_{ij}\|^5}
-\frac{I}{\|\mathbf R_{ij}\|^3}.
$$

Stack the kernel's predictions at training positions into a vector $\mathbf t$. Let $C$ subtract the per-component mean across positions, and let $S$ contain the column norms of $CD$. The implemented solution is

$$
A=CDS^{-1},\qquad
\mathbf w=(A^{\mathsf T}A+\lambda I)^{-1}A^{\mathsf T}C\mathbf t,
\qquad\mathbf q=S^{-1}\mathbf w.
$$

The background is the mean residual $\mathbf b_0=\operatorname{mean}_i[\widehat{\mathbf B}_K(\mathbf r_i)-(D\mathbf q)_i]$. Computation uses an SVD for numerical stability. Thus the workflow is **measured samples → Gaussian kernel → fitted theoretical dipole field → held-out measured samples**. No generated physical observations enter the experiment. Kernel targets are model outputs, not extra independent observations.

## 4. Frozen experimental protocol and results

- Data: [AaltoML magnetic-data](https://github.com/AaltoML/magnetic-data), hash-verified pinned archive. Original calibrated and orientation-corrected recordings.
- Reuse the original 2D protocol: Invensense trials 1–2 for fitting (500 total positions); trial 3 for model selection; trials 4–5 for stable evaluation; trials 6–9 for changed-environment stress. Preserve the excluded spatial rectangle and buffer. Other instruments use the existing trial-1-only affine alignment.
- Fit moments to kernel outputs at training positions only. Search grid sides 4, 7, 10; depths 0.25, 0.5, 1 m; ridge strengths 0.00001, 0.001, 0.1, 1. Choose minimum measured source-validation vector RMSE among these 36 candidates before opening test labels.
- Selected: **100 dipoles, assumed depth 1 m, ridge 0.001**. This is a 303-coefficient source representation, not a discovered one-term law. Validation error decreases from **8.63 to 6.18**.

Stable held-out vector RMSE (lower is better):

| Instrument | Original kernel | Kernel → dipoles | Same dipoles fitted directly to measurements |
| --- | ---: | ---: | ---: |
| Invensense | 8.09 | **5.22** | 5.31 |
| Nexus phone | 10.34 | **8.83** | 8.85 |
| Trivisio | 8.03 | **5.56** | 5.71 |

On Invensense, the reconstructed component shapes correlate **0.938, 0.908, 0.958** with held-out measurements. Error falls by about **35%** relative to the original kernel. The direct-data control performs almost as well: this supports the usefulness of the supplied physical representation, but does not show that the kernel stage is necessary. Its settings were selected for the distilled model; it is a matched control, not an independently optimized competitor.

[Plot: all three components, full first test trial and a fixed spatial cut](comparison.png). The cut compares two model curves; no measured true curve is available exactly along it. [All results](RESULTS.md), [coefficients, provenance and diagnostics](results.json), and `predictions.npz` retain stable and stress cases, including large changed-environment phone errors.

## 5. What follows, and what does not

We now have a theoretical magnetic-field function inferred through a real-data kernel, with demonstrably similar measured shapes and better predictive accuracy in this experiment. Theory acts as a useful constraint on interpolation. The previous failed scalar autonomous ODE extraction did not rule out this richer vector representation.

This is an inverse approximation, not a unique derivation of the actual room's sources. Limited planar data cannot establish the true three-dimensional continuation, source depths or currents. The selected depth is at the largest searched value and source count at the largest searched grid; neither should be interpreted as an estimated physical property or an established optimum. Sources below the plane are equivalent mathematical sources, not located objects. Maxwell structure was supplied; measured accuracy was tested. Test acquisitions were examined in earlier work, so this is exploratory reuse rather than a fresh confirmation.

The next useful checks are source-depth and regularization sensitivity, stability under alternate training trials, and replication on independent public data with three-dimensional positions. These would establish how robust the reconstructed shape is beyond this selected model.

Reproduce: `python experiments/magnetic_mapping/run_dipole_field.py`. Verify: `python -m pytest experiments/magnetic_mapping/tests/test_dipole_field.py -q`.

# Blind field-equation discovery: real-data limits and a synthetic positive control

16 November 2026

For the derivations, fitted equations, and an assessment of how far the results reach toward Maxwell, see [the mathematical walkthrough](../../docs/MATHEMATICS.md).

**Simple workflow verification:** [the kernel demonstration](../kernel_benchmark/README.md) starts from a hidden magnetic-like function, learns only its samples, recovers a derivative equation, and constructs an integral-kernel version. [Results](../kernel_benchmark/results/RESULTS.md) compare with known synthetic truth, with and without noise.

**Real-measurement follow-up:** [the public-data experiments](../magnetic_mapping/README.md) compare learned maps against explicit physical descriptions using public phone/magnetometer recordings and an independent optical-pose dataset. The [phone pilot](../magnetic_mapping/results/RESULTS.md) and [optical-pose results](../magnetic_mapping/public_transfer/RESULTS.md) are separate from the synthetic recovery below. The active research path requires no new physical recordings.

**Kernel-to-theory experiment:** [the radio response study](../radio_response/README.md) documents Maxwell's Green-function form and tests learned response kernels on public radio measurements. See [the theory and experiment](../radio_response/THEORY_AND_EXPERIMENT.md) and [measured results](../radio_response/results/RESULTS.md).

## What was actually achieved

A generic sparse relation learner recovered all eight scalar components of Maxwell's equations and the charge-continuity equation from a separate, explicitly synthetic control dataset. It was not supplied divergence/curl operators, Maxwell constraints, dipole models, the number of laws, or the true coefficient values. This is algorithm validation, not an experimental rediscovery of electromagnetism from phone data.

The real ADVIO-12 phone recording cannot identify the full Maxwell system: it has no electric-field, charge-density, or current-density observations, and its position samples are almost planar. We did not manufacture missing observables from a pretrained physics model.

## Real-data diagnostic

Principal-component fractions of the measured route position variance:

1. 0.8500292715
2. 0.1499401807
3. 0.0000305478

Thus 99.996945% of variance is within a best-fit plane. RMS displacement normal to that plane is 0.1047 m, compared with 17.4682 m along the largest principal direction. This is a data-coverage diagnostic, not a proof that every spatial derivative is exactly unobservable. A single moving route still does not supply a dense 3D field survey or separate spatial and temporal variation without further assumptions.

## Learner

Ten unnamed scalar channels u0 through u9 were provided, with coordinates (x,y,z,t). The candidate library contains a constant, 10 raw channels, 40 first coordinate derivatives, and 55 raw quadratic products: 106 terms. No precombined electromagnetic operators appear in this library.

The learner:

1. Scales candidate columns by training RMS.
2. Uses singular-value decomposition to locate approximate linear dependencies, selecting nullity from the largest singular-value gap if that gap exceeds 10.
3. Searches that subspace for sparse relations by L1 linear programming, anchoring every candidate coefficient in turn. In symbols, minimize ||N alpha||_1 subject to (N alpha)_j=1, where N contains approximate null vectors.
4. Removes coefficients smaller than 2.5% of the largest standardized coefficient, refits on the selected support, and retains relations with validation relative residual below 2%.
5. Deduplicates identical supports and evaluates on separate held-out experiments.

The zero-coefficient solution is excluded by the anchoring constraint. All terms are treated generically. Coordinate derivatives, polynomial features, smoothness needed for finite differences, and sparsity are mathematical inductive biases: the learner does not literally start without any assumptions.

## Synthetic generator (not hidden from the reader)

The generator deliberately encodes electromagnetic physics. Smooth random scalar/vector potentials are constructed from 24 Fourier modes with arbitrary wave vectors and temporal frequencies, without a vacuum dispersion constraint. In dimensionless units epsilon=mu=1:

E = -grad(phi) - partial_t A

B = curl A

rho = div E

J = curl B - partial_t E

The learner receives channel arrays rather than these generator definitions. Channel interpretation is revealed only after the equations have been selected. Random potential amplitudes and frequencies vary between experiments. This is a favorable, synthetic benchmark with access to all observables and local samples in three space dimensions and time.

Derivative features are computed by central differences with step 0.02, from independently perturbed pointwise field samples. Pointwise noise is 0.0001 times each channel's RMS (0.01%). Analytic derivative values are not supplied to the learner. There are 12 training experiments (6,144 center points), 4 validation experiments, and 4 test experiments (2,048 center points each). Stencil samples are additional observations used for numerical derivatives. All test experiments use new random seeds and source configurations.

The synthetic units make expected coefficients one. Recovering them does not independently determine dimensional epsilon_0, mu_0, or the speed of light.

## Discovered relations and controls

After revealing the channel meanings, the recovered relations match:

div E = rho

div B = 0

partial_t B = -curl E

partial_t E = curl B - J

partial_t rho + div J = 0

All supports match these relations. Coefficient deviations from their generating values are below approximately 0.06%; test relative cancellation residuals range from about 0.10% to 0.20%. The residual denominator is the root mean square of the sum of the squared individual equation terms, before cancellation. The additional continuity relation is a consequence of Maxwell's equations, not a new independent physical law.

Two controls use exactly the same learner:

- Only the B channels, but still sampled throughout a synthetic 3D volume: one identity is recovered, div B approximately zero. Full electromagnetic dynamics are not recovered from B alone.
- Candidate columns independently shuffled: no accepted identities are recovered. This negative control retains marginal feature distributions while destroying their joint alignment; it is not a physically realizable field experiment.

See results.json for every unrounded coefficient, term index, residual, and diagnostic.

## Why the phone data cannot settle the spatial laws

Even perfect samples on the plane z=0 cannot distinguish a field B from B + (0,0,lambda*z): the added field vanishes on that plane but changes divergence by lambda. Likewise B + (lambda*z,0,0) agrees on the plane while changing curl by (0,lambda,0). These are mathematical alternatives used to demonstrate ambiguity; their exclusion would require assuming physical laws rather than discovering them.

The actual route is not exactly planar, and its full field varies along a curve. The planar construction illustrates why sparse off-plane coverage and sensor error make spatial-derivative discovery fragile. Separately, the absence of measured E, rho and J prevents identification of the complete Maxwell system without additional assumptions or instruments.

## Reproduce

Python dependencies: numpy, scipy; matplotlib for the figure.

Place the upstream ADVIO pose CSV at `data/advio-12/iphone/arkit.csv`. From the repository root, run:

```
python experiments/maxwell_control/run_discovery.py
python experiments/maxwell_control/plot_results.py
```

The script regenerates the synthetic dataset, executes the learner and controls, and overwrites its results. It makes no network requests. Original real-data positions and PCA outputs are also provided in real_route_geometry.npz. The synthetic_control.npz file contains candidate matrices and feature names; the generator source documents the complete process.

## Evidence and scope

- Generic sparse equation discovery: Brunton, Proctor, Kutz (2016), https://doi.org/10.1073/pnas.1517384113 .
- Spatiotemporal PDE discovery: Rudy et al. (2017), https://doi.org/10.1126/sciadv.1602614 .
- Real-data source: Cortés, Solin, Rahtu, Kannala (2018), ADVIO: An Authentic Dataset for Visual-Inertial Odometry, ECCV. https://zenodo.org/records/1476931 and https://github.com/AaltoVision/ADVIO .
- The included ADVIO pose data and derived route geometry are subject to CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ . Attribute the authors; that licence does not permit commercial use.

No claim of novelty, unknown physics, or successful Maxwell discovery from the phone recording is made.

# Hidden-wire discovery: protocol fixed before the first run

Question: can the sensor workflow (observations -> Gaussian kernel -> explicit
field -> new observations) recover a wire's magnetic field when the explicit
representation is selected from general algebra rather than supplied physics?

This is a synthetic workflow experiment, not a historical reconstruction or a
new measurement of electromagnetism. The generator is isolated in `world.py`;
the learner imports neither that module nor any magnetic response formula.

## Observations and fixed cases

- Inputs: Cartesian x, y in metres and signed current I in amperes. Outputs:
  three Cartesian magnetic components in microtesla. No radius, angle, field
  derivatives, wire formula, permeability or symmetry features enter learning.
- Hidden world: infinite straight wire on the z axis. Generator and scorer use
  B = 0.2 I (-y, x, 0)/(x^2+y^2) microtesla. The idealization is deliberate.
- Main observed domain: radius 0.25--2 m, all angles, |I| = 0.5--4 A.
  Radius is sampled uniformly, not area-uniformly. Current signs are balanced
  in expectation. Training and validation acquisitions are independent draws.
- Five cases: noiseless; 1% noise; 5% noise; 1% noise with only 80 training
  observations; 1% noise with angles restricted to +/- pi/6. Other cases have
  480 training observations. Every case has 180 validation observations.
- Noise is independent Gaussian per component with standard deviation equal
  to the stated fraction of the clean training field's pooled component RMS.
  This simulator noise scale is not passed to the learner or model selector.
- Three fixed seeds: 20261116, 20261117, 20261118. Each case is fit independently.

## Learning and extraction

1. Gaussian kernel ridge regression in standardized (x,y,I), with separate
   output coefficients and a constant mean. Select scalar length in
   {0.35,0.6,1,1.8,3} and ridge in {1e-8,1e-6,1e-4,0.01,0.1} by measured
   validation vector MSE. Standardization uses training coordinates only.
2. Query this frozen teacher at the actual training inputs, matching the
   real-data dipole experiment. These are model targets, not new observations.
3. Search general component polynomials of total degree 0--4 and shared-
   denominator rational functions P_j(x,y,I)/Q(x,y,I), with numerator and
   denominator total degrees 1 or 2. All monomials up to that degree are
   included; no inverse distance or magnetic response is a supplied feature.
4. Rational initialization: smallest right singular vector of the normalized
   homogeneous system B_j Q - P_j = 0. Prune normalized coefficients at
   thresholds {0,0.003,0.01,0.03,0.1,0.2}; refit the surviving homogeneous
   system for at most eight iterations. Fix the largest denominator
   coefficient to one, then refine surviving coefficients by nonlinear least
   squares on field values. This is a finite, explicit search grammar, not
   unrestricted symbolic discovery. No manually rounded equation is scored.
5. Select the fewest-coefficient model within 5% of the minimum measured
   validation MSE (plus 1e-10 times training output variance). Break ties by
   validation MSE. Independently run the same entire expression search on raw
   training observations as the direct-fit control.
6. Freeze all selected models before opening any test labels. Test labels and
   oracle derivatives never enter fitting or model selection. Validation is
   reused across kernel and expression selection, as in the sensor study;
   independent tests are therefore essential.

## Fixed evaluations

800 independent points per test: same-domain conditions; full-angle positions;
unseen inward radii 0.10--0.22 m; unseen outward radii 2.3--4 m; unseen currents
|I|=5--8 A. Each test compares constant, kernel, distilled expression and
direct expression against the clean oracle and noisy observations. A separate
stress test moves the wire to (0.35,-0.25) m without retraining and excludes
points within 0.15 m of its new axis. This is an unobserved environmental
change, so failure is expected and does not refute a conditional field law.

Also score analytic model derivatives against hidden derivatives, planar
divergence/curl away from the axis, and circulation at several radii. These
are post-fit diagnostics, not independent experimental measurements. A
nonmagnetic polynomial world is a negative control: the same search should
not force a wire formula. Tests also check a displaced wire and block access
to the generator during fitting.

## Comparison with the measured sensor workflow

Use the retained Aalto predictions and dipole coefficients, without refitting
or revisiting their model selection. Compare available inputs, sample sizes,
assumptions, kernel fit, teacher-to-expression discrepancy, held-out errors,
direct-fit control and environment change. Use vector RMSE divided by clean
field RMS for simulation; divide by observed vector RMS for Aalto and label
that distinction. The datasets, noise and geometry are different; those
normalized scores are descriptive and do not rank the experiments' difficulty.

All cases and all three seeds must be reported, including failures. Any later
protocol change must be documented instead of silently replacing this plan.

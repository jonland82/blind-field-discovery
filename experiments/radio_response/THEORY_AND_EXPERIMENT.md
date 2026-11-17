# From a learned similarity kernel to an electromagnetic response kernel

This document connects our learned predictor to the integral form of Maxwell's equations and specifies a public-data radio experiment. It distinguishes the response we would ideally identify from the response the available instruments actually determine.

**Experiment performed:** learn a relative radio response from real DICHASUS measurements and predict other antennas' responses in a held-out acquisition. Compare a generic learner, a limited Maxwell-based geometric description, and their combination. [Numerical results](results/RESULTS.md).

**Scope:** the experiment identifies predictive relationships between measured antenna responses. It does not reconstruct a unique fundamental electromagnetic theory, estimate light speed, or establish an absolute causal field Green function.

## 1. Quantities and units

| Symbol | Meaning |
| --- | --- |
| $\mathbf x,t$ | Position and time at which a field is evaluated |
| $\mathbf x',t'$ | Position and time of a source contribution |
| $r$ | Source-to-observation distance, metres |
| $\rho,\mathbf J$ | Charge density and current density |
| $\phi,\mathbf A$ | Scalar and vector electromagnetic potentials |
| $\mathbf E,\mathbf B$ | Electric and magnetic fields |
| $c,\epsilon_0,\mu_0$ | Propagation speed and vacuum electromagnetic constants |
| $G_{\rm ret}$ | Retarded field-propagation Green function |
| $s,y_a$ | A transmitted signal and the signal measured at receiver port $a$ |
| $H_a(f;\mathbf x)$ | Measured complex channel response at port $a$ for transmitter position $\mathbf x$ |
| $Q_{ar}=H_a/H_r$ | Relative response from reference receiver port $r$ to another receiver port $a$ |
| $k_\ell(\mathbf x,\mathbf x_i)$ | Statistical similarity between two transmitter positions |
| $\ell,\lambda$ | Regression length scale in metres and regularization parameter |

The subscript $r$ on $H_r$ denotes a reference port; the distance $r$ in the Maxwell formulas is a separate scalar. The port-level ratios are dimensionless. The DICHASUS CSI values include the instrument chain; they are not calibrated samples of all components of $\mathbf E$ and $\mathbf B$.

## 2. Maxwell as a weighted integral over source history

In unbounded free space, introduce the wave operator

$$
L=\frac{1}{c^2}\partial_t^2-\nabla^2.
$$

Its retarded Green function is

$$
G_{\rm ret}(\mathbf x,t;\mathbf x',t')
=\frac{\delta(t-t'-r/c)}{4\pi r},
\qquad r=\|\mathbf x-\mathbf x'\|,
$$

which satisfies, as a distribution,

$$
L G_{\rm ret}=\delta^{(3)}(\mathbf x-\mathbf x')\delta(t-t').
$$

The kernel combines a distance dependence with a travel-time condition. A contribution reaches the observation event after a delay $r/c$. In three-dimensional free space the displayed impulse is supported on the past light cone; it is not an arbitrary smooth weighting of every past event.

For charge-conserving sources, the Lorenz-gauge potentials have the source-driven representation

$$
\begin{pmatrix}\phi\\\mathbf A\end{pmatrix}(\mathbf x,t)
=\begin{pmatrix}\phi_{\rm free}\\\mathbf A_{\rm free}\end{pmatrix}(\mathbf x,t)
+\int_{-\infty}^{t}dt'\int_{\mathbb R^3}d^3x'\,
G_{\rm ret}(\mathbf x,t;\mathbf x',t')
\begin{pmatrix}\rho(\mathbf x',t')/\epsilon_0\\\mu_0\mathbf J(\mathbf x',t')\end{pmatrix}.
$$

Here the free terms account for independent source-free fields consistent with the chosen initial/boundary conditions. The source continuity condition is $\partial_t\rho+\nabla\cdot\mathbf J=0$. The physical fields follow from

$$
\mathbf E=-\nabla\phi-\partial_t\mathbf A,
\qquad\mathbf B=\nabla\times\mathbf A.
$$

This is a solution representation of Maxwell with specified conditions, not a statistical fit to training examples. The kernel need not be a symmetric, positive-definite similarity function. For a derivation, see [Tong's electromagnetic radiation notes, Section 6](https://scipp.ucsc.edu/~haber/ph214/Tong--electromagnetism.pdf). The formulas here are written in SI units.

## 3. From field propagation to the response of an experiment

An antenna measures a projection of a field through an instrument response. Materials, boundaries, antenna patterns, and electronics contribute to what the recorder reports. Under an approximately linear, time-invariant description of one stationary source/receiver configuration,

$$
y_a(t;\mathbf x)
=\int h_a(\mathbf x,\tau)\,s(t-\tau)\,d\tau+\eta_a(t).
$$

In frequency space,

$$
Y_a(f;\mathbf x)=H_a(f;\mathbf x)S(f)+N_a(f).
$$

The experimental kernel $h_a$ includes propagation, antennas, and measurement hardware. Even if Maxwell accurately describes the fields, $h_a$ is generally more complicated than a single delayed free-space impulse. In particular, reflections can produce multiple arrivals.

A full source-to-receiver kernel experiment would need a calibrated transmitter time/phase reference and a known sounding waveform. Our chosen public dataset supplies estimated complex radio channel coefficients and geometry, rather than independent raw source currents or all underlying time-domain waveforms.

## 4. What the public data can identify

The source is [DICHASUS dcxx, DaRUS version 1.0](https://doi.org/10.18419/DARUS-3831), measured by Euchner, Stephan, and Gauger (2023). It provides a robot-mounted transmitter's positions and complex frequency responses at 64 receiver channels in two arrays. The nominal carrier is 3.438 GHz and the bandwidth is 50 MHz. Geometry, channel assignments, and source warnings are retained in `data/dichasus/spec.json` under the repository root.

The instrument architecture has common transmitter/receiver clock and phase ambiguity. In the schematic model

$$
\widetilde H_a(f,n)
=a_n(f)\,d_a(f)\,H_a^{\rm physical}(f;\mathbf x_n),
$$

the factor $a_n(f)$ is common to receiver channels in a snapshot and may include a phase and delay offset. The factor $d_a(f)$ describes a fixed channel-specific instrument response. Without external timing information, a common shift in all arrival times is indistinguishable from a change to the transmitter clock. The provider's [dataset revision discussion](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/rev2-datasets/) explains why global timing offsets are not physical source-to-receiver travel times.

We therefore predict the identifiable **relative response** within each array:

$$
Q_{ar}(f;\mathbf x)
=\frac{\widetilde H_a(f;\mathbf x)}{\widetilde H_r(f;\mathbf x)}
=\frac{d_a(f)}{d_r(f)}
\frac{H_a^{\rm physical}(f;\mathbf x)}{H_r^{\rm physical}(f;\mathbf x)}.
$$

The common nuisance factor cancels. The quotient still contains a fixed relative instrument response, which models may learn from training data. The same cancellation applies to the unknown common transmitted probe in an ideal noiseless linear measurement. Noise and a weak reference channel make division unstable; the quality rules below address that practical limitation.

This yields a functional prediction on real measured responses:

$$
\widehat{\widetilde H}_a(f;\mathbf x)
=\widehat Q_{ar}(f;\mathbf x)\,\widetilde H_r(f;\mathbf x).
$$

At test time we provide the transmitter position and the measured reference-antenna response. Other antennas' measured responses remain targets. This is a conditional response-transfer task, not prediction of every antenna's signal from geometry alone.

The spec warns about cross-array offset accuracy, so a reference and its targets are always in the same array. The experiment does not assume that arbitrary phase comparisons between the two arrays are calibrated.

## 5. Two different meanings of kernel

The experiment involves both a learned response kernel and a regression similarity kernel:

1. $Q_{ar}$ is the physical/instrumental response we are trying to predict.
2. $k_\ell$ specifies how our regression model shares information between nearby source positions.

They play different mathematical roles. The Gaussian similarity is not itself asserted to be Maxwell's Green function.

For every receiver/frequency output, the unconstrained learner has the form

$$
\widehat Q_{ar}(f;\mathbf x)
=\overline Q_{ar}(f)+\sum_{i=1}^{n}k_\ell(\mathbf x,\mathbf x_i)W_{i,ar}(f),
$$

$$
k_\ell(\mathbf x,\mathbf x_i)
=\exp\left(-\frac{\|\mathbf x-\mathbf x_i\|^2}{2\ell^2}\right).
$$

Thus the **learned response** is an output of kernel regression. It is complex-valued and frequency-dependent, with no required propagation law or time-delay sign.

Training observations determine the weights through

$$
W=(K+\lambda D_w^{-1})^{-1}(Z-\mathbf1\overline Q),
$$

where $Z$ stacks the measured quotients and $K_{ij}=k_\ell(\mathbf x_i,\mathbf x_j)$. The mean $\overline Q$ is a training-only weighted mean. $D_w$ contains sample weights proportional to the reference's mean spectral power, normalized and clipped to $[0.05,20]$. This reduces the influence of noisy quotients without fitting a new weight from test errors. A small diagonal numerical jitter is added in the implementation.

The candidate length scales are 0.25, 0.5, 1, 2, 4, and 8 m, and ridge values are 0.001, 0.01, and 0.1. Validation chooses a combination separately for each array.

## 6. The Maxwell-based comparison

Use a scalar direct-path approximation, with Fourier convention $H(f)=\int h(\tau)e^{-2\pi i f\tau}d\tau$:

$$
G_a(f;\mathbf x)\propto\frac{e^{-2\pi i f r_a/c}}{r_a},
\qquad r_a=\|\mathbf x-\mathbf r_a\|.
$$

Here $\mathbf r_a$ is receiver port $a$'s position. Taking a ratio gives

$$
P_{ar}(f;\mathbf x)
=\frac{G_a}{G_r}
=\frac{r_r}{r_a}\exp\left(-\frac{2\pi i f(r_a-r_r)}{c}\right).
$$

The **geometric_green** model is

$$
\widehat Q_{ar}^{\rm green}(f;\mathbf x)
=\gamma_{ar}(f)P_{ar}(f;\mathbf x).
$$

The constant complex coefficient $\gamma_{ar}(f)$ is fitted to training measurements by minimizing

$$
\sum_{i\in\mathrm{train}}
\left|\widetilde H_{a,i}(f)
-\gamma_{ar}(f)P_{ar}(f;\mathbf x_i)\widetilde H_{r,i}(f)\right|^2.
$$

It absorbs a fixed receiver/frequency response, not a position-dependent correction. The propagation speed is supplied as $c=299792458$ m/s. Recovering that number is not part of this experiment.

Antenna element positions are reconstructed from the published array centers, direction vectors, spacings, and assignment grids. The specification does not fully resolve every row/column and complex-IQ sign convention. Eight discrete axis/phase choices are therefore fitted on training data and selected on validation data. They remain frozen for tests. This finite convention calibration is disclosed rather than interpreted as discovered physics.

The model is a limited Maxwell-based description of a direct path. It does not solve the complete hallway boundary-value problem or all antenna/electronics behavior. Reflections, diffraction, scattering, antenna patterns and coupling can all matter.

## 7. A combined representation

The **green_envelope** model factors out the geometric response and learns what remains:

$$
D_{ar}(f;\mathbf x_i)=\frac{Q_{ar}(f;\mathbf x_i)}{P_{ar}(f;\mathbf x_i)},
$$

$$
\widehat Q_{ar}^{\rm hybrid}(f;\mathbf x)
=P_{ar}(f;\mathbf x)\left[
\overline D_{ar}(f)+\sum_i k_\ell(\mathbf x,\mathbf x_i)V_{i,ar}(f)
\right].
$$

It uses the same spatial regression family and grid as the unconstrained learner. The propagation factor comes from the geometric model's validation-selected convention. Validation then selects the envelope's regression settings. No held-out acquisition labels enter either step.

This is the central representation experiment: **does factoring out a physical propagation rule make the remaining measured relationship easier to learn and transfer?** It tests one concrete ML-to-theory bridge rather than demanding that a learner reproduce named equations.

## 8. Acquisition split and quality controls

| Public file | Role | Fixed downloaded snapshot count |
| --- | --- | ---: |
| dc01 | Training, unchanged hallway | 256 |
| dc02 | Validation, unchanged hallway | 128 |
| dc04 | Held-out acquisition, unchanged hallway | 256 |
| dc13 | Separate frozen-model stress test with a metal obstruction | 128 |

These are samples of much larger acquisitions. For each file, the downloader selects evenly spaced original record indices across the complete acquisition. It fetches those full records by validated HTTP byte ranges rather than downloading hundreds of gigabytes. The fixed-length schema is checked at every selected record, including CRC32C. Individual downloaded ranges receive SHA-256 hashes. The source full-file MD5 values are retained but cannot be verified from a partial download.

A two-metre-radius disk around the origin in the published horizontal coordinates is held out spatially. Training and validation exclude the disk plus a 0.5 m buffer. The main test reports all accepted positions and positions inside the disk separately. These are new positions in the same environment, not a new laboratory.

The reference antenna is chosen independently for each array using the highest median SNR in training observations outside the spatial exclusion. Training-only per-bin reference RMS sets an amplitude floor at 2% of that RMS. Accepted snapshots must have finite relevant values, positive supplied reference SNR, absolute tachymeter interpolation age at most 0.1 s, and reference magnitude above the floor in every selected frequency bin. The implementation reports counts and original indices. It does not assign unverified units to the SNR statistic or remove observations based on prediction error.

The comparison also includes a zero filter, a best constant coupling per receiver/frequency, and a learner whose measured quotient vectors have been shuffled across training positions. The shuffled learner uses the same selection grid. Each array's models are scored on exactly the same accepted test observations.

The primary error is

$$
\mathrm{NMSE}
=\frac{\sum_{i,a,f}\left|\widehat Q_{ar}(f;\mathbf x_i)\widetilde H_{r,i}(f)
-\widetilde H_{a,i}(f)\right|^2}
{\sum_{i,a,f}|\widetilde H_{a,i}(f)|^2}.
$$

It evaluates actual complex measured responses, not invented source waveforms. An NMSE of one is the zero-output baseline; zero is perfect. The report also preserves quotient errors, median snapshot errors, per-receiver errors, and nearest-training-position distances. The evaluation ports are the same physical ports used in training, so no new-device generalization is claimed.

## 9. What the impulse-response plot means

The model uses bins $0,16,32,\ldots,1008$ of each measured 1024-bin response. Native frequency order is interpreted using the [authors' processing code](https://github.com/NVlabs/diff-rt-calibration/blob/main/code/utils.py), which applies `fftshift` to obtain centered frequency order. The selected 64-bin responses can be transformed to periodic relative-delay filters by an IFFT.

The nominal bin spacing is $50\,\mathrm{MHz}/64$, giving a 1.28 microsecond periodic delay window and 20 ns spacing between displayed delay samples. Neither is a claim about independent physical delay resolution beyond the measurement bandwidth, hardware and processing limits.

The response from one receiver to another can have negative relative delay: the reference receiver may see a contribution later than the target receiver. Division by a reference channel can also introduce inverse-filter structure. A periodic finite-band IFFT may spread an impulse across both delay signs. Its plot is therefore **not an absolute-causality or speed-of-light test**.

For transparency, the saved example always uses the first accepted dc04 snapshot and first output port for each array. It is not selected for good agreement. All scored predictions remain available separately.

## 10. Interpretation and remaining gap

- A useful learned filter demonstrates predictive regularities in an experimentally measured response operator.
- A useful geometric model demonstrates that its supplied propagation structure explains some of those observations after fixed instrument calibration.
- A benefit from the learned envelope supports the usefulness of combining that representation with empirical corrections.
- Failure under the metal-obstruction acquisition demonstrates limited transfer when an unmodeled environmental variable changes.

None of these establishes that one representation is uniquely fundamental. In particular, a direct-path model's failure is not a failure of every Maxwell-based solution, and a kernel learner's success is not recovery of the electromagnetic ontology.

The remaining gap to the original causal-source question is specific: independently calibrated source history, absolute timing, and a sufficiently specified measurement/boundary model are needed to compare a learned absolute response with $G_{\rm ret}$ itself. Those requirements can be sought in other **public datasets**; this project does not require new physical measurements from the user.

## Reproduce

From the workspace root:

```powershell
python -m pip install -r experiments/radio_response/requirements.txt
python experiments/radio_response/fetch_data.py
python experiments/radio_response/run_experiment.py
python -m pytest experiments/radio_response/tests -q
```

Data labels are measured CSI only. Artificial numerical arrays used in software tests are temporary fixtures and are excluded from all experimental results. Earlier synthetic, phone and optical-pose outputs remain separate.

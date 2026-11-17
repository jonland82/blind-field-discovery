# Learning a measured radio response kernel

**Completed public-data experiment:** train a complex response kernel on one DICHASUS acquisition, choose settings on a second, and predict another acquisition plus a changed-environment stress test. No new recordings or synthetic labels are used.

- [Theory, definitions, equations and experimental protocol](THEORY_AND_EXPERIMENT.md)
- [Results and limitations](results/RESULTS.md)
- [Comparison figure](results/comparison.png)
- [Measured/predicted relative impulse responses](results/relative_impulse_responses.png)

The key comparison is whether factoring out a free-space propagation rule makes the measured response easier to learn. It does help on the main held-out acquisition, with modest gains over a constant response filter. The combined model has the lowest error in the north array; the geometric model has the lowest error in the south array. Residual errors remain large, and all fitted models fail to beat zero output in the changed environment.

The data's clocks do not determine an absolute transmitter-to-receiver delay. The identifiable target is therefore a **relative antenna response**, conditional on another antenna's measured signal. Its delay-domain representation can include negative relative delays. This experiment does not measure light speed or absolute causality, and its direct-path comparator is not a full Maxwell solution for the hallway.

From the workspace root:

```powershell
python -m pip install -r experiments/radio_response/requirements.txt
python experiments/radio_response/fetch_data.py
python experiments/radio_response/run_experiment.py
python -m pytest experiments/radio_response/tests -q
```

The downloader fetches 768 complete selected records, approximately 403 MB, from files that are individually several gigabytes. Exact byte ranges, record indices, CRC32C checks, and hashes are recorded. Reruns reuse verified local records. The analyzer does not access the network. Results are written under `experiments/radio_response/results`, independently of the earlier studies.

Data: Euchner, Stephan, and Gauger (2023), [DICHASUS dcxx](https://doi.org/10.18419/DARUS-3831), version 1.0, CC BY 4.0. Source metadata and warnings accompany the download. Both the learned and geometric models are evaluated against measured complex CSI, not ray-traced or analytically generated targets.

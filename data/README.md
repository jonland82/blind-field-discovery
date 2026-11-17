# Public data and local caches

Raw datasets live here locally and are excluded from Git. Reproduction scripts fetch pinned or versioned public sources and verify their hashes. Tracked [manifests](manifests/) preserve the download provenance; the result JSON files also retain their source and protocol metadata.

| Local directory | Source | Fetch from the repository root | Attribution and terms |
| --- | --- | --- | --- |
| `aalto_magnetic/` | [AaltoML magnetic-data](https://github.com/AaltoML/magnetic-data), commit `b45da3bc7ab623fdef6e01247021daebf7983f29` | `python experiments/magnetic_mapping/fetch_data.py` | Solin et al. (2018), [paper](https://doi.org/10.1109/TRO.2018.2830326). Upstream calibration and orientation correction retained. No explicit license file in the pinned archive; no redistribution license is inferred. |
| `fiu_marg/` | [FIUMARGDB](https://github.com/LABDSP/FIUMARGDB_marg_signals_and_reference_orientations), commit `998c9ebbd243b3e13ba2dc8bbf94e958b6be1b8b` | `python experiments/magnetic_mapping/fetch_fiu.py` | Sonchan et al. (2023), [paper](https://doi.org/10.3390/s23083786). Retain the upstream GPL v3 LICENSE and attribution. |
| `dichasus/` | [DICHASUS dcxx](https://doi.org/10.18419/DARUS-3831), version 1.0 | `python experiments/radio_response/fetch_data.py` | Euchner, Stephan and Gauger (2023), CC BY 4.0. Downloader fetches bounded, verified records, approximately 403 MB. |
| `advio-12/` | [ADVIO](https://github.com/AaltoVision/ADVIO), sequence 12 | Obtain `iphone/arkit.csv` from the [upstream archive](https://zenodo.org/records/1476931). | Cortés, Solin, Rahtu and Kannala (2018), CC BY-NC 4.0. Used only for the earlier route-geometry diagnostic. |

The papers rebuild from retained result arrays without these caches. Raw data are needed to rerun the measured experiments. No new phone recordings are required.

The large derived radio array `experiments/radio_response/results/predictions.npz` is also kept locally and ignored. Regenerate it with `python experiments/radio_response/run_experiment.py` after fetching DICHASUS. Reports, figures and the small impulse-response examples are versioned.

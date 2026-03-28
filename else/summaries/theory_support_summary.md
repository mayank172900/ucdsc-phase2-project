# Theory Support Update

## Status

The strengthening pass is complete.

This update adds two things beyond the earlier Phase 3 and Phase 4 package:

- embedding-based anisotropy diagnostics on the frozen `BloodMNIST` and `DermaMNIST` runs
- a stronger low-dimension `BloodMNIST` experiment using `feat_dim = 32` across all `5` splits, compared against the existing full baseline run at the default high dimension

## New Code

- `/Users/goodday/Downloads/pap/paper_conf_exp/core/experiment_export.py`
- `/Users/goodday/Downloads/pap/paper_conf_exp/core/test.py`
- `/Users/goodday/Downloads/pap/paper_conf_exp/scripts/anisotropy_diagnostic.py`
- `/Users/goodday/Downloads/pap/paper_conf_exp/scripts/accepted_unknown_alignment.py`
- `/Users/goodday/Downloads/pap/paper_conf_exp/scripts/aggregate_dim_sweep_full.py`
- `/Users/goodday/Downloads/pap/paper_conf_exp/scripts/delta_cap_estimator.py`

## New Outputs

- Low-dimension BloodMNIST summary:
  - `/Users/goodday/Downloads/pap/blood_dim32_summary.md`
  - `/Users/goodday/Downloads/pap/blood_dim32_summary.csv`
  - `/Users/goodday/Downloads/pap/blood_dim32_per_split.csv`
- Anisotropy diagnostic:
  - `/Users/goodday/Downloads/pap/anisotropy_summary.md`
  - `/Users/goodday/Downloads/pap/anisotropy_summary.csv`
  - `/Users/goodday/Downloads/pap/anisotropy_summary.png`
- Accepted-unknown alignment check:
  - `/Users/goodday/Downloads/pap/accepted_unknown_alignment.md`
  - `/Users/goodday/Downloads/pap/accepted_unknown_alignment.csv`
- Paper-style `delta_cap` estimator:
  - `/Users/goodday/Downloads/pap/delta_cap_summary.md`
  - `/Users/goodday/Downloads/pap/delta_cap_dataset_summary.csv`
  - `/Users/goodday/Downloads/pap/delta_cap_split_summary.csv`
  - `/Users/goodday/Downloads/pap/delta_cap_summary.png`

## Dimension Evidence

### BloodMNIST low dimension (`feat_dim = 32`, `100` epochs, `5` splits)

- `ACC 98.633 +- 0.606`
- `AUROC 78.108 +- 12.801`
- `OSCR 77.839 +- 12.824`
- `TNR 37.083 +- 25.597`
- `DTACC 75.831 +- 8.164`

Source:

- `/Users/goodday/Downloads/pap/blood_dim32_summary.md`

### BloodMNIST frozen baseline (default high dimension, `400` epochs, `5` splits)

- `ACC 98.242 +- 0.662`
- `AUROC 81.122 +- 8.230`
- `OSCR 80.544 +- 8.193`
- `TNR 34.601 +- 18.037`
- `DTACC 76.820 +- 5.796`

Source:

- `/Users/goodday/Downloads/pap/baseline_summary.md`

### Interpretation

- The low-dimension run is not catastrophic, but it is weaker on the ranking-style open-set metrics:
  - `AUROC`: `78.108` vs `81.122`
  - `OSCR`: `77.839` vs `80.544`
- The stronger signal appears at the operating-point level:
  - low-dimension `best-DTACC` FAR mean: `0.289 +- 0.184`
  - frozen baseline `best-DTACC` FAR mean: `0.082 +- 0.026`
- At the recall-target threshold, the comparison is noisier:
  - low-dimension FAR mean: `0.627 +- 0.256`
  - frozen baseline FAR mean: `0.652 +- 0.180`

Conclusion for the theory:

- The new low-vs-high comparison gives real empirical support that embedding dimension matters for practical operating quality.
- It supports the paper best through `AUROC`, `OSCR`, and especially the `best-DTACC` operating point.
- It does **not** give a clean monotone confirmation of the strongest isotropic FAR-decay claim.

## Anisotropy Evidence

### Paper-style `delta_cap` estimator

From `/Users/goodday/Downloads/pap/delta_cap_summary.md`:

- `BloodMNIST` mean `\hat{\delta}_{cap}`: about `0.0167 +- 0.0077`
- `DermaMNIST` mean `\hat{\delta}_{cap}`: about `0.0147 +- 0.0065`
- Both values are small and very similar.
- The maximizing thresholds are also both small:
  - about `0.028` for BloodMNIST
  - about `0.040` for DermaMNIST

### Accepted-unknown alignment diagnostic

From `/Users/goodday/Downloads/pap/accepted_unknown_alignment.md`:

- accepted unknowns are only slightly more prototype-aligned than all unknowns
- this is true for both the recall-target and best-DTACC thresholds
- the effect is not strong enough to cleanly explain why `DermaMNIST` is harder

### Interpretation

- The paper-style `\hat{\delta}_{cap}` estimator does **not** separate the datasets in the way the failure-mode story would need.
- The simpler proxy diagnostics also do **not** produce a clean explanation.
- In fact, under those simpler proxies, `BloodMNIST` can look *more* prototype-aligned than `DermaMNIST`, even though `DermaMNIST` is empirically harder.

Conclusion for the theory:

- The anisotropy part of the paper remains plausible as a theoretical warning.
- The current empirical diagnostics are still not strong confirmation.
- This part should be written honestly as:
  - a measured diagnostic attempt
  - not a solved empirical validation

## Bottom Line

- The paper now has stronger empirical support for the threshold/operating-point story.
- The paper now has stronger empirical support that dimension affects open-set behavior.
- The paper does **not** yet have strong empirical confirmation of the anisotropy-failure mechanism.

## Recommended Paper Framing

- Present the dimension result as `initial multi-split empirical support` for dimension sensitivity.
- Present the anisotropy section as `diagnostic evidence remains inconclusive`.
- Do not overclaim that the current experiments have fully verified the anisotropy theory.

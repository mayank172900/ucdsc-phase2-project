# Phase 1 Baseline Summary

## Status

Phase 1 is complete.

The baseline reproductions are frozen and summarized here for the conference workflow.

## Locked Inputs

- Original repo: `/Users/goodday/Downloads/pap/UCDSC-main`
- Locked baseline repo: `/Users/goodday/Downloads/pap/paper_conf_run`

## Locked Baseline Outputs

- BloodMNIST log: `/Users/goodday/Downloads/pap/out/live/baseline_bloodmnist_20260324_182550.log`
- DermaMNIST log: `/Users/goodday/Downloads/pap/out/live/baseline_dermamnist_20260324_211538.log`
- BloodMNIST results: `/Users/goodday/Downloads/pap/out/results/resnet18_NirvanaOpenset_38.0_False_0.0/bloodmnist.csv`
- BloodMNIST normalized results: `/Users/goodday/Downloads/pap/out/results/resnet18_NirvanaOpenset_38.0_False_0.0/bloodmnist_b9.csv`
- DermaMNIST results: `/Users/goodday/Downloads/pap/out/results/resnet18_NirvanaOpenset_38.0_False_0.0/dermamnist.csv`
- DermaMNIST normalized results: `/Users/goodday/Downloads/pap/out/results/resnet18_NirvanaOpenset_38.0_False_0.0/dermamnist_b9.csv`

## Run Configuration

- Model: `resnet18`
- Loss: `NirvanaOpenset`
- Margin: `38.0`
- Outlier exposure: `False` (`--no-oe`)
- Optimizer: `RMSprop`
- Epochs: `400`

## Primary Baseline Summary

These are the main baseline numbers from the standard score outputs.

| Dataset | Splits | ACC mean +- std | AUROC mean +- std | OSCR mean +- std | TNR mean +- std | DTACC mean +- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BloodMNIST | 5 | 98.242 +- 0.662 | 81.122 +- 8.230 | 80.544 +- 8.193 | 34.601 +- 18.037 | 76.820 +- 5.796 |
| DermaMNIST | 4 | 84.414 +- 6.389 | 67.373 +- 7.903 | 60.133 +- 11.355 | 14.545 +- 5.641 | 68.146 +- 3.255 |

## Normalized `b9` Summary

These are the normalized-score baseline numbers from the `_b9` outputs.

| Dataset | Splits | ACC mean +- std | AUROC mean +- std | OSCR mean +- std | TNR mean +- std | DTACC mean +- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BloodMNIST `b9` | 5 | 98.242 +- 0.662 | 81.068 +- 8.104 | 80.471 +- 8.079 | 33.936 +- 18.154 | 76.869 +- 5.832 |
| DermaMNIST `b9` | 4 | 84.414 +- 6.389 | 70.398 +- 4.968 | 62.562 +- 8.699 | 16.019 +- 5.837 | 68.815 +- 2.208 |

## Per-Split Primary Results

### BloodMNIST

| Split | ACC | AUROC | OSCR | TNR | DTACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 98.162 | 68.074 | 67.670 | 22.458 | 68.543 |
| 1 | 98.986 | 93.493 | 93.132 | 67.481 | 86.673 |
| 2 | 97.569 | 79.191 | 78.617 | 15.068 | 76.036 |
| 3 | 97.480 | 84.455 | 83.276 | 31.176 | 77.269 |
| 4 | 99.014 | 80.397 | 80.023 | 36.819 | 75.577 |

### DermaMNIST

| Split | ACC | AUROC | OSCR | TNR | DTACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 87.198 | 72.765 | 65.236 | 14.613 | 69.088 |
| 1 | 75.570 | 54.635 | 41.858 | 22.034 | 63.212 |
| 2 | 82.017 | 67.077 | 60.848 | 6.149 | 68.000 |
| 3 | 92.872 | 75.017 | 72.590 | 15.385 | 72.286 |

## Runtime Notes

- BloodMNIST baseline runtime: `0:24:11`
- DermaMNIST baseline runtime: `0:25:33`

## Phase 1 Conclusion

- `BloodMNIST` is a strong baseline dataset for the paper.
- `DermaMNIST` is a harder dataset and gives a useful second baseline with noticeably lower rejection performance.
- These frozen results should be treated as the baseline reference for all later theory-driven experiments.

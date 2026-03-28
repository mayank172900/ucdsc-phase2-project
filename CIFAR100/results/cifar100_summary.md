# CIFAR-100 Extension Summary

## Purpose

This is an extension experiment outside the original medical-imaging scope of the UCDSC paper.

It was run to test whether the method still behaves strongly on a non-medical open-set image benchmark.

## Main Results

Source files:

- `cifar100_50.csv`
- `cifar100_50_b9.csv`

Primary mean +- std over 5 splits:

- `ACC`: `94.53 +- 1.00`
- `AUROC`: `92.03 +- 2.06`
- `OSCR`: `87.95 +- 1.71`
- `TNR`: `59.33 +- 7.81`

Normalized `b9` mean +- std over 5 splits:

- `ACC`: `94.70 +- 1.02`
- `AUROC`: `92.07 +- 2.08`
- `OSCR`: `88.11 +- 1.64`
- `TNR`: `60.34 +- 7.22`

## Interpretation

- The run completed correctly across all 5 splits.
- Performance remains strong despite moving outside the original medical dataset setting.
- This extension suggests the method is not limited only to the medical benchmarks used in the base paper.

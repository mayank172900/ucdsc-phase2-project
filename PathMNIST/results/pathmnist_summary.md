# PathMNIST Extension Summary

## Purpose

This is an extension experiment, not a direct reproduction of the original UCDSC paper.

It was designed to test the low-dimensional regime:

- known classes `C = 6`
- embedding dimension `d = 4`
- therefore `d < C - 1` because `4 < 5`

## Main Results

Source files:

- `pathmnist.csv`
- `pathmnist_b9.csv`

Primary mean +- std over 5 splits:

- `ACC`: `42.840 +- 5.705`
- `AUROC`: `36.141 +- 13.591`
- `OSCR`: `12.491 +- 6.034`
- `TNR`: `1.923 +- 3.582`

Normalized `b9` mean +- std over 5 splits:

- `ACC`: `42.840 +- 5.705`
- `AUROC`: `30.964 +- 16.765`
- `OSCR`: `13.593 +- 6.234`
- `TNR`: `1.902 +- 3.537`

## Interpretation

- The run is technically valid and complete.
- Performance is weak and highly variable across splits.
- This is qualitatively consistent with the project theory that rejection behavior degrades in the low-dimensional regime `d < C - 1`.

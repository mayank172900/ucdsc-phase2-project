# UCDSC Phase 2 Project

## Team

- Mayank Sharma
- Rohit Mourya
- Danie George John

## Overview

This repository contains our Phase-2 project based on:

- `UCDSC: Open Set UnCertainty aware Deep Simplex Classifier for Medical Image Datasets`

Our work includes:

1. paper-aligned baseline reproduction on `BloodMNIST` and `DermaMNIST`
2. threshold / operating-point analysis
3. low-dimensional `BloodMNIST` study
4. `CIFAR-100` extension outside the original medical domain
5. `PathMNIST` extension in the low-dimensional regime `d < C - 1`

## Repository Layout

- `baseline/`
- `PathMNIST/`
- `CIFAR100/`
- `else/`

## Folder Details

### `baseline/`

Contains:

- baseline reproduction code
- baseline result files and summaries

Expected local dataset folder:

- `baseline/data/`

Recommended dataset contents:

- `bloodmnist/`
- `dermamnist/`
- `octmnist/`
- `tissuemnist/`
- `asc/`
- `300K_random_images/`

### `PathMNIST/`

Contains:

- Mac/MPS PathMNIST extension code
- PathMNIST result files and summary

Expected local dataset folder:

- `PathMNIST/data/pathmnist/pathmnist.npz`

### `CIFAR100/`

Contains:

- Windows/CUDA CIFAR-100 extension code
- CIFAR-100 result files and summary

Expected local dataset folder:

- `CIFAR100/data/cifar100/`

### `else/`

Contains:

- final Phase-2 PPT
- final report PDF
- threshold/theory/analysis summaries
- geometry plots, geometry summary table, and plot generator

## Important Framing

- `BloodMNIST` and `DermaMNIST` are paper-aligned reproduced baselines.
- `PathMNIST` and `CIFAR-100` are extension experiments.
- `PathMNIST` is a theory-motivated low-dimensional extension experiment, not a direct original-paper reproduction.

For a direct mapping between the repository contents and our updated paper, see:

- `PAPER_REPO_CONNECTION.md`

## Key Results

### Reproduced Baselines

| Dataset | ACC | AUROC | OSCR | TNR | DTACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| BloodMNIST | 98.242 +- 0.662 | 81.122 +- 8.230 | 80.544 +- 8.193 | 34.601 +- 18.037 | 76.820 +- 5.796 |
| DermaMNIST | 84.414 +- 6.389 | 67.373 +- 7.903 | 60.133 +- 11.355 | 14.545 +- 5.641 | 68.146 +- 3.255 |

### CIFAR-100 Extension

- `ACC`: `94.53 +- 1.00`
- `AUROC`: `92.03 +- 2.06`
- `OSCR`: `87.95 +- 1.71`
- `TNR`: `59.33 +- 7.81`

### PathMNIST Extension (`d < C - 1`)

Setup:

- known classes `C = 6`
- embedding dimension `d = 4`
- therefore `d < C - 1`

Main results:

- `ACC`: `42.840 +- 5.705`
- `AUROC`: `36.141 +- 13.591`
- `OSCR`: `12.491 +- 6.034`
- `TNR`: `1.923 +- 3.582`

## Geometry Evidence

Geometry outputs are stored in:

- `else/geometry_plots/README.md`
- `else/geometry_plots/geometry_summary.csv`
- `else/geometry_plots/all_datasets_geometry_summary.png`
- `else/geometry_plots/generate_geometry_plots.py`

Each saved run also has:

- distance heatmap
- cosine heatmap
- `lambda_j` bar plot
- PCA projection plot

Examples:

- `else/geometry_plots/bloodmnist_baseline_distance_heatmap.png`
- `else/geometry_plots/dermamnist_baseline_lambda_bar.png`
- `else/geometry_plots/pathmnist_d4_cosine_heatmap.png`
- `else/geometry_plots/cifar100_extension_projection.png`

Main geometry summary:

- `BloodMNIST baseline`, `DermaMNIST baseline`, `BloodMNIST feat_dim=32`, and `CIFAR-100` are simplex-compatible in the saved runs: pairwise-distance CV is effectively `0`, norm CV is effectively `0`, and `lambda_mean = 1`.
- `PathMNIST` is visibly non-symmetric in the saved run: distance CV `0.3411`, norm CV `0.4290`, `lambda_mean = 1.0412`, and `lambda_max = 1.0768`.
- This makes the geometry evidence directly useful for the paper's dimension discussion, because it separates the symmetric saved runs from the degraded low-dimensional saved run.
- Important caveat: the `PathMNIST` non-symmetry comes from the implemented low-dimensional fallback that skips simplex initialization when `feat_dim < num_classes - 1`, so this is evidence from our saved implementation, not a direct test of a balanced equal-norm low-dimensional construction from the theory.

## Results Mapped To The Paper

- `baseline/results/bloodmnist.csv` and `baseline/results/dermamnist.csv` directly support the reproduced UCDSC baseline discussion in the paper.
- `else/summaries/phase3_threshold_summary.*` and `else/summaries/phase4_threshold_calibration.*` directly support the threshold-selection and operating-point analysis in the paper.
- `baseline/results/blood_dim32_summary.*` directly supports the low-dimensional `BloodMNIST` dimension-sensitivity discussion in the paper.
- `else/summaries/anisotropy_summary.*`, `delta_cap_*`, and `accepted_unknown_alignment.*` support the paper's anisotropy / diagnostic discussion.
- `else/geometry_plots/*` supports the paper's geometry story by showing which saved runs remain simplex-compatible and where visible symmetry degradation appears.
- `PathMNIST/results/*` is a theory-linked companion experiment for the paper's central `d < C - 1` question.
- `CIFAR100/results/*` is a theory-linked companion stress test showing behavior beyond the reproduced medical baseline setting.

## Dataset Note

Datasets are intentionally not committed to this GitHub-ready package because they are large.

They should be placed manually in the expected local folders before rerunning the experiments.

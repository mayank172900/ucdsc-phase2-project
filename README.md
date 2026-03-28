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

## Important Framing

- `BloodMNIST` and `DermaMNIST` are paper-aligned reproduced baselines.
- `PathMNIST` and `CIFAR-100` are extension experiments.
- `PathMNIST` is a theory-motivated low-dimensional extension experiment, not a direct original-paper reproduction.

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

## Dataset Note

Datasets are intentionally not committed to this GitHub-ready package because they are large.

They should be placed manually in the expected local folders before rerunning the experiments.

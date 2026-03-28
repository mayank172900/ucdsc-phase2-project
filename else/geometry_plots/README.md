# Geometry Plots

This folder contains prototype-geometry evidence derived from the saved artifacts/checkpoints for each run.

## Summary

| Dataset | Centers | feat_dim | dist CV | norm CV | lambda_mean | lambda_max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BloodMNIST baseline | 4 | 512 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| DermaMNIST baseline | 4 | 512 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| BloodMNIST feat_dim=32 | 4 | 32 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| PathMNIST d=4 | 6 | 4 | 0.3411 | 0.4290 | 1.0412 | 1.0768 |
| CIFAR-100 extension | 4 | 128 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## Notes

- `BloodMNIST baseline`: Saved split-0 artifact export from the frozen baseline run.
- `DermaMNIST baseline`: Saved split-0 artifact export from the frozen baseline run.
- `BloodMNIST feat_dim=32`: Saved split-0 artifact export from the low-dimension run.
- `PathMNIST d=4`: Low-dimensional fallback run. The saved code skips simplex initialization when feat_dim < num_classes - 1, so non-symmetry here reflects the actual saved implementation choice.
- `CIFAR-100 extension`: Saved split-0 criterion checkpoint from the CIFAR-100 extension run.

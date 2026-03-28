# How Our Experimental Results Connect to the Theory in Our Updated Paper

In our updated paper, the central claim is that simplex-ratio open-set recognition should not be understood only in the regular-simplex regime `d >= C - 1`. The paper develops a broader geometric framework that covers all embedding dimensions. For arbitrary distinct prototypes, the uncertainty score `U` remains globally Lipschitz and the acceptance region remains compact. Under balanced equal-norm codes, which exist in every `d >= 2`, the auxiliary score `U_2` has exact ball-union sublevel geometry. The sharp dichotomy in the paper is that perfect one-distance simplex symmetry holds if and only if `d >= C - 1`; below that threshold the geometry does not vanish, but degrades in a structured way governed by the simplex-defect parameter.

Our experiments connect to that theory in a direct way.

## BloodMNIST and DermaMNIST

Our reproduced UCDSC baseline is much stronger on BloodMNIST than on DermaMNIST. On BloodMNIST, we obtain `ACC 98.242 +- 0.662`, `AUROC 81.122 +- 8.230`, and `OSCR 80.544 +- 8.193`. On DermaMNIST, we obtain `ACC 84.414 +- 6.389`, `AUROC 67.373 +- 7.903`, and `OSCR 60.133 +- 11.355`.

This directly matches the empirical story in the updated paper: the method does not behave equally across medical datasets, and the gap is visible not only in closed-set accuracy but also in open-set ranking and operating-point metrics. In the paper, this difference is used to motivate why the theory must talk about acceptance geometry, score behavior, and threshold selection, rather than only about classification accuracy. These two datasets therefore anchor the empirical part of the paper and provide the reference point for the threshold and dimension analyses that follow.

## Threshold Selection

The threshold results provide the clearest empirical support for the paper's practical theory. At the recall-target operating point corresponding to about `95%` known-class recall, the mean FAR is `0.652 +- 0.180` on BloodMNIST and `0.855 +- 0.056` on DermaMNIST. At the best-DTACC threshold, the mean FAR drops to `0.082 +- 0.026` on BloodMNIST and `0.244 +- 0.225` on DermaMNIST.

These numbers support the paper's threshold-selection theory directly. In the updated paper, the FAR--KRR result shows that threshold choice is a genuine operating decision and not a cosmetic post-processing step. Our reproduced measurements show exactly that: controlling known-class recall forces a significant FAR tradeoff, especially on the harder DermaMNIST setting. This is why the paper emphasizes data-driven thresholding and validation-based operating-point selection instead of treating the reject threshold as an afterthought.

## Low-Dimensional BloodMNIST

Our low-dimensional BloodMNIST run gives the most direct empirical support to the paper's dimension argument. When we reduce the embedding dimension from the default setting to `32`, the mean AUROC drops from `81.122 +- 8.230` to `78.108 +- 12.801`, the mean OSCR drops from `80.544 +- 8.193` to `77.839 +- 12.824`, and the best-DTACC FAR worsens from `0.082 +- 0.026` to `0.289 +- 0.184`.

This aligns with the paper's central low-dimensional message. The sharp dichotomy in the theory says that `d >= C - 1` is the threshold for perfect simplex symmetry, not the threshold for meaningful rejection geometry itself. Once we move away from the ideal regime, we should expect degradation rather than total collapse. The BloodMNIST dimension experiment matches that prediction: open-set behavior remains meaningful, but it becomes weaker at lower dimension, especially at the operating-point level.

## Geometry Evidence from the Saved Runs

We also generated explicit geometry plots from the saved prototype artifacts. The main outputs are stored under `else/geometry_plots/`, including `geometry_summary.csv`, `all_datasets_geometry_summary.png`, per-run distance heatmaps, cosine heatmaps, `lambda_j` bar plots, PCA projections, and the generator script `generate_geometry_plots.py`.

This geometry evidence supports the paper in a very direct way. `BloodMNIST baseline`, `DermaMNIST baseline`, `BloodMNIST feat_dim=32`, and `CIFAR-100` all come out as simplex-compatible in the saved runs: pairwise-distance CV is effectively zero, norm CV is effectively zero, and `lambda_mean = 1`. That matches the symmetric geometry discussed in the paper and is consistent with the part of the theory that treats the regular-simplex regime as the exact one-distance case.

`PathMNIST`, on the other hand, is visibly non-symmetric in the saved run: distance CV `0.3411`, norm CV `0.4290`, `lambda_mean = 1.0412`, and `lambda_max = 1.0768`. This makes the geometry degradation visible in a way that matches the paper's low-dimensional direction. At the same time, there is an important caveat. In our saved implementation, the code skips simplex initialization when `feat_dim < num_classes - 1`, so the `PathMNIST` non-symmetry comes from that low-dimensional fallback. It is therefore valid evidence from our implementation, but it is not the same as directly testing the balanced equal-norm low-dimensional construction described in the theory.

## PathMNIST and the `d < C - 1` Regime

Our PathMNIST experiment is the clearest companion experiment for the paper's low-dimensional theory. In this setup, the number of known classes is `C = 6` and the embedding dimension is `d = 4`, so we are explicitly in the regime `d < C - 1`.

The resulting performance is weak and unstable: `ACC 42.840 +- 5.705`, `AUROC 36.141 +- 13.591`, `OSCR 12.491 +- 6.034`, and `TNR 1.923 +- 3.582`.

This is important because the updated paper does not claim that low-dimensional behavior remains ideal. Its claim is more precise: below the simplex threshold, exact simplex symmetry is no longer available, but the rejection geometry survives with degradation. PathMNIST gives us a concrete cancer-domain experiment that is consistent with that theoretical picture. The results do not suggest collapse into meaningless behavior; instead, they show a much harder and weaker operating regime, which is exactly the kind of degradation the paper predicts should become possible once `d < C - 1`.

## CIFAR-100 and the Broader Scope of the Framework

Our CIFAR-100 results are strong: `ACC 94.53 +- 1.00`, `AUROC 92.03 +- 2.06`, `OSCR 87.95 +- 1.71`, and `TNR 59.33 +- 7.81`.

These results support a different part of the paper's theory direction. The updated paper is written as a general framework for simplex-ratio open-set recognition, not as a statement tied only to one MedMNIST benchmark family. The CIFAR-100 experiment therefore serves as a companion stress test for that broader viewpoint. The strong results suggest that the operating principles studied in the paper can remain useful beyond the exact medical benchmark setting used for the core reproduced analysis.

## Anisotropy Diagnostics

The updated paper is careful not to overstate the anisotropy story, and our results support that caution. The empirical `delta_cap` values remain small and similar across the reproduced datasets, approximately `0.0167 +- 0.0077` on BloodMNIST and `0.0147 +- 0.0065` on DermaMNIST.

This is why the paper concludes that the experiments support the threshold and dimension analyses more strongly than the anisotropy explanation. In other words, the anisotropy warning remains theoretically motivated, but our present diagnostics do not isolate it cleanly enough to explain the BloodMNIST-DermaMNIST gap on their own. That is not a contradiction of the paper. It is part of the paper's conclusion.

## Final Connection

Taken together, our results support the updated paper at the exact points where the paper is strongest.

BloodMNIST and DermaMNIST support the reproduced baseline story. The threshold experiments support the operating-point theory. The low-dimensional BloodMNIST study supports the dimension-sensitivity claim. The geometry plots show where the saved runs remain simplex-compatible and where visible symmetry degradation appears. PathMNIST provides a concrete companion experiment for the paper's `d < C - 1` theory direction. CIFAR-100 supports the broader scope of the framework beyond the original reproduced medical setup. The anisotropy diagnostics match the paper's own conclusion that this part remains suggestive rather than fully isolated.

So the code and results in this repository are the empirical side of the theory developed in our updated paper.

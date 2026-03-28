# Mac MPS Run Guide

This bundle adds `PathMNIST` support for a cancer-focused low-dimensional experiment.

## Why this dataset

`PathMNIST` is a colorectal cancer histology dataset from MedMNIST with 9 classes.

In this bundle, the predefined `pathmnist` open-set splits use `C = 6` known classes per run. The recommended command below sets `d = 4` via `--embed-dim 4`, so the experiment is explicitly in the `d < C - 1` regime:

```text
d = 4, C - 1 = 5, therefore 4 < 5
```

That inequality is an inference from the chosen split design plus the configured projected embedding dimension.

## 1. Environment

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
```

If you need an explicit Apple Silicon wheel:

```bash
pip install torch torchvision
```

## 2. Recommended first run

Quick single-split validation on MPS:

```bash
python3 ./NirvanaOSR.py \
  --dataset pathmnist \
  --dataroot ./data \
  --outf ./out \
  --model classifier32 \
  --embed-dim 4 \
  --batch-size 128 \
  --max-epoch 100 \
  --lr 0.01 \
  --optim rmsprop \
  --l1-weight 0.001 \
  --l2-weight 0.001 \
  --Expand 100 \
  --margin 48 \
  --uncertainty-weight 5.0 \
  --outlier-weight 1.0 \
  --eval-freq 100 \
  --print-freq 999999 \
  --split-idx 0 \
  --no-oe
```

## 3. Full 5-split run

```bash
./run_pathmnist_lowdim_mps.sh
```

Equivalent direct command:

```bash
python3 ./NirvanaOSR.py \
  --dataset pathmnist \
  --dataroot ./data \
  --outf ./out \
  --model classifier32 \
  --embed-dim 4 \
  --batch-size 128 \
  --max-epoch 100 \
  --lr 0.01 \
  --optim rmsprop \
  --l1-weight 0.001 \
  --l2-weight 0.001 \
  --Expand 100 \
  --margin 48 \
  --uncertainty-weight 5.0 \
  --outlier-weight 1.0 \
  --eval-freq 100 \
  --print-freq 999999 \
  --no-oe
```

## 4. Outputs

Logs:

```text
./out/live/
```

Models:

```text
./out/models/pathmnist/
```

Results:

```text
./out/results/classifier32_ed4_NirvanaOpenset_48.0_False_0.0/pathmnist.csv
```

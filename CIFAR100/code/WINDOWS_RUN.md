# Windows CUDA Run Guide

This bundle uses the CIFAR-100-capable code path from the original DSC/UCDSC workspace.

## 1. Environment

Open PowerShell in this folder and run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-windows.txt
```

Install a CUDA-enabled PyTorch build if plain `pip install -r ...` does not pull one automatically for your machine. For example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## 2. Recommended first run

This runs CIFAR-100 OSR on Windows GPU without outlier exposure, so CIFAR downloads automatically and no extra background file is required:

```powershell
python .\NirvanaOSR.py `
  --dataset cifar100 `
  --dataroot .\data `
  --outf .\out `
  --model classifier32 `
  --batch-size 128 `
  --max-epoch 100 `
  --lr 0.1 `
  --margin 48 `
  --Expand 200 `
  --out-num 50 `
  --no-oe
```

Equivalent `cmd.exe` launcher:

```cmd
run_cifar100_cuda.bat
```

## 3. Full run with outlier exposure

If you also want the 300K random images background set, place:

```text
.\data\300K_random_images\300K_random_images.npy
```

Then run:

```powershell
python .\NirvanaOSR.py `
  --dataset cifar100 `
  --dataroot .\data `
  --outf .\out `
  --model classifier32 `
  --batch-size 128 `
  --max-epoch 100 `
  --lr 0.1 `
  --margin 48 `
  --Expand 200 `
  --out-num 50 `
  --oe `
  --oe-path .\data\300K_random_images\300K_random_images.npy
```

## 4. Outputs

Results go under:

```text
.\out\results\
.\out\models\
```

For CIFAR-100 with `out-num=50`, the main CSV is written as:

```text
.\out\results\classifier32_NirvanaOpenset_48.0_False_0.0\cifar100_50.csv
```

Resume checkpoints are written per split under:

```text
.\out\models\cifar100\classifier32_NirvanaOpenset_48.0_False_50\
```

If a run stops midway, rerun the exact same command. The script now reloads the per-split resume checkpoint automatically and continues from the next unfinished epoch.

@echo off
setlocal

set SCRIPT_DIR=%~dp0
set DATA_ROOT=%SCRIPT_DIR%data
set OUT_DIR=%SCRIPT_DIR%out

python "%SCRIPT_DIR%NirvanaOSR.py" ^
  --dataset cifar100 ^
  --dataroot "%DATA_ROOT%" ^
  --outf "%OUT_DIR%" ^
  --model classifier32 ^
  --batch-size 128 ^
  --max-epoch 100 ^
  --lr 0.1 ^
  --margin 48 ^
  --Expand 200 ^
  --out-num 50 ^
  --no-oe

endlocal

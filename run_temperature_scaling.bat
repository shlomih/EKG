@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo Running temperature scaling calibration for V3 model...
echo Fits global + per-class temperature, re-tunes thresholds, saves best result
echo.
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe temperature_scaling.py --model models/ecg_multilabel_v3_best.pt
echo.
pause

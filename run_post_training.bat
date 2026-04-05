@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo ============================================================
echo  EKG V3 Post-Training Pipeline
echo  1. Threshold tuning  2. Temperature scaling  3. AUROC eval
echo ============================================================
echo.

echo [Step 1/3] Threshold tuning (V3 best checkpoint)...
echo ---------------------------------------------------
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe tune_thresholds.py --model v3
if errorlevel 1 (
    echo ERROR: Threshold tuning failed!
    pause
    exit /b 1
)
echo.

echo [Step 2/3] Temperature scaling calibration...
echo ----------------------------------------------
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe temperature_scaling.py --model models/ecg_multilabel_v3_best.pt
if errorlevel 1 (
    echo ERROR: Temperature scaling failed!
    pause
    exit /b 1
)
echo.

echo [Step 3/3] Per-class AUROC evaluation...
echo -----------------------------------------
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe eval_v3_auroc.py --model models/ecg_multilabel_v3_best.pt
if errorlevel 1 (
    echo ERROR: AUROC evaluation failed!
    pause
    exit /b 1
)
echo.

echo ============================================================
echo  All 3 steps completed!
echo  Results:
echo    models/thresholds_v3.json  (updated thresholds)
echo    eval_v3_auroc_results.json (per-class AUROC + F1)
echo ============================================================
pause

@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo Running V3 per-class AUROC evaluation...
echo Evaluates PTB-XL test, Challenge test, and combined — takes ~5 min on CPU
echo.
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe eval_v3_auroc.py
echo.
echo Results saved to eval_v3_auroc_results.json
pause

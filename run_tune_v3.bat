@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo Running threshold tuning for V3 model (26 classes)...
echo This will take a few minutes (loading ~111k records + inference on val+test)
echo.
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe tune_thresholds.py --model v3
echo.
echo Done. Results saved to models/thresholds_v3.json
pause

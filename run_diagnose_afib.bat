@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo Running AFIB signal loading diagnostic...
echo This checks all ~11,600 AFIB records for zero/corrupted signals
echo.
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe diagnose_afib.py
echo.
pause

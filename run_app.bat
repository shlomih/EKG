@echo off
title EKG Intelligence Platform

:: ---- Find local IPv4 (first non-loopback result) ----
set LOCAL_IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4"') do (
    if not defined LOCAL_IP set LOCAL_IP=%%a
)
:: Trim leading space
set LOCAL_IP=%LOCAL_IP: =%

echo.
echo ============================================================
echo   EKG Intelligence Platform -- Local Dev Server
echo ============================================================
echo   PC    : http://localhost:8501
echo   Phone : http://%LOCAL_IP%:8501
echo.
echo   Phone must be on the same WiFi as this PC.
echo   If phone can't connect, run once in elevated PowerShell:
echo   New-NetFirewallRule -DisplayName "Streamlit 8501" ^
echo     -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow
echo ============================================================
echo.

:: ---- Kill any process already on port 8501 ----
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 "') do (
    taskkill /PID %%p /F >nul 2>&1
)

:: ---- Start Streamlit on all interfaces ----
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run app.py ^
    --server.address 0.0.0.0 ^
    --server.port 8501

pause

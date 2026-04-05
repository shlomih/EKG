@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo Fixing colab-proxy MCP trust setting in ~/.claude.json...
echo.
C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe fix_colab_mcp_trust.py
echo.
pause

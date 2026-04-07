@echo off
cd /d C:\Users\osnat\Documents\Shlomi\EKG
echo.
echo === Step 1: Build CODE-15%% index ===
"C:\Users\osnat\AppData\Local\Programs\Python\Python314\python.exe" dataset_code15.py --index
if errorlevel 1 (
    echo ERROR: Index build failed. Aborting.
    pause
    exit /b 1
)

echo.
echo === Step 2: Create tar for Drive upload ===
echo Archiving ekg_datasets\code15\ ...
tar -cf ekg_datasets\code15.tar -C ekg_datasets code15
if errorlevel 1 (
    echo ERROR: tar failed.
    pause
    exit /b 1
)

echo.
echo === Done ===
for %%I in (ekg_datasets\code15.tar) do echo code15.tar size: %%~zI bytes
echo.
echo Next: upload ekg_datasets\code15.tar to Google Drive at:
echo   MyDrive/EKG/ekg_datasets/code15.tar
echo.
pause

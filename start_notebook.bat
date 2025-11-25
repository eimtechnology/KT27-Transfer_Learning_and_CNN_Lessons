@echo off
cd /d "%~dp0"
echo ===================================================
echo Transfer Learning Course - One-Click Launcher
echo ===================================================

if not exist "transfer_learning_env" (
    echo [INFO] First time setup detected. Installing environment...
    python setup.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Setup failed!
        pause
        exit /b %ERRORLEVEL%
    )
)

echo [INFO] Checking environment integrity...
if not exist "transfer_learning_env\Scripts\jupyter.exe" (
    echo [WARN] Jupyter not found in environment. Attempting to install dependencies...
    if exist "transfer_learning_env\Scripts\python.exe" (
        "transfer_learning_env\Scripts\python.exe" -m pip install -r requirements.txt
    ) else (
        echo [ERROR] Python executable not found in environment. Re-running setup...
        python setup.py
    )
)

echo [INFO] Activating environment...
if exist "transfer_learning_env\Scripts\activate.bat" (
    call transfer_learning_env\Scripts\activate.bat
)

echo [INFO] Starting Jupyter Notebook...
echo [TIP] A browser window should open automatically.
echo [TIP] To stop, press Ctrl+C in this terminal.

if exist "transfer_learning_env\Scripts\jupyter.exe" (
    "transfer_learning_env\Scripts\jupyter.exe" notebook
) else (
    echo [ERROR] Failed to find jupyter.exe. Please run 'python setup.py' manually to fix the environment.
    pause
    exit /b 1
)

pause

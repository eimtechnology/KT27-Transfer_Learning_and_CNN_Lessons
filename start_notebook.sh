#!/bin/bash
cd "$(dirname "$0")"

echo "==================================================="
echo "Transfer Learning Course - One-Click Launcher"
echo "==================================================="

if [ ! -d "transfer_learning_env" ]; then
    echo "[INFO] First time setup detected. Installing environment..."
    python3 setup.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Setup failed!"
        exit 1
    fi
fi

# Check for jupyter executable
if [ ! -f "transfer_learning_env/bin/jupyter" ]; then
    echo "[WARN] Jupyter not found. Attempting to install dependencies..."
    ./transfer_learning_env/bin/python -m pip install -r requirements.txt
fi

echo "[INFO] Activating environment..."
source transfer_learning_env/bin/activate

echo "[INFO] Starting Jupyter Notebook..."
if [ -f "transfer_learning_env/bin/jupyter" ]; then
    ./transfer_learning_env/bin/jupyter notebook
else
    echo "[ERROR] Failed to find jupyter executable. Please run 'python3 setup.py' manually."
    exit 1
fi

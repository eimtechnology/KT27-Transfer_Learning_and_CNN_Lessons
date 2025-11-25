#!/usr/bin/env python3
"""
Simplified Environment Setup Script - Transfer Learning Course
One-click setup for deep learning environment
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# Configuration  
VENV_NAME = "transfer_learning_env"
REQUIREMENTS_FILE = "requirements.txt"

def print_step(message):
    """Display current step"""
    print(f"\n>>> {message}")

def print_success(message):
    """Display success message"""
    print(f"[OK] {message}")

def print_error(message):
    """Display error message"""
    print(f"[ERROR] {message}")

def check_python_version():
    """Check Python version"""
    print_step("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print_error(f"Python version too old (current: {sys.version_info.major}.{sys.version_info.minor})")
        print_error("Please install Python 3.8 or higher")
        return False
    
    print_success(f"Python version check passed (version: {sys.version_info.major}.{sys.version_info.minor})")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print_step("Creating virtual environment...")
    
    venv_path = Path(VENV_NAME)
    
    # Remove old environment
    if venv_path.exists():
        print("Removing old virtual environment...")
        try:
            shutil.rmtree(venv_path)
        except Exception as e:
            print_error(f"Could not remove old environment: {e}")
            print("Please close all Jupyter/Python processes and try again")
            return False
    
    try:
        # Create new environment
        subprocess.run([sys.executable, "-m", "venv", VENV_NAME], check=True)
        print_success(f"Virtual environment created successfully: {VENV_NAME}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Virtual environment creation failed: {e}")
        return False

def get_python_path():
    """Get Python path in virtual environment"""
    if platform.system() == "Windows":
        return Path(VENV_NAME) / "Scripts" / "python.exe"
    else:
        return Path(VENV_NAME) / "bin" / "python"

def install_packages():
    """Install dependency packages"""
    print_step("Installing dependency packages...")
    
    if not Path(REQUIREMENTS_FILE).exists():
        print_error(f"Requirements file not found: {REQUIREMENTS_FILE}")
        return False
    
    python_path = get_python_path()
    
    try:
        # Upgrade pip
        print("Upgrading pip...")
        subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install dependencies
        print("Installing project dependencies...")
        subprocess.run([str(python_path), "-m", "pip", "install", "-r", REQUIREMENTS_FILE], 
                      check=True, capture_output=True)
        
        print_success("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Dependency installation failed: {e}")
        return False

def verify_installation():
    """Verify installation"""
    print_step("Verifying installation...")
    
    python_path = get_python_path()
    
    test_packages = ["torch", "torchvision", "matplotlib", "pandas", "jupyter"]
    
    for package in test_packages:
        try:
            subprocess.run([str(python_path), "-c", f"import {package}"], 
                          check=True, capture_output=True)
            print_success(f"{package} import successful")
        except subprocess.CalledProcessError:
            print_error(f"{package} import failed")
            return False
    
    return True

def setup_jupyter_kernel():
    """Setup Jupyter kernel"""
    print_step("Setting up Jupyter kernel...")
    
    python_path = get_python_path()
    
    try:
        # Install ipykernel
        subprocess.run([str(python_path), "-m", "pip", "install", "ipykernel"], 
                      check=True, capture_output=True)
        
        # Register kernel
        subprocess.run([str(python_path), "-m", "ipykernel", "install", "--user", 
                       "--name", "transfer_learning", "--display-name", "Transfer Learning Course"], 
                      check=True, capture_output=True)
        
        print_success("Jupyter kernel registered: transfer_learning")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"Kernel setup failed: {e}")
        print("You can still use the default Python kernel")
        return False

def create_activation_script():
    """Create activation script"""
    print_step("Creating activation script...")
    
    if platform.system() == "Windows":
        # Windows batch file
        script_content = f'''@echo off
echo Activating deep learning environment...
call "{VENV_NAME}\\Scripts\\activate.bat"
echo.
echo Environment activated successfully!
echo Usage:
echo   jupyter notebook  - Start Jupyter Notebook
echo   jupyter lab       - Start Jupyter Lab  
echo   deactivate        - Exit virtual environment
echo.
cmd /k
'''
        with open("activate.bat", "w", encoding="utf-8") as f:
            f.write(script_content)
        print_success("Windows activation script created: activate.bat")
    
    # Unix shell script
    script_content = f'''#!/bin/bash
echo "Activating deep learning environment..."
source "{VENV_NAME}/bin/activate"
echo ""
echo "Environment activated successfully!"
echo "Usage:"
echo "  jupyter notebook  - Start Jupyter Notebook"
echo "  jupyter lab       - Start Jupyter Lab"
echo "  deactivate        - Exit virtual environment"
echo ""
exec "$SHELL"
'''
    with open("activate.sh", "w") as f:
        f.write(script_content)
    os.chmod("activate.sh", 0o755)
    print_success("Unix activation script created: activate.sh")

def main():
    """Main function"""
    print("Transfer Learning Course - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    if not install_packages():
        return False
    
    # Verify installation
    if not verify_installation():
        print("Installation verification failed, but environment may still be usable")
    
    # Setup Jupyter kernel
    setup_jupyter_kernel()
    
    # Create activation script
    create_activation_script()
    
    # Completion message
    print("\n" + "=" * 50)
    print("Environment setup completed!")
    print("\nHow to use:")
    
    if platform.system() == "Windows":
        print("1. Double-click activate.bat")
    else:
        print("1. Run ./activate.sh")
    
    print("2. Run 'jupyter notebook' to start learning")
    print("3. Start your deep learning journey!")
    print("\nIf you encounter issues, re-run this script")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
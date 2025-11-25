# 🎓 Transfer Learning Course - Installation Guide

## 🚀 Quick Start (Recommended)

### For Windows Users
Double-click **`start_notebook.bat`**.
The script will automatically check the environment, install it if missing, and launch Jupyter Notebook.

### For macOS / Linux Users
Open a terminal and run:
```bash
# Grant execution permission and run
chmod +x start_notebook.sh
./start_notebook.sh
```

## 🔧 Manual Installation

If you prefer manual control, use the following commands:

### 1. Create Environment
```bash
python setup.py
```

### 2. Activate Environment
```bash
# Windows
transfer_learning_env\Scripts\activate

# macOS/Linux
source transfer_learning_env/bin/activate
```

### 3. Start Jupyter
```bash
jupyter notebook
```

## 📦 Course Environment

This course uses a virtual environment named `transfer_learning_env`, containing the following main components:

- ✅ **PyTorch** - Deep Learning Framework
- ✅ **TorchVision** - Computer Vision Tools
- ✅ **NumPy** - Numerical Computing
- ✅ **Pandas** - Data Processing
- ✅ **Matplotlib** - Data Visualization
- ✅ **Jupyter Lab/Notebook** - Interactive Development Environment

## 🎯 Verify Installation

After starting the environment, please open and run the following Notebook to verify:
`lessons/lesson1_environment_setup/setup_check.ipynb`

## 🚨 Troubleshooting

### Network Connection Issues
**Issue**: Package download fails or times out.
**Solution**:
Try installing manually using a trusted mirror (if applicable to your region) or check your internet connection.

```bash
# After activating environment
pip install -r requirements.txt
```

### Permission Issues
**Solution**:
- **Windows**: Run terminal as Administrator.
- **macOS/Linux**: Check directory permissions or use `chmod +x`.

### Reinstall
If you encounter unresolvable environment issues, the simplest method is a full reset:
1. Delete the `transfer_learning_env` folder.
2. Re-run `start_notebook.bat` or `start_notebook.sh`.

## 📞 Support

If you encounter problems:
1. Check Python version >= 3.8
2. Ensure network connection works
3. Check error logs
4. Try re-running the installation script

---
**Happy Learning!** 🚀

# 🔧 Troubleshooting Guide - Lesson 1

## Common Issues and Solutions

This guide helps you resolve common problems encountered during environment setup. Issues are organized by category for easy navigation.

---

## 🐍 Python Environment Issues

### Issue: "Python not found" or "python command not recognized"

**Symptoms:**
- Command prompt says `'python' is not recognized as an internal or external command`
- Cannot execute Python scripts

**Solutions:**

#### Windows:
```bash
# Check if Python is installed
py --version

# If Python is installed but 'python' command doesn't work:
# Add Python to PATH or use 'py' instead of 'python'
py -m venv dl_env
```

#### Linux/Mac:
```bash
# Check Python installation
python3 --version

# Use python3 instead of python
python3 -m venv dl_env
```

### Issue: Python version too old

**Symptoms:**
- Error: "Python 3.8+ required"
- Syntax errors with modern Python features

**Solution:**
```bash
# Check current version
python --version

# Install Python 3.8+ from python.org
# Or use conda to manage versions
conda install python=3.9
```

---

## 📦 Virtual Environment Issues

### Issue: Virtual environment creation fails

**Symptoms:**
- `python -m venv` command fails
- Permission errors during venv creation

**Solutions:**

#### Method 1: Update pip and try again
```bash
python -m pip install --upgrade pip
python -m venv dl_env
```

#### Method 2: Use conda instead
```bash
conda create -n dl_env python=3.9
conda activate dl_env
```

#### Method 3: Use virtualenv
```bash
pip install virtualenv
virtualenv dl_env
```

### Issue: Virtual environment activation fails

**Symptoms:**
- Environment doesn't activate
- Command prompt doesn't show environment name

**Solutions:**

#### Windows:
```bash
# PowerShell (if scripts execution is disabled)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
dl_env\Scripts\activate
```

#### Linux/Mac:
```bash
# Make sure to use 'source'
source dl_env/bin/activate

# Check if activated
which python
```

---

## 🔥 PyTorch Installation Issues

### Issue: PyTorch installation fails

**Symptoms:**
- `pip install torch` fails
- CUDA version conflicts
- Memory errors during installation

**Solutions:**

#### Method 1: Use official PyTorch installer
Visit [pytorch.org](https://pytorch.org/get-started/locally/) and get the exact command for your system.

```bash
# Example for Windows with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Example for CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Method 2: Clear pip cache
```bash
pip cache purge
pip install torch torchvision
```

#### Method 3: Use conda
```bash
conda install pytorch torchvision -c pytorch
```

### Issue: CUDA not detected after installation

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- GPU not utilized for computations

**Solutions:**

#### Check CUDA installation:
```bash
nvidia-smi  # Should show CUDA version
nvcc --version  # Should show CUDA compiler version
```

#### Reinstall PyTorch with correct CUDA version:
```bash
# Check CUDA version from nvidia-smi output
# Install matching PyTorch version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Verify installation:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
```

---

## 💾 Memory and Performance Issues

### Issue: Out of memory during benchmarks

**Symptoms:**
- "CUDA out of memory" errors
- System freezes during large matrix operations

**Solutions:**

#### Reduce benchmark sizes:
Edit the notebook and change:
```python
# Instead of [256, 512, 1024, 2048]
benchmark_sizes = [128, 256, 512]  # Smaller sizes
```

#### Clear GPU memory:
```python
import torch
torch.cuda.empty_cache()  # Clear GPU cache
```

#### Monitor memory usage:
```python
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

### Issue: Slow performance on expected fast hardware

**Symptoms:**
- GPU performance similar to CPU
- Low GFLOPS in benchmarks

**Solutions:**

#### Verify GPU utilization:
```bash
# Monitor GPU usage during benchmarks
nvidia-smi -l 1  # Update every 1 second
```

#### Check tensor device placement:
```python
# Make sure tensors are on GPU
x = torch.randn(1000, 1000, device='cuda')  # Explicit GPU placement
print(f"Tensor device: {x.device}")
```

#### Update GPU drivers:
- Download latest drivers from NVIDIA website
- Restart computer after installation

---

## 📊 Jupyter Notebook Issues

### Issue: Jupyter not starting or kernel not found

**Symptoms:**
- Jupyter Lab/Notebook won't start
- Kernel connection errors
- Wrong Python environment in notebook

**Solutions:**

#### Install Jupyter in virtual environment:
```bash
# Make sure virtual environment is activated
pip install jupyter jupyterlab ipykernel
```

#### Register kernel with Jupyter:
```bash
python -m ipykernel install --user --name=dl_env --display-name="Deep Learning"
```

#### Start Jupyter Lab:
```bash
jupyter lab
```

#### Select correct kernel:
In Jupyter Lab: Kernel → Change Kernel → "Deep Learning"

### Issue: Plots not showing in notebook

**Symptoms:**
- Matplotlib plots don't appear
- Empty output cells where plots should be

**Solutions:**

#### Enable inline plotting:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

#### Check backend:
```python
import matplotlib
print(matplotlib.get_backend())

# If needed, set backend explicitly
matplotlib.use('Agg')  # For non-interactive backend
```

---

## 🍎 Apple Silicon (M1/M2) Specific Issues

### Issue: MPS not available

**Symptoms:**
- `torch.backends.mps.is_available()` returns `False`
- Only CPU acceleration available

**Solutions:**

#### Update PyTorch:
```bash
pip install --upgrade torch torchvision
```

#### Check macOS version:
MPS requires macOS 12.3+
```bash
sw_vers  # Check macOS version
```

#### Verify MPS support:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Issue: Installation conflicts on Apple Silicon

**Symptoms:**
- Package installation failures
- Architecture mismatch errors

**Solutions:**

#### Use conda for better ARM64 support:
```bash
conda create -n dl_env python=3.9
conda activate dl_env
conda install pytorch torchvision -c pytorch
```

#### Install Rosetta 2 (if needed):
```bash
softwareupdate --install-rosetta
```

---

## 🔍 Debugging Tips

### General debugging approach:

1. **Check Python environment:**
   ```python
   import sys
   print(f"Python executable: {sys.executable}")
   print(f"Python version: {sys.version}")
   ```

2. **Verify package versions:**
   ```python
   import torch, torchvision, numpy, matplotlib
   print(f"PyTorch: {torch.__version__}")
   print(f"Torchvision: {torchvision.__version__}")
   print(f"NumPy: {numpy.__version__}")
   print(f"Matplotlib: {matplotlib.__version__}")
   ```

3. **Test basic operations:**
   ```python
   import torch
   x = torch.randn(2, 2)
   print(x)
   print(x.device)
   ```

4. **Check available devices:**
   ```python
   print(f"CUDA available: {torch.cuda.is_available()}")
   if hasattr(torch.backends, 'mps'):
       print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

### Get help from logs:

Most errors provide detailed information. Look for:
- **Import errors**: Missing packages
- **Runtime errors**: Version conflicts
- **Memory errors**: Insufficient RAM/VRAM
- **Device errors**: Hardware configuration issues

---

## 📞 Additional Resources

### Official Documentation:
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Troubleshooting](https://pytorch.org/docs/stable/notes/faq.html)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-toolkit)

### Community Support:
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)
- [Stack Overflow PyTorch Tag](https://stackoverflow.com/questions/tagged/pytorch)

### System-specific Guides:
- [Windows GPU Setup](https://pytorch.org/get-started/locally/#windows-python)
- [Linux GPU Setup](https://pytorch.org/get-started/locally/#linux-python)
- [macOS Setup](https://pytorch.org/get-started/locally/#macos-python)

---

## 🆘 Still Having Issues?

If you're still experiencing problems:

1. **Document the issue:**
   - Error message (full text)
   - Your operating system
   - Python version
   - Steps you tried

2. **Run the diagnostic script:**
   ```bash
   python setup.py  # Automated environment setup
   ```

3. **Check system resources:**
   - Available RAM/disk space
   - GPU driver versions
   - Background processes

4. **Try alternative approaches:**
   - Use Google Colab for cloud computing
   - Use Docker containers
   - Use conda instead of pip

Remember: Environment setup can be tricky, but once it's working, you'll have a solid foundation for all your deep learning projects!

---

**💡 Pro Tip:** Save your working environment configuration! Once everything works, export your requirements:
```bash
pip freeze > my_working_requirements.txt
```
# 🎓 Transfer Learning Course with PyTorch

A beginner-friendly deep learning transfer learning course using PyTorch framework, perfect for getting started with computer vision.

## 📚 Course Overview

This course teaches you transfer learning techniques from scratch through hands-on image classification projects using pre-trained models:

- **Lesson 1**: Environment Setup & Verification
- **Lesson 2**: Data Exploration & Analysis  
- **Lesson 3**: ResNet18 Transfer Learning
- **Lesson 4**: ResNet50 Transfer Learning
- **Lesson 5**: EfficientNet-B0 Transfer Learning
- **Lesson 6**: EfficientNet-B3 Transfer Learning
- **Lesson 7**: MobileNet-V2 Transfer Learning

## 🌸 Dataset Information

### Flowers-102 Dataset
This course uses the **Oxford Flowers-102** dataset for all transfer learning experiments:

- **Total Images**: 8,189 flower images
- **Classes**: 102 different flower categories
- **Split**: 
  - Training: ~1,020 images
  - Validation: ~1,020 images  
  - Test: ~6,149 images
- **Image Size**: Variable (resized to 224×224 for training)
- **Source**: [Oxford Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

**Sample Categories**: Alpine Sea Holly, Anthurium, Bee Balm, Bird of Paradise, Bolero Deep Blue, Bougainvillea, Camellia, Canna Lily, Canterbury Bells, Carnation, and 92 more beautiful flower species.

**Why Flowers-102?**
- Perfect balance of complexity and manageability
- High-quality, diverse images
- Ideal for demonstrating transfer learning benefits
- Automatic download through torchvision
- Well-established benchmark dataset

## 🚀 Quick Start

### For Windows Users
1. Double-click **`start_notebook.bat`**
2. The script will automatically:
   - Set up the Python environment (first time only)
   - Install necessary packages
   - Launch Jupyter Notebook in your browser

### For macOS / Linux Users
1. Open terminal
2. Run the startup script:
   ```bash
   chmod +x start_notebook.sh  # First time only
   ./start_notebook.sh
   ```

### Manual Setup (Advanced)
If you prefer manual control, you can use `python setup.py` to create the environment, and then activate it manually:

```bash
# Create env
python setup.py

# Activate (Windows)
transfer_learning_env\Scripts\activate

# Activate (Mac/Linux)
source transfer_learning_env/bin/activate

# Start Jupyter
jupyter notebook
```

### Learning Path

1. **Start with Lesson 1**: Open `lessons/lesson1_environment_setup/setup_check.ipynb` to verify your environment.
2. **Follow the progression**: Lessons 1-7 build upon each other.
3. **New to Notebooks?**: See [QUICKSTART.md](QUICKSTART.md).

## 📖 Course Structure

```
transfer-learning-course/
├── start_notebook.bat           # One-click launch for Windows
├── start_notebook.sh            # One-click launch for Mac/Linux
├── setup.py                     # Environment setup script
├── requirements.txt             # Package dependencies
├── lessons/
│   ├── lesson1_environment_setup/ # Environment verification
│   ├── lesson2_data_exploration/  # Data exploration
│   ├── lesson3_resnet18/          # ResNet18 hands-on
│   ├── lesson4_resnet50/          # ResNet50 hands-on
│   ├── lesson5_efficientnet_b0/   # EfficientNet-B0 hands-on
│   ├── lesson6_efficientnet_b3/   # EfficientNet-B3 hands-on
│   └── lesson7_mobilenet_v2/      # MobileNet-V2 hands-on
```

## 💻 System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **GPU**: Optional but recommended (CUDA-compatible)

## 🎯 Learning Objectives

By completing this course, you will learn:

1. **Understand transfer learning principles and applications**
2. **Master PyTorch deep learning framework**
3. **Use pre-trained models for image classification**
4. **Compare different model architectures and performance**
5. **Optimize training processes and hyperparameters**
6. **Deploy and apply trained models**

## 📊 Model Performance Comparison

Expected performance on Flowers-102 dataset:

| Model | Parameters | Training Time | Accuracy |
|-------|------------|---------------|----------|
| ResNet18 | 11.7M | ~20 min | ~85% |
| ResNet50 | 25.6M | ~35 min | ~88% |
| EfficientNet-B0 | 5.3M | ~25 min | ~90% |
| EfficientNet-B3 | 12.2M | ~40 min | ~92% |
| MobileNet-V2 | 3.5M | ~15 min | ~83% |

*Performance may vary based on hardware and training settings*

## 🔧 Troubleshooting

### Installation Issues

1. **Python Version**: Ensure Python 3.8+
2. **Network Issues**: Check internet connection
3. **Permission Issues**: Run in administrator mode
4. **Reinstall**: Delete `transfer_learning_env` folder and run setup again

### Cannot Start Jupyter?

1. Ensure virtual environment is activated
2. Run `jupyter --version` to check installation
3. Try `jupyter notebook --no-browser` to start

### Slow Training?

1. **Use GPU**: Install CUDA version of PyTorch
2. **Reduce batch size**: If memory insufficient
3. **Use smaller models**: Like MobileNet-V2

## 🔬 Advanced Features

- **Automatic dataset download**: No manual data preparation needed
- **Progress tracking**: Real-time training monitoring
- **Model comparison**: Side-by-side architecture analysis
- **Visualization tools**: Training curves and predictions
- **Export capabilities**: Save and load trained models

## 📚 Documentation

**📑 Documentation Index:**
- **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Complete documentation index

**⚙️ Setup Guides:**
- **[INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)** - Installation Guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick Start Guide

## 📞 Getting Help

- **Issues**: Submit on GitHub
- **Discussions**: Check course discussion board
- **Documentation**: Read detailed lesson instructions and tutorials above

## 🎉 Start Learning

Everything is ready! Open `lessons/lesson1_environment_setup/setup_check.ipynb` to begin your deep learning journey!

---

**Happy Learning!** 🚀

> 💡 Tip: Follow lessons in order, each includes detailed explanations and code examples.

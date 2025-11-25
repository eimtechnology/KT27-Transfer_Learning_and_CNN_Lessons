"""
Global Configuration for Transfer Learning Course
Centralized settings for all lessons
"""
import os
import torch
from pathlib import Path

# Project Structure
PROJECT_ROOT = Path(__file__).parent.parent
LESSONS_DIR = PROJECT_ROOT / "lessons"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SHARED_DIR = PROJECT_ROOT / "shared"

# Create directories if they don't exist
for directory in [DATA_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(exist_ok=True)

# Dataset Configuration
DATASET_CONFIG = {
    "flowers102": {
        "name": "Oxford Flowers-102",
        "num_classes": 102,
        "image_size": 224,
        "url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "path": DATA_DIR / "flowers-102",
        "description": "102 flower categories dataset for transfer learning"
    },
    "default": "flowers102"  # Default dataset for the course
}

# Model Configuration
MODEL_CONFIG = {
    "resnet18": {
        "name": "ResNet-18",
        "pretrained": True,
        "feature_size": 512,
        "description": "Lightweight ResNet for quick prototyping"
    },
    "resnet50": {
        "name": "ResNet-50", 
        "pretrained": True,
        "feature_size": 2048,
        "description": "Standard ResNet with deeper architecture"
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "pretrained": True,
        "feature_size": 1280,
        "description": "Efficient baseline model"
    },
    "efficientnet_b3": {
        "name": "EfficientNet-B3",
        "pretrained": True,
        "feature_size": 1536,
        "description": "Larger EfficientNet with better accuracy"
    },
    "mobilenet_v2": {
        "name": "MobileNet-V2",
        "pretrained": True,
        "feature_size": 1280,
        "description": "Mobile-optimized lightweight model"
    }
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": {
        "small": 16,    # For limited memory
        "medium": 32,   # Standard batch size
        "large": 64     # For high-end GPUs
    },
    "learning_rates": {
        "feature_extraction": 1e-3,    # For frozen backbone
        "fine_tuning": 1e-4,          # For full model training
        "head_only": 1e-2             # For classifier head only
    },
    "epochs": {
        "quick_test": 2,      # For testing/debugging
        "standard": 10,       # Standard training
        "full_training": 25   # Complete training
    }
}

# Hardware Detection
def get_device():
    """Detect and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return {
            "device": device,
            "name": device_name,
            "type": "cuda",
            "memory_gb": memory_gb,
            "description": f"NVIDIA GPU: {device_name} ({memory_gb:.1f}GB)"
        }
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return {
            "device": torch.device("mps"),
            "name": "Apple Silicon GPU",
            "type": "mps", 
            "memory_gb": None,  # MPS uses unified memory
            "description": "Apple Silicon GPU (MPS)"
        }
    else:
        return {
            "device": torch.device("cpu"),
            "name": "CPU",
            "type": "cpu",
            "memory_gb": None,
            "description": "CPU (no GPU acceleration)"
        }

# Auto-detect device
DEVICE_INFO = get_device()

# Visualization Configuration
PLOT_CONFIG = {
    "figsize": {
        "small": (8, 6),
        "medium": (12, 8),
        "large": (16, 10)
    },
    "dpi": 100,
    "style": "default",
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "warning": "#d62728",
        "info": "#9467bd"
    }
}

# Testing Configuration
TEST_CONFIG = {
    "tolerance": {
        "loose": 1e-3,
        "normal": 1e-4,
        "strict": 1e-6
    },
    "timeout": {
        "quick": 30,     # 30 seconds
        "normal": 300,   # 5 minutes
        "long": 1800     # 30 minutes
    },
    "performance_thresholds": {
        "accuracy": {
            "minimum": 0.70,    # 70% minimum accuracy
            "good": 0.80,       # 80% good performance
            "excellent": 0.90   # 90% excellent performance
        },
        "training_time": {
            "quick": 60,        # 1 minute per epoch
            "normal": 300,      # 5 minutes per epoch
            "slow": 600         # 10 minutes per epoch
        }
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": EXPERIMENTS_DIR / "course.log"
}

# Course Metadata
COURSE_INFO = {
    "name": "Transfer Learning with PyTorch",
    "version": "2.0", 
    "description": "Comprehensive course on transfer learning for computer vision",
    "jupyter_kernel": "transfer_learning",
    "venv_name": "dl_course_env",
    "lessons": [
        "lesson0_environment_setup",
        "lesson2_data_exploration", 
        "lesson3_resnet18",
        "lesson4_resnet50",
        "lesson5_efficientnet_b0",
        "lesson6_efficientnet_b3",
        "lesson7_mobilenet_v2"
    ],
    "estimated_time": "8-10 hours",
    "prerequisites": ["Python basics", "Basic machine learning concepts"],
    "learning_outcomes": [
        "Master transfer learning concepts and techniques",
        "Implement state-of-the-art CNN architectures",
        "Compare and select models for different use cases",
        "Deploy optimized models for production"
    ]
}

# Export commonly used configurations
__all__ = [
    'PROJECT_ROOT', 'LESSONS_DIR', 'DATA_DIR', 'EXPERIMENTS_DIR',
    'DATASET_CONFIG', 'MODEL_CONFIG', 'TRAINING_CONFIG',
    'DEVICE_INFO', 'get_device',
    'PLOT_CONFIG', 'TEST_CONFIG', 'LOGGING_CONFIG',
    'COURSE_INFO'
]
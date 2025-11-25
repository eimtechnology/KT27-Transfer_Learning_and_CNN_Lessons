"""
Common Utilities for Transfer Learning Course
Shared functions used across all lessons
"""

import os
import time
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from .config import DEVICE_INFO, PLOT_CONFIG, MODEL_CONFIG, EXPERIMENTS_DIR


def setup_plotting():
    """Configure matplotlib and seaborn for consistent plots"""
    plt.style.use(PLOT_CONFIG["style"])
    plt.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]
    sns.set_palette("husl")
    

def print_section_header(title: str, width: int = 60):
    """Print a formatted section header"""
    border = "=" * width
    try:
        print(f"\n{border}")
        print(f">>> {title}")
        print(f"{border}")
    except UnicodeEncodeError:
        print(f"\n{border}")
        print(f">>> {title}")
        print(f"{border}")


def print_device_info():
    """Display current device information"""
    device_info = DEVICE_INFO
    try:
        print(f"Device: {device_info['device']}")
        print(f"Type: {device_info['description']}")
        if device_info.get('memory_gb'):
            print(f"Memory: {device_info['memory_gb']:.1f} GB")
    except UnicodeEncodeError:
        print(f"Device: {device_info['device']}")
        print(f"Type: {device_info['description']}")
        if device_info.get('memory_gb'):
            print(f"Memory: {device_info['memory_gb']:.1f} GB")


def save_experiment_results(lesson_name: str, results: Dict[str, Any]):
    """Save experiment results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{lesson_name}_{timestamp}.json"
    filepath = EXPERIMENTS_DIR / filename
    
    # Add metadata
    results["metadata"] = {
        "lesson": lesson_name,
        "timestamp": timestamp,
        "device": str(DEVICE_INFO["device"]),
        "device_name": DEVICE_INFO["name"]
    }
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    try:
        print(f"Results saved to: {filepath}")
    except UnicodeEncodeError:
        print(f"Results saved to: {filepath}")
    return filepath


def load_latest_results(lesson_name: str) -> Optional[Dict[str, Any]]:
    """Load the most recent results for a lesson"""
    pattern = f"{lesson_name}_*.json"
    result_files = list(EXPERIMENTS_DIR.glob(pattern))
    
    if not result_files:
        return None
    
    # Get most recent file
    latest_file = max(result_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def create_model_from_config(model_name: str, num_classes: int) -> nn.Module:
    """Create a model based on configuration"""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIG[model_name]
    
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=config["pretrained"])
            model.fc = nn.Linear(config["feature_size"], num_classes)
        elif model_name == "resnet50":
            model = torchvision.models.resnet50(pretrained=config["pretrained"])
            model.fc = nn.Linear(config["feature_size"], num_classes)
    
    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            model = torchvision.models.efficientnet_b0(pretrained=config["pretrained"])
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(config["feature_size"], num_classes)
            )
        elif model_name == "efficientnet_b3":
            model = torchvision.models.efficientnet_b3(pretrained=config["pretrained"])
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(config["feature_size"], num_classes)
            )
    
    elif model_name == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=config["pretrained"])
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config["feature_size"], num_classes)
        )
    
    else:
        raise ValueError(f"Model creation not implemented for: {model_name}")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def set_model_training_mode(model: nn.Module, mode: str):
    """Set model training mode (feature_extraction, fine_tuning, full_training)"""
    if mode == "feature_extraction":
        # Freeze all layers except final classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layer(s)
        if hasattr(model, 'fc'):  # ResNet
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):  # EfficientNet, MobileNet
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    elif mode == "fine_tuning":
        # Unfreeze last few layers
        # Implementation depends on specific architecture
        for param in model.parameters():
            param.requires_grad = True
        
        # Optionally freeze early layers with lower learning rates
        # This would require different learning rates for different layers
    
    elif mode == "full_training":
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    
    else:
        raise ValueError(f"Unknown training mode: {mode}")


def benchmark_model(model: nn.Module, input_size: Tuple[int, ...], 
                   device: torch.device, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed"""
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size, device=device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "fps": 1000.0 / np.mean(times)
    }


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None):
    """Plot training and validation metrics"""
    setup_plotting()
    
    metrics = list(history.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].plot(history[metric], linewidth=2, label=f'Training {metric}')
        axes[i].set_title(f'{metric.title()} History')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.title())
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def compare_models(models_info: List[Dict[str, Any]], save_path: Optional[Path] = None):
    """Create comparison visualization of multiple models"""
    setup_plotting()
    
    model_names = [info['name'] for info in models_info]
    metrics = ['accuracy', 'parameters', 'inference_time']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    
    # Accuracy comparison
    accuracies = [info.get('accuracy', 0) for info in models_info]
    axes[0].bar(model_names, accuracies, color=PLOT_CONFIG['colors']['primary'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Parameters comparison (in millions)
    params = [info.get('parameters', 0) / 1e6 for info in models_info]
    axes[1].bar(model_names, params, color=PLOT_CONFIG['colors']['secondary'])
    axes[1].set_title('Model Size Comparison')
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Inference time comparison
    times = [info.get('inference_time_ms', 0) for info in models_info]
    axes[2].bar(model_names, times, color=PLOT_CONFIG['colors']['success'])
    axes[2].set_title('Inference Speed Comparison')
    axes[2].set_ylabel('Time (ms)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()


def create_confusion_matrix(y_true: List[int], y_pred: List[int], 
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[Path] = None):
    """Create and display confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    setup_plotting()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


class ProgressTracker:
    """Track and display training progress"""
    
    def __init__(self, total_epochs: int, metrics: List[str]):
        self.total_epochs = total_epochs
        self.metrics = metrics
        self.history = {metric: [] for metric in metrics}
        self.current_epoch = 0
        self.start_time = time.time()
    
    def update(self, epoch: int, **metric_values):
        """Update progress with new metric values"""
        self.current_epoch = epoch
        
        for metric, value in metric_values.items():
            if metric in self.history:
                self.history[metric].append(value)
    
    def display_progress(self):
        """Display current training progress"""
        elapsed = time.time() - self.start_time
        eta = (elapsed / (self.current_epoch + 1)) * (self.total_epochs - self.current_epoch - 1)
        
        print(f"Epoch {self.current_epoch + 1}/{self.total_epochs}")
        print(f"Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        
        # Display latest metrics
        for metric in self.metrics:
            if self.history[metric]:
                latest_value = self.history[metric][-1]
                print(f"{metric}: {latest_value:.4f}")
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.history.copy()


# Export commonly used functions
__all__ = [
    'setup_plotting', 'print_section_header', 'print_device_info',
    'save_experiment_results', 'load_latest_results',
    'create_model_from_config', 'count_parameters', 'set_model_training_mode',
    'benchmark_model', 'plot_training_history', 'compare_models', 
    'create_confusion_matrix', 'ProgressTracker'
]
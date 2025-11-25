# EfficientNet-B3 Transfer Learning Configuration
# Lesson 6: Compound Scaling for Enhanced Flower Classification

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'efficientnet_b3',
    'pretrained': True,
    'num_classes': 102,
    'input_size': 300,  # Higher resolution than B0
    'total_parameters': 12231232,  # ~12.2M parameters
    'trainable_parameters_phase1': 156876,  # Only classifier in Phase 1
    'trainable_parameters_phase2': 12231232,  # All parameters in Phase 2
    'model_size_mb': 48.9,  # Approximate model size in MB
    'flops': 1.8e9,  # 1.8 GFLOPs
    'imagenet_accuracy': 81.6,  # ImageNet Top-1 accuracy
    'dropout_rate': 0.3,  # Higher dropout for B3
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 24,  # Reduced from 32 due to higher resolution
    'learning_rate': 0.0008,  # Slightly lower than B0 for stability
    'weight_decay': 0.01,
    'optimizer': 'AdamW',
    'loss_function': 'CrossEntropyLoss',
    'device': 'auto',  # 'cuda', 'mps', 'cpu', or 'auto'
    'num_workers': 2,
    'pin_memory': True,
    'gradient_clip_value': None,  # Optional gradient clipping
}

# Two-Phase Training Strategy
PHASE_CONFIG = {
    'phase1': {
        'epochs': 20,
        'freeze_features': True,
        'train_classifier_only': True,
        'description': 'Feature extraction with frozen backbone',
        'expected_accuracy': 82.0,  # Higher than B0 due to better features
    },
    'phase2': {
        'epochs': 30,
        'freeze_features': False,
        'train_all_layers': True,
        'description': 'End-to-end fine-tuning',
        'expected_accuracy': 92.0,  # Higher than B0 due to compound scaling
    }
}

# Data Preprocessing Configuration
DATA_CONFIG = {
    'dataset': 'Flowers102',
    'train_transforms': transforms.Compose([
        transforms.Resize((320, 320)),  # Larger resize for B3
        transforms.RandomCrop(300),     # B3 native resolution
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val_transforms': transforms.Compose([
        transforms.Resize((300, 300)),  # Direct resize for validation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'normalization': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'source': 'ImageNet'
    }
}

# Performance Expectations
PERFORMANCE_CONFIG = {
    'expected_results': {
        'phase1_accuracy': 82.0,
        'phase2_accuracy': 92.0,
        'training_time_minutes': 22,
        'epochs_to_convergence': 40,
    },
    'comparison_with_other_models': {
        'resnet18': {
            'parameters': 11689512,
            'accuracy': 85.0,
            'efficiency_ratio': 7.27,  # accuracy / M parameters
        },
        'resnet50': {
            'parameters': 25557032,
            'accuracy': 88.0,
            'efficiency_ratio': 3.44,  # accuracy / M parameters
        },
        'efficientnet_b0': {
            'parameters': 5288548,
            'accuracy': 90.0,
            'efficiency_ratio': 17.02,  # accuracy / M parameters
        },
        'efficientnet_b3': {
            'parameters': 12231232,
            'accuracy': 92.0,
            'efficiency_ratio': 7.52,  # accuracy / M parameters
        }
    }
}

# Compound Scaling Configuration
COMPOUND_SCALING_CONFIG = {
    'scaling_coefficients': {
        'alpha': 1.2,  # Depth scaling
        'beta': 1.1,   # Width scaling
        'gamma': 1.15, # Resolution scaling
        'phi': 3,      # Compound coefficient for B3
    },
    'scaling_results': {
        'depth_multiplier': 1.728,    # α^φ = 1.2^3
        'width_multiplier': 1.331,    # β^φ = 1.1^3
        'resolution_multiplier': 1.520, # γ^φ = 1.15^3
    },
    'b0_to_b3_comparison': {
        'depth': {'b0': 18, 'b3': 26, 'ratio': 1.44},
        'width': {'b0': 1.0, 'b3': 1.33, 'ratio': 1.33},
        'resolution': {'b0': 224, 'b3': 300, 'ratio': 1.34},
        'parameters': {'b0': 5.3, 'b3': 12.2, 'ratio': 2.30},
        'flops': {'b0': 0.39, 'b3': 1.8, 'ratio': 4.62},
        'imagenet_accuracy': {'b0': 77.3, 'b3': 81.6, 'improvement': 4.3},
    }
}

# Architecture Configuration
ARCHITECTURE_CONFIG = {
    'mbconv_blocks': {
        'description': 'Mobile Inverted Bottleneck Convolution (scaled)',
        'stages': [
            {'stage': 1, 'blocks': 2, 'channels': 24, 'stride': 1, 'kernel': 3},
            {'stage': 2, 'blocks': 3, 'channels': 32, 'stride': 2, 'kernel': 3},
            {'stage': 3, 'blocks': 4, 'channels': 48, 'stride': 2, 'kernel': 5},
            {'stage': 4, 'blocks': 5, 'channels': 96, 'stride': 2, 'kernel': 3},
            {'stage': 5, 'blocks': 6, 'channels': 136, 'stride': 1, 'kernel': 5},
            {'stage': 6, 'blocks': 7, 'channels': 232, 'stride': 2, 'kernel': 5},
            {'stage': 7, 'blocks': 2, 'channels': 384, 'stride': 1, 'kernel': 3},
        ]
    },
    'activation_function': 'Swish',  # f(x) = x * sigmoid(x)
    'normalization': 'BatchNorm2d',
    'se_attention': True,  # Squeeze-and-Excitation blocks
    'stem_channels': 40,   # Scaled from 32 in B0
    'head_channels': 1536, # Scaled from 1280 in B0
}

# Memory Management Configuration
MEMORY_CONFIG = {
    'optimization_settings': {
        'batch_size_reduction': 'From 32 to 24 due to higher resolution',
        'gradient_checkpointing': False,  # Can be enabled if needed
        'mixed_precision': False,  # Can be enabled for modern GPUs
        'memory_efficient_attention': True,  # SE blocks are memory efficient
    },
    'memory_usage_comparison': {
        'b0_memory_mb': 1536,  # Approximate GPU memory usage
        'b3_memory_mb': 2560,  # Approximate GPU memory usage
        'memory_ratio': 1.67,  # B3 uses 67% more memory
    }
}

# Utility Functions
def get_model():
    """Initialize EfficientNet-B3 model for transfer learning"""
    model = models.efficientnet_b3(pretrained=MODEL_CONFIG['pretrained'])
    
    # Replace classifier with higher dropout
    model.classifier = nn.Sequential(
        nn.Dropout(MODEL_CONFIG['dropout_rate']),
        nn.Linear(model.classifier[1].in_features, MODEL_CONFIG['num_classes'])
    )
    
    return model

def get_optimizer_phase1(model):
    """Get optimizer for Phase 1 (classifier only)"""
    return optim.AdamW(
        model.classifier.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

def get_optimizer_phase2(model):
    """Get optimizer for Phase 2 (all parameters)"""
    return optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

def freeze_features(model):
    """Freeze feature extraction layers"""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_all(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def get_device():
    """Get optimal device for training"""
    if TRAINING_CONFIG['device'] == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(TRAINING_CONFIG['device'])

def calculate_efficiency_ratio(accuracy, parameters_millions):
    """Calculate efficiency ratio (accuracy per million parameters)"""
    return accuracy / parameters_millions

# Model Summary
def print_model_summary():
    """Print comprehensive model summary"""
    print("=" * 70)
    print("EfficientNet-B3 Transfer Learning Configuration")
    print("=" * 70)
    
    print(f"Model Architecture: {MODEL_CONFIG['architecture']}")
    print(f"Total Parameters: {MODEL_CONFIG['total_parameters']:,}")
    print(f"Model Size: {MODEL_CONFIG['model_size_mb']:.1f}MB")
    print(f"FLOPs: {MODEL_CONFIG['flops']:.2e}")
    print(f"ImageNet Accuracy: {MODEL_CONFIG['imagenet_accuracy']:.1f}%")
    print(f"Input Resolution: {MODEL_CONFIG['input_size']}×{MODEL_CONFIG['input_size']}")
    
    print("\nTraining Configuration:")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']} (reduced from 32)")
    print(f"  Learning Rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  Optimizer: {TRAINING_CONFIG['optimizer']}")
    print(f"  Total Epochs: {TRAINING_CONFIG['num_epochs']}")
    
    print("\nTwo-Phase Strategy:")
    print(f"  Phase 1: {PHASE_CONFIG['phase1']['description']}")
    print(f"    Epochs: {PHASE_CONFIG['phase1']['epochs']}")
    print(f"    Expected Accuracy: {PHASE_CONFIG['phase1']['expected_accuracy']:.1f}%")
    print(f"  Phase 2: {PHASE_CONFIG['phase2']['description']}")
    print(f"    Epochs: {PHASE_CONFIG['phase2']['epochs']}")
    print(f"    Expected Accuracy: {PHASE_CONFIG['phase2']['expected_accuracy']:.1f}%")
    
    print("\nCompound Scaling Analysis:")
    scaling = COMPOUND_SCALING_CONFIG['b0_to_b3_comparison']
    print(f"  Depth: {scaling['depth']['b0']} → {scaling['depth']['b3']} layers ({scaling['depth']['ratio']:.2f}×)")
    print(f"  Width: {scaling['width']['b0']:.1f} → {scaling['width']['b3']:.2f}× channels")
    print(f"  Resolution: {scaling['resolution']['b0']} → {scaling['resolution']['b3']} pixels")
    print(f"  Parameters: {scaling['parameters']['b0']:.1f}M → {scaling['parameters']['b3']:.1f}M ({scaling['parameters']['ratio']:.2f}×)")
    print(f"  ImageNet Accuracy: {scaling['imagenet_accuracy']['b0']:.1f}% → {scaling['imagenet_accuracy']['b3']:.1f}% (+{scaling['imagenet_accuracy']['improvement']:.1f}%)")
    
    print("\nEfficiency Comparison:")
    for model_name, stats in PERFORMANCE_CONFIG['comparison_with_other_models'].items():
        print(f"  {model_name}: {stats['accuracy']:.1f}% accuracy, "
              f"{stats['efficiency_ratio']:.1f}% per M params")
    
    print("\nMemory Usage:")
    mem = MEMORY_CONFIG['memory_usage_comparison']
    print(f"  B0 Memory: {mem['b0_memory_mb']}MB")
    print(f"  B3 Memory: {mem['b3_memory_mb']}MB ({mem['memory_ratio']:.2f}× more)")
    
    print("=" * 70)

if __name__ == "__main__":
    print_model_summary() 
# EfficientNet-B0 Transfer Learning Configuration
# Lesson 5: Efficient Neural Architecture for Flower Classification

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'efficientnet_b0',
    'pretrained': True,
    'num_classes': 102,
    'input_size': 224,
    'total_parameters': 5288548,  # ~5.3M parameters
    'trainable_parameters_phase1': 104346,  # Only classifier in Phase 1
    'trainable_parameters_phase2': 5288548,  # All parameters in Phase 2
    'model_size_mb': 21.2,  # Approximate model size in MB
    'flops': 0.39e9,  # 0.39 GFLOPs
    'imagenet_accuracy': 77.3,  # ImageNet Top-1 accuracy
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
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
        'expected_accuracy': 80.0,
    },
    'phase2': {
        'epochs': 30,
        'freeze_features': False,
        'train_all_layers': True,
        'description': 'End-to-end fine-tuning',
        'expected_accuracy': 90.0,
    }
}

# Data Preprocessing Configuration
DATA_CONFIG = {
    'dataset': 'Flowers102',
    'train_transforms': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val_transforms': transforms.Compose([
        transforms.Resize((224, 224)),
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
        'phase1_accuracy': 80.0,
        'phase2_accuracy': 90.0,
        'training_time_minutes': 12,
        'epochs_to_convergence': 35,
    },
    'comparison_with_resnet': {
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
            'efficiency_ratio': 17.02,  # accuracy / M parameters (best!)
        }
    }
}

# EfficientNet Architecture Details
ARCHITECTURE_CONFIG = {
    'compound_scaling': {
        'description': 'Uniform scaling of depth, width, and resolution',
        'scaling_coefficients': {
            'depth': 1.0,    # α = 1.0 for B0
            'width': 1.0,    # β = 1.0 for B0  
            'resolution': 1.0 # γ = 1.0 for B0
        }
    },
    'mbconv_blocks': {
        'description': 'Mobile Inverted Bottleneck Convolution',
        'stages': [
            {'stage': 1, 'blocks': 1, 'channels': 16, 'stride': 1, 'kernel': 3},
            {'stage': 2, 'blocks': 2, 'channels': 24, 'stride': 2, 'kernel': 3},
            {'stage': 3, 'blocks': 2, 'channels': 40, 'stride': 2, 'kernel': 5},
            {'stage': 4, 'blocks': 3, 'channels': 80, 'stride': 2, 'kernel': 3},
            {'stage': 5, 'blocks': 3, 'channels': 112, 'stride': 1, 'kernel': 5},
            {'stage': 6, 'blocks': 4, 'channels': 192, 'stride': 2, 'kernel': 5},
            {'stage': 7, 'blocks': 1, 'channels': 320, 'stride': 1, 'kernel': 3},
        ]
    },
    'activation_function': 'Swish',  # f(x) = x * sigmoid(x)
    'normalization': 'BatchNorm2d',
    'dropout_rate': 0.2,
    'se_attention': True,  # Squeeze-and-Excitation blocks
}

# Utility Functions
def get_model():
    """Initialize EfficientNet-B0 model for transfer learning"""
    model = models.efficientnet_b0(pretrained=MODEL_CONFIG['pretrained'])
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(ARCHITECTURE_CONFIG['dropout_rate']),
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

# Model Summary
def print_model_summary():
    """Print comprehensive model summary"""
    print("=" * 60)
    print("EfficientNet-B0 Transfer Learning Configuration")
    print("=" * 60)
    
    print(f"Model Architecture: {MODEL_CONFIG['architecture']}")
    print(f"Total Parameters: {MODEL_CONFIG['total_parameters']:,}")
    print(f"Model Size: {MODEL_CONFIG['model_size_mb']:.1f}MB")
    print(f"FLOPs: {MODEL_CONFIG['flops']:.2e}")
    print(f"ImageNet Accuracy: {MODEL_CONFIG['imagenet_accuracy']:.1f}%")
    
    print("\nTraining Configuration:")
    print(f"  Batch Size: {TRAINING_CONFIG['batch_size']}")
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
    
    print("\nEfficiency Comparison:")
    for model_name, stats in PERFORMANCE_CONFIG['comparison_with_resnet'].items():
        print(f"  {model_name}: {stats['accuracy']:.1f}% accuracy, "
              f"{stats['efficiency_ratio']:.1f}% per M params")
    
    print("=" * 60)

if __name__ == "__main__":
    print_model_summary() 
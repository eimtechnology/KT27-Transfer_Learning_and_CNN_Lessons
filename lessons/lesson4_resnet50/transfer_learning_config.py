# ResNet50 Transfer Learning Configuration
# Lesson 4: ResNet50 Transfer Learning for Flower Classification

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Model Configuration
MODEL_CONFIG = {
    'name': 'ResNet50',
    'architecture': 'resnet50',
    'layers': 50,
    'parameters': 25_557_032,  # Total parameters
    'pretrained': True,
    'num_classes': 102,  # Flowers102 classes
    'input_size': (224, 224),
    'channels': 3
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,  # May need reduction for memory limits
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 50,
    'freeze_epochs': 20,
    'finetune_epochs': 30,
    'optimizer': 'AdamW',
    'criterion': 'CrossEntropyLoss',
    'device': 'auto',  # Auto-detect best device
    'seed': 42
}

# Data Configuration
DATA_CONFIG = {
    'dataset': 'Flowers102',
    'data_dir': './data',
    'num_workers': 2,
    'pin_memory': True,
    'persistent_workers': True,
    'download': True
}

# ImageNet normalization (required for pre-trained models)
IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'train_transforms': [
        ('Resize', (256, 256)),
        ('RandomCrop', 224),
        ('RandomHorizontalFlip', 0.5),
        ('RandomRotation', 15),
        ('ColorJitter', {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }),
        ('ToTensor', None),
        ('Normalize', IMAGENET_STATS)
    ],
    'val_transforms': [
        ('Resize', (224, 224)),
        ('ToTensor', None),
        ('Normalize', IMAGENET_STATS)
    ]
}

# Performance Expectations
PERFORMANCE_EXPECTATIONS = {
    'phase1_accuracy': 78.0,  # Feature extraction accuracy
    'phase2_accuracy': 88.0,  # Fine-tuning accuracy
    'test_accuracy': 86.0,    # Expected test accuracy
    'training_time_minutes': 25,  # Expected training time
    'memory_usage_gb': 3.5,   # Expected GPU memory usage
    'resnet18_comparison': {
        'accuracy_improvement': 3.0,  # % improvement over ResNet18
        'parameter_ratio': 2.2,       # Parameter increase
        'training_time_ratio': 1.5    # Training time increase
    }
}

# Memory Management Configuration
MEMORY_CONFIG = {
    'recommended_gpu_memory': 4.0,  # GB
    'fallback_batch_sizes': [16, 8],  # If memory issues
    'enable_memory_cleanup': True,
    'gradient_checkpointing': False,  # Not needed for ResNet50
    'mixed_precision': False  # Can be enabled for speed
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_interval': 10,  # Log every N batches
    'save_checkpoints': True,
    'checkpoint_dir': './checkpoints',
    'log_metrics': [
        'train_loss', 'train_accuracy',
        'val_loss', 'val_accuracy',
        'epoch_time', 'memory_usage'
    ]
}

def get_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_model(num_classes=102):
    """Get ResNet50 model with modified classifier"""
    import torchvision.models as models
    
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_optimizer(model, config=None):
    """Get AdamW optimizer with configured parameters"""
    if config is None:
        config = TRAINING_CONFIG
    
    return optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

def get_criterion():
    """Get CrossEntropyLoss criterion"""
    return nn.CrossEntropyLoss()

def get_transforms():
    """Get training and validation transforms"""
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_STATS['mean'],
            std=IMAGENET_STATS['std']
        )
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_STATS['mean'],
            std=IMAGENET_STATS['std']
        )
    ])
    
    return train_transforms, val_transforms

def set_parameter_requires_grad(model, feature_extracting):
    """Freeze/unfreeze model parameters for two-phase training"""
    if feature_extracting:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze only the classifier
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

# Configuration validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if TRAINING_CONFIG['batch_size'] < 1:
        errors.append("Batch size must be positive")
    
    if TRAINING_CONFIG['learning_rate'] <= 0:
        errors.append("Learning rate must be positive")
    
    if TRAINING_CONFIG['freeze_epochs'] + TRAINING_CONFIG['finetune_epochs'] != TRAINING_CONFIG['epochs']:
        errors.append("Freeze epochs + finetune epochs must equal total epochs")
    
    if len(errors) > 0:
        raise ValueError("Configuration errors: " + ", ".join(errors))
    
    return True

# Export main configuration
CONFIG = {
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'data': DATA_CONFIG,
    'augmentation': AUGMENTATION_CONFIG,
    'performance': PERFORMANCE_EXPECTATIONS,
    'memory': MEMORY_CONFIG,
    'logging': LOGGING_CONFIG
}

if __name__ == "__main__":
    # Validate configuration on import
    validate_config()
    print("✅ ResNet50 configuration validated successfully!")
    
    # Display key configuration
    print(f"\n📊 ResNet50 Configuration Summary:")
    print(f"   🏗️  Model: {MODEL_CONFIG['name']} ({MODEL_CONFIG['parameters']:,} parameters)")
    print(f"   📦 Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   🎯 Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   🔄 Epochs: {TRAINING_CONFIG['epochs']} (freeze: {TRAINING_CONFIG['freeze_epochs']}, finetune: {TRAINING_CONFIG['finetune_epochs']})")
    print(f"   🎯 Expected accuracy: {PERFORMANCE_EXPECTATIONS['test_accuracy']:.1f}%")
    print(f"   ⏱️  Expected training time: {PERFORMANCE_EXPECTATIONS['training_time_minutes']} minutes")
    print(f"   💾 Expected memory usage: {PERFORMANCE_EXPECTATIONS['memory_usage_gb']} GB") 
# Lesson 4: ResNet50 Transfer Learning for Flower Classification

## Overview
This lesson explores ResNet50, a deeper and more powerful variant of the ResNet architecture. We'll compare its performance with ResNet18 from Lesson 3 and understand how increased depth affects transfer learning performance.

## Learning Objectives
- Understand ResNet50 architecture and its differences from ResNet18
- Implement transfer learning with a deeper network
- Compare performance between ResNet18 and ResNet50
- Analyze computational trade-offs between model depth and performance

## ResNet50 vs ResNet18 Comparison

### Architecture Differences

| Feature | ResNet18 | ResNet50 |
|---------|----------|----------|
| **Layers** | 18 layers | 50 layers |
| **Parameters** | 11.7M | 25.6M |
| **Residual Blocks** | Basic blocks | Bottleneck blocks |
| **Block Structure** | 3x3 + 3x3 conv | 1x1 + 3x3 + 1x1 conv |
| **Depth per Stage** | [2,2,2,2] | [3,4,6,3] |
| **Model Size** | ~45MB | ~98MB |

### ResNet50 Architecture Details

ResNet50 introduces **bottleneck blocks** that are more efficient for deeper networks:

#### Bottleneck Block Structure
```
Input → [1×1 Conv, 64] → [3×3 Conv, 64] → [1×1 Conv, 256] → Output
   ↓                                                           ↑
   └─────────────────── Skip Connection ──────────────────────┘
```

#### Key Innovations
- **1×1 Convolutions**: Reduce computational complexity
- **Channel Reduction**: First 1×1 conv reduces channels (256→64)
- **Channel Expansion**: Last 1×1 conv expands channels (64→256)
- **Computational Efficiency**: 3×3 conv operates on fewer channels

### Network Architecture

#### Stage-by-Stage Breakdown
```
Stage 0: Input Processing
├── 7×7 Conv, 64 filters, stride 2
├── Batch Normalization
├── ReLU Activation
└── 3×3 Max Pooling, stride 2

Stage 1: conv2_x (64 → 256 channels)
├── Bottleneck Block × 3
├── First block: stride 1, projection shortcut
└── Remaining blocks: stride 1, identity shortcut

Stage 2: conv3_x (256 → 512 channels)
├── Bottleneck Block × 4
├── First block: stride 2, projection shortcut
└── Remaining blocks: stride 1, identity shortcut

Stage 3: conv4_x (512 → 1024 channels)
├── Bottleneck Block × 6
├── First block: stride 2, projection shortcut
└── Remaining blocks: stride 1, identity shortcut

Stage 4: conv5_x (1024 → 2048 channels)
├── Bottleneck Block × 3
├── First block: stride 2, projection shortcut
└── Remaining blocks: stride 1, identity shortcut

Output Layer:
├── Global Average Pooling
├── Fully Connected Layer (2048 → 1000)
└── Softmax (for ImageNet)
```

## Transfer Learning Strategy

### Why ResNet50 for Transfer Learning?

**Advantages:**
- **Richer Features**: Deeper network captures more complex patterns
- **Better Generalization**: More layers provide better feature hierarchy
- **Proven Performance**: State-of-the-art results on many vision tasks
- **Stable Training**: Residual connections enable stable deep training

**Considerations:**
- **More Parameters**: 25.6M vs 11.7M (2.2× larger)
- **Slower Training**: More computation per forward/backward pass
- **Memory Usage**: Higher GPU memory requirements
- **Overfitting Risk**: More parameters may overfit on small datasets

### Training Strategy

We use the same two-phase approach as ResNet18:

#### Phase 1: Feature Extraction (Epochs 1-20)
```python
# Freeze all layers except final classifier
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True
```

#### Phase 2: Fine-tuning (Epochs 21-50)
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
```

### Expected Performance

| Metric | ResNet18 | ResNet50 | Improvement |
|--------|----------|----------|-------------|
| **Phase 1 Accuracy** | ~75% | ~78% | +3% |
| **Phase 2 Accuracy** | ~85% | ~88% | +3% |
| **Training Time** | 15-20 min | 25-30 min | +67% |
| **Memory Usage** | ~2GB | ~3GB | +50% |

## Technical Implementation

### Model Initialization
```python
import torchvision.models as models

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify final layer for 102 flower classes
model.fc = nn.Linear(model.fc.in_features, 102)
```

### Key Parameters
- **Input Size**: 224×224×3 (same as ResNet18)
- **Batch Size**: 32 (may need reduction for memory limits)
- **Learning Rate**: 0.001 (consistent with ResNet18)
- **Optimizer**: AdamW with weight decay
- **Epochs**: 50 (20 frozen + 30 fine-tuning)

### Data Augmentation
Same augmentation strategy as ResNet18:
- Random crop and resize
- Random horizontal flip
- Color jittering
- ImageNet normalization

## Performance Analysis

### Computational Complexity

#### FLOPs Comparison
- **ResNet18**: ~1.8 GFLOPs
- **ResNet50**: ~4.1 GFLOPs
- **Ratio**: 2.3× more computation

#### Memory Requirements
- **ResNet18**: ~11.7M parameters × 4 bytes = ~47MB
- **ResNet50**: ~25.6M parameters × 4 bytes = ~102MB
- **Ratio**: 2.2× more memory

### When to Use ResNet50

**Choose ResNet50 when:**
- You have sufficient computational resources
- Dataset is large enough to benefit from deeper features
- Maximum accuracy is more important than training speed
- You're working on complex visual recognition tasks

**Choose ResNet18 when:**
- Limited computational resources
- Fast prototyping and experimentation
- Smaller datasets where depth might cause overfitting
- Real-time inference requirements

## Practical Considerations

### Training Tips
1. **Monitor GPU Memory**: May need to reduce batch size
2. **Learning Rate**: Start with same LR as ResNet18
3. **Early Stopping**: Watch for overfitting with deeper model
4. **Data Augmentation**: More important for preventing overfitting

### Debugging Common Issues
- **CUDA Out of Memory**: Reduce batch size to 16 or 8
- **Slow Training**: Ensure using GPU acceleration
- **Poor Convergence**: Check learning rate and data normalization

## Expected Outcomes

After completing this lesson, you should be able to:
- Understand the trade-offs between model depth and performance
- Implement transfer learning with ResNet50
- Compare results with ResNet18 quantitatively
- Make informed decisions about model selection for your use case

## File Structure
```
lesson4_resnet50/
├── README.md                 # This comprehensive guide
├── resnet50_training.ipynb   # Main training notebook
└── transfer_learning_config.py  # Configuration settings
```

## Next Steps
- **Lesson 5**: EfficientNet-B0 (efficient architecture)
- **Lesson 6**: EfficientNet-B3 (scaled efficient architecture)
- **Lesson 7**: MobileNet-V2 (mobile-optimized architecture)
- **Lesson 8**: Model comparison and selection

---

*This lesson builds upon Lesson 3 (ResNet18) and provides deeper insights into how architectural choices affect transfer learning performance.* 
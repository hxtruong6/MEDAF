# Multi-Label Novelty Detection with MEDAF

## Overview

This guide explains how to use the novelty detection functionality that has been added to your MEDAF implementation. The system can detect unknown/novel samples in multi-label classification settings using a hybrid scoring mechanism.

## Key Components

### 1. MultiLabelNoveltyDetector Class

The core novelty detection class that implements the hybrid scoring mechanism described in your guide:

```python
from core.multilabel_novelty_detection import MultiLabelNoveltyDetector

# Create detector
detector = MultiLabelNoveltyDetector(gamma=1.0, temperature=1.0)
```

**Key Methods:**

- `compute_logit_score()`: Computes logit-based novelty score using Joint Energy
- `compute_feature_score()`: Computes feature-based score using CAM diversity
- `compute_hybrid_score()`: Combines both scores for robust detection
- `calibrate_threshold()`: Sets rejection threshold on validation data
- `detect_novelty()`: Main detection function
- `classify_novelty_type()`: Classifies types of novelty (independent, mixed, combinatorial)

### 2. Enhanced MultiLabelMEDAF Class

Your existing MEDAF model now includes novelty detection methods:

```python
from core.multilabel_net import MultiLabelMEDAF

# Load your trained model
model = MultiLabelMEDAF(args)
model.load_state_dict(checkpoint["state_dict"])

# Detect novelty in test samples
novelty_results = model.detect_novelty(inputs, novelty_detector=detector)
```

**New Methods:**

- `detect_novelty()`: Detect novel samples using hybrid scoring
- `calibrate_novelty_detector()`: Calibrate threshold on validation data
- `evaluate_novelty_detection()`: Evaluate detection performance

## How Novelty Detection Works

### 1. Hybrid Scoring Mechanism

The system uses a two-component hybrid score:

**Logit-based Score (S_lg):**

```python
S_lg(x) = -log(sum_k exp(l_g,k / T))
```

- Measures confidence in known label predictions
- Low energy = tight predictions (known samples)
- High energy = dispersed predictions (unknown samples)

**Feature-based Score (S_ft):**

```python
S_ft(x) = (1/|Y_hat|) * sum_{y in Y_hat} ||(1/N) * sum_i M_{i,y}||_2
```

- Uses CAM diversity to detect distributional shifts
- High CAM norms = compact activations (known samples)
- Low CAM norms = dispersed activations (unknown samples)

**Hybrid Score:**

```python
S(x) = S_lg(x) + γ * S_ft(x)
```

- Combines both components for robust detection
- Higher scores indicate more "known-like" samples

### 2. Types of Multi-Label Novelty

The system handles three types of novelty:

1. **Independent Novelty**: Only unknown labels present
   - Example: Image with only novel disease labels
   - Detection: Low confidence across all known labels

2. **Mixed Novelty**: Unknown + known labels (most challenging)
   - Example: Image with known "pneumonia" + unknown "rare condition"
   - Detection: Requires per-label uncertainty analysis

3. **Combinatorial Novelty**: Novel combinations of known labels
   - Example: Known diseases in unprecedented combinations
   - Detection: Advanced dependency modeling (future work)

### 3. Threshold Calibration

The rejection threshold is calibrated on validation data containing only known samples:

```python
# Calibrate detector
detector = model.calibrate_novelty_detector(val_loader, device, fpr_target=0.05)
```

**Process:**

1. Compute hybrid scores for all validation samples
2. Set threshold at target FPR (e.g., 5th percentile for 5% FPR)
3. This ensures most known samples are accepted while unknowns are rejected

## Usage Examples

### Basic Usage

```python
import torch
from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_novelty_detection import MultiLabelNoveltyDetector

# Load your trained model
model = MultiLabelMEDAF(args)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

# Create and calibrate detector
detector = MultiLabelNoveltyDetector(gamma=1.0, temperature=1.0)
detector.calibrate_threshold(model, val_loader, device, fpr_target=0.05)

# Detect novelty in test samples
novelty_results = model.detect_novelty(inputs, novelty_detector=detector)

# Access results
is_novel = novelty_results["is_novel"]  # Boolean tensor [B]
novelty_scores = novelty_results["novelty_scores"]  # Hybrid scores [B]
novelty_types = novelty_results["novelty_types"]  # List of type strings
predictions = novelty_results["predictions"]  # Binary predictions [B, num_classes]
```

### Complete Workflow

```python
# 1. Load model
model = MultiLabelMEDAF(args)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

# 2. Calibrate detector
detector = model.calibrate_novelty_detector(val_loader, device, fpr_target=0.05)

# 3. Detect novelty
for inputs, targets in test_loader:
    inputs = inputs.to(device)
    
    novelty_results = model.detect_novelty(inputs, novelty_detector=detector)
    
    is_novel = novelty_results["is_novel"]
    novelty_scores = novelty_results["novelty_scores"]
    novelty_types = novelty_results["novelty_types"]
    
    # Process results
    for i in range(len(inputs)):
        if is_novel[i]:
            print(f"Sample {i} is novel: {novelty_types[i]}")
            print(f"Novelty score: {novelty_scores[i].item():.4f}")
        else:
            print(f"Sample {i} is known")
```

### Evaluation

```python
# Evaluate novelty detection performance
results = model.evaluate_novelty_detection(
    known_loader, unknown_loader, device, detector
)

print(f"AUROC: {results['auroc']:.4f}")
print(f"Detection Accuracy: {results['detection_accuracy']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

## File Structure

```
aidan-medaf/
├── core/
│   ├── multilabel_novelty_detection.py  # Novelty detection implementation
│   ├── multilabel_net.py                # Enhanced MEDAF model
│   └── ...
├── example_novelty_detection.py          # Complete workflow example
├── test_novelty_detection.py            # Simple test script
└── NOVELTY_DETECTION_GUIDE.md           # This guide
```

## Key Features

### 1. Robust Detection

- Combines logit confidence and feature diversity
- Handles different types of multi-label novelty
- Calibrated thresholds for reliable detection

### 2. Easy Integration

- Seamlessly integrates with existing MEDAF model
- Simple API for novelty detection
- Comprehensive evaluation metrics

### 3. Flexible Configuration

- Adjustable gamma and temperature parameters
- Configurable FPR targets for calibration
- Support for different novelty types

## Parameters

### MultiLabelNoveltyDetector

- `gamma`: Weight for feature-based score (default: 1.0)
- `temperature`: Temperature for logit-based energy (default: 1.0)

### Calibration

- `fpr_target`: Target false positive rate (default: 0.05 for 5% FPR)
- `val_loader`: Validation data loader (known samples only)

### Detection

- `threshold`: Prediction threshold for binary classification (default: 0.5)
- `novelty_detector`: Pre-calibrated detector instance

## Best Practices

### 1. Calibration

- Always calibrate on validation data containing only known samples
- Use appropriate FPR target (5-10% is common)
- Ensure sufficient validation data for reliable calibration

### 2. Parameter Tuning

- Start with default parameters (gamma=1.0, temperature=1.0)
- Adjust gamma to balance logit and feature components
- Use temperature to control logit-based energy sensitivity

### 3. Evaluation

- Use separate known and unknown test sets
- Report AUROC, precision, recall, and F1-score
- Analyze different novelty types separately

### 4. Interpretation

- Higher novelty scores indicate more "known-like" samples
- Lower scores indicate more "novel/unknown" samples
- Mixed novelty is most challenging to detect

## Troubleshooting

### Common Issues

1. **"Detector not calibrated" warning**
   - Solution: Call `calibrate_threshold()` before detection

2. **Low detection performance**
   - Check calibration data quality
   - Adjust gamma and temperature parameters
   - Ensure sufficient training data

3. **Memory issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear GPU cache between operations

### Performance Tips

1. **Batch Processing**
   - Process samples in batches for efficiency
   - Use appropriate batch sizes for your GPU memory

2. **Model Optimization**
   - Use model.eval() for inference
   - Disable gradient computation with torch.no_grad()

3. **Data Loading**
   - Use multiple workers for data loading
   - Pin memory for GPU acceleration

## Future Enhancements

### 1. Advanced Novelty Types

- Combinatorial novelty detection
- Label dependency modeling
- Graph-based novelty detection

### 2. Uncertainty Quantification

- Evidential deep learning integration
- Bayesian uncertainty estimation
- Confidence interval computation

### 3. Adaptive Thresholds

- Dynamic threshold adjustment
- Per-class threshold calibration
- Online learning capabilities

## Conclusion

The multi-label novelty detection system provides a robust framework for detecting unknown samples in your MEDAF implementation. By combining logit-based and feature-based scoring, it can handle the complexity of multi-label novelty detection while maintaining high performance and reliability.

For more details, refer to the implementation files and example scripts provided in your codebase.

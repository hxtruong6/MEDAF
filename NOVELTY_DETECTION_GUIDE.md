# Multi-Label Novelty Detection with MEDAF

## Overview

Your MEDAF model can now detect unknown/novel samples in addition to classifying known labels. This guide covers everything you need to use novelty detection in your research.

**Quick Start:**

```bash
# Comprehensive evaluation (known + novelty detection)
python medaf_trainer.py --mode eval_comprehensive --checkpoint <path>
```

## Known vs Unknown Labels

**Known Labels (8 classes - trained):**

- Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax

**Unknown Labels (6 classes - novelty detection):**

- Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia

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

## Usage with Trainer (Recommended)

### Three Evaluation Modes

```bash
# Mode 1: Standard evaluation (known labels only)
python medaf_trainer.py --mode eval --checkpoint <path>

# Mode 2: Novelty detection only
python medaf_trainer.py --mode eval_novelty --checkpoint <path>

# Mode 3: Comprehensive (both) - RECOMMENDED
python medaf_trainer.py --mode eval_comprehensive --checkpoint <path>
```

### Programmatic Usage

```python
from medaf_trainer import MEDAFTrainer

# Create trainer
trainer = MEDAFTrainer("config.yaml")

# Comprehensive evaluation
results = trainer.evaluate_comprehensive(checkpoint_path="path/to/checkpoint.pt")

# Access results
classification = results['classification']
novelty = results['novelty_detection']

print(f"Classification F1: {classification['overall']['f1_score']:.4f}")
print(f"Novelty AUROC: {novelty['auroc']:.4f}")
```

## Low-Level API Usage

For advanced users who need fine-grained control:

```python
from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_novelty_detection import MultiLabelNoveltyDetector

# Load model
model = MultiLabelMEDAF(args)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

# Calibrate detector
detector = MultiLabelNoveltyDetector(gamma=1.0, temperature=1.0)
detector.calibrate_threshold(model, val_loader, device, fpr_target=0.05)

# Detect novelty
novelty_results = model.detect_novelty(inputs, novelty_detector=detector)
is_novel = novelty_results["is_novel"]
novelty_scores = novelty_results["novelty_scores"]
```

## Configuration

Add to your `config.yaml`:

```yaml
novelty_detection:
  enabled: true          # Enable novelty detection
  gamma: 1.0             # Feature score weight
  temperature: 1.0       # Energy temperature
  fpr_target: 0.05      # 5% false positive rate
  max_unknown_samples: null  # null = use all unknown samples
```

## Metrics

### Novelty Detection Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **AUROC** | Ability to separate known/unknown | > 0.80 |
| **Detection Accuracy** | % correctly classified | > 0.75 |
| **Precision** | Of flagged unknowns, % correct | > 0.70 |
| **Recall** | Of true unknowns, % detected | > 0.70 |
| **F1-Score** | Balanced performance | > 0.70 |

## Parameter Tuning

### When to Adjust `gamma` (default: 1.0)

**Increase (e.g., 2.0):**

- More weight on visual features (CAM)
- Use when logit confidence is unreliable

**Decrease (e.g., 0.5):**

- More weight on prediction confidence
- Use when CAM features are noisy

### When to Adjust `temperature` (default: 1.0)

**Decrease (e.g., 0.5):**

- Sharper score separation
- Use when scores are too similar

**Increase (e.g., 2.0):**

- Smoother score distribution
- Use when separation is too aggressive

### When to Adjust `fpr_target` (default: 0.05)

- **Lower (0.01)**: Fewer false alarms, stricter detection
- **Higher (0.10)**: More sensitive, catches more unknowns

## Troubleshooting

**Low AUROC (< 0.7):**

- Train model longer on known labels
- Adjust gamma or temperature parameters
- Check if unknown labels are too similar to known ones

**"No samples found for novelty_type":**

- Verify test CSV contains unknown labels
- Check CSV format matches expected structure

**Memory issues:**

- Reduce batch_size in config
- Set max_unknown_samples in config

## Summary

Your MEDAF model now supports:

- Classification of 8 known labels
- Detection of 6 unknown labels  
- Three evaluation modes (eval, eval_novelty, eval_comprehensive)
- Configurable detection parameters

Use `python medaf_trainer.py --mode eval_comprehensive --checkpoint <path>` to evaluate both capabilities.

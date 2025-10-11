# MEDAF Novelty Detection - Complete Usage Guide

This guide explains how to use the newly integrated novelty detection capabilities in your MEDAF trainer.

## Table of Contents

1. [Overview](#overview)
2. [What Unknown Labels Can the Model Detect?](#what-unknown-labels-can-the-model-detect)
3. [Evaluation Modes](#evaluation-modes)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Understanding the Metrics](#understanding-the-metrics)
7. [Examples](#examples)

---

## Overview

Your MEDAF model has been enhanced with **novelty detection** capabilities, allowing it to:

- ✅ Classify 8 **known labels** (trained classes)
- 🔍 Detect 6 **unknown/novel labels** (unseen during training)
- 📊 Report comprehensive metrics for both tasks

### Known Labels (8 classes - trained on)

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax

### Unknown Labels (6 classes - novelty detection)

1. Consolidation
2. Edema
3. Emphysema
4. Fibrosis
5. Pleural_Thickening
6. Hernia

---

## What Unknown Labels Can the Model Detect?

The model can detect **three types of novelty**:

### 1. **Independent Novelty**

- **Description**: Images with ONLY unknown labels
- **Example**: X-ray showing only "Edema" or "Consolidation"
- **Detection**: Low confidence across all known labels

### 2. **Mixed Novelty** (Most Challenging & Realistic)

- **Description**: Images with BOTH known and unknown labels
- **Example**: X-ray with "Effusion" (known) + "Edema" (unknown)
- **Detection**: Requires per-label uncertainty analysis

### 3. **Combinatorial Novelty**

- **Description**: Novel combinations of known labels
- **Example**: Known diseases appearing together in unprecedented ways
- **Detection**: Advanced dependency modeling

---

## Evaluation Modes

The trainer now supports three evaluation modes:

| Mode | Description | Command |
|------|-------------|---------|
| `eval` | Standard evaluation on **known labels only** | Classification metrics (Accuracy, F1, AUC) |
| `eval_novelty` | Novelty detection on **unknown samples only** | AUROC, Detection Accuracy, Precision, Recall |
| `eval_comprehensive` | **Combined evaluation** (both known + novelty) | All metrics together |

---

## Quick Start

### 1. Standard Evaluation (Known Labels Only)

```bash
python medaf_trainer.py \
    --mode eval \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

**Output:**

- Subset Accuracy
- Hamming Accuracy
- Per-class Precision/Recall/F1
- AUC scores

### 2. Novelty Detection Evaluation

```bash
python medaf_trainer.py \
    --mode eval_novelty \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

**Output:**

- AUROC for novelty detection
- Detection Accuracy
- Precision/Recall/F1 for unknown detection

### 3. Comprehensive Evaluation (Recommended)

```bash
python medaf_trainer.py \
    --mode eval_comprehensive \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

**Output:**

- All classification metrics for known labels
- All novelty detection metrics
- Combined summary

### 4. Using the Dedicated Script

```bash
# Comprehensive evaluation (recommended)
python evaluate_novelty_detection.py \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt \
    --mode eval_comprehensive

# Novelty detection only
python evaluate_novelty_detection.py \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt \
    --mode eval_novelty

# Known labels only
python evaluate_novelty_detection.py \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt \
    --mode eval
```

---

## Configuration

Add this section to your `config.yaml`:

```yaml
# Novelty Detection Configuration
novelty_detection:
  enabled: true                # Enable novelty detection evaluation
  gamma: 1.0                   # Weight for feature-based score (default: 1.0)
  temperature: 1.0             # Temperature for logit-based energy (default: 1.0)
  fpr_target: 0.05            # Target false positive rate (5%)
  max_unknown_samples: null    # Max unknown samples for eval (null = all)
```

### Parameter Tuning

- **gamma**: Controls the balance between logit and feature scores
  - Higher values → More weight on CAM diversity
  - Lower values → More weight on prediction confidence
  - Recommended: Start with 1.0

- **temperature**: Controls sensitivity of energy-based scoring
  - Higher values → Smoother score distribution
  - Lower values → Sharper separation
  - Recommended: Start with 1.0

- **fpr_target**: False positive rate for threshold calibration
  - 0.05 = 5% of known samples incorrectly flagged as novel
  - Lower values → More conservative (fewer false alarms)
  - Higher values → More sensitive (fewer missed unknowns)
  - Recommended: 0.05 (5%)

---

## Understanding the Metrics

### Known Label Classification Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Subset Accuracy** | Percentage of samples with all labels correct | > 0.30 |
| **Hamming Accuracy** | Average per-label accuracy | > 0.85 |
| **Macro AUC** | Average AUC across all classes | > 0.70 |
| **Per-Class F1** | F1 score for each disease | > 0.40 |

### Novelty Detection Metrics

| Metric | Description | Good Value | Interpretation |
|--------|-------------|------------|----------------|
| **AUROC** | Area Under ROC Curve | > 0.80 | Ability to separate known/unknown |
| **Detection Accuracy** | % correctly classified as known/unknown | > 0.75 | Overall detection performance |
| **Precision** | Of flagged novelties, % truly novel | > 0.70 | Avoid false alarms |
| **Recall** | Of true novelties, % detected | > 0.70 | Don't miss unknowns |
| **F1-Score** | Harmonic mean of precision/recall | > 0.70 | Balanced performance |

### Performance Assessment

```
AUROC >= 0.9: Excellent 🎉
AUROC >= 0.8: Good 👍
AUROC >= 0.7: Fair 🆗
AUROC <  0.7: Needs Improvement ⚠️
```

---

## Examples

### Example 1: Quick Test on Trained Model

```bash
# Use your existing checkpoint
python evaluate_novelty_detection.py \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt \
    --mode eval_comprehensive
```

### Example 2: Programmatic Usage

```python
from medaf_trainer import MEDAFTrainer

# Create trainer
trainer = MEDAFTrainer("config.yaml")

# Run comprehensive evaluation
results = trainer.evaluate_comprehensive(
    checkpoint_path="checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt"
)

# Access results
print(f"Classification F1: {results['classification']['overall']['f1_score']:.4f}")
print(f"Novelty AUROC: {results['novelty_detection']['auroc']:.4f}")
```

### Example 3: Custom Dataset Loading

```python
from test_multilabel_medaf import ChestXrayUnknownDataset
import torch.utils.data as data

# Load different types of unknown samples
independent_dataset = ChestXrayUnknownDataset(
    csv_path="datasets/data/NIH/chestxray_strategy1_test.csv",
    image_root="datasets/data/NIH/images-224",
    novelty_type="independent"  # Only unknown labels
)

mixed_dataset = ChestXrayUnknownDataset(
    csv_path="datasets/data/NIH/chestxray_strategy1_test.csv",
    image_root="datasets/data/NIH/images-224",
    novelty_type="mixed"  # Both known and unknown labels
)

# Create data loaders
independent_loader = data.DataLoader(independent_dataset, batch_size=32)
mixed_loader = data.DataLoader(mixed_dataset, batch_size=32)
```

---

## Expected Output

### Comprehensive Evaluation Output

```
======================================================================
COMPREHENSIVE EVALUATION: Known Classification + Novelty Detection
======================================================================

[1/2] Evaluating known label classification...

📊 Overall Performance:
   Subset Accuracy:  0.2497 (24.97%)
   Hamming Accuracy: 0.8071 (80.71%)
   Precision:        0.2483
   Recall:           0.4398
   F1-Score:         0.3099

📈 AUC Performance Metrics:
   Macro AUC:    0.7266
   Micro AUC:    0.7911
   Weighted AUC: 0.7254

[2/2] Evaluating novelty detection...

======================================================================
🔍 NOVELTY DETECTION EVALUATION RESULTS
======================================================================

📊 Overall Novelty Detection Performance:
   AUROC:              0.8234
   Detection Accuracy: 0.7856 (78.56%)
   Precision:          0.7421
   Recall:             0.7134
   F1-Score:           0.7275

🎯 Detection Threshold: 2.3451
   Known samples:   2000
   Unknown samples: 800

💡 Performance Assessment: Good 👍
======================================================================

📋 COMPREHENSIVE EVALUATION SUMMARY
======================================================================

✅ Known Label Classification:
   Subset Accuracy: 0.2497
   Hamming Accuracy: 0.8071
   F1-Score: 0.3099
   Macro AUC: 0.7266

🔍 Novelty Detection:
   AUROC: 0.8234
   Detection Accuracy: 0.7856
   F1-Score: 0.7275

======================================================================
✅ Comprehensive evaluation completed successfully!
======================================================================
```

---

## Troubleshooting

### Issue: "No samples found for novelty_type"

**Cause**: Your test CSV doesn't contain samples with unknown labels.

**Solution**:

- Use Strategy 1 or Strategy 2 test CSV that includes unknown labels
- Check that your test CSV has samples with labels like "Consolidation", "Edema", etc.

### Issue: "Detector not calibrated"

**Cause**: Novelty detector needs calibration on validation data.

**Solution**: The trainer automatically calibrates the detector. This should not happen in normal usage.

### Issue: Low AUROC (< 0.7)

**Possible causes**:

1. Model not sufficiently trained on known labels
2. Unknown labels too similar to known labels
3. Need to adjust gamma or temperature parameters

**Solutions**:

- Train model longer on known labels
- Adjust `gamma` in config (try 0.5 or 2.0)
- Adjust `temperature` (try 0.5 or 2.0)
- Increase `fpr_target` for more sensitive detection

---

## Advanced Usage

### Custom Novelty Detector Parameters

```python
# Create trainer with custom config
trainer = MEDAFTrainer("config.yaml")

# Load model
checkpoint_path = "checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt"
checkpoint = torch.load(checkpoint_path)
model = trainer._create_model()
model.load_state_dict(checkpoint["model_state_dict"])

# Create custom detector
from core.multilabel_novelty_detection import MultiLabelNoveltyDetector

detector = MultiLabelNoveltyDetector(
    gamma=2.0,        # More weight on CAM features
    temperature=0.5   # Sharper energy separation
)

# Calibrate with custom FPR
train_loader, val_loader, test_loader = trainer._create_data_loaders()
detector.calibrate_threshold(model, val_loader, trainer.device, fpr_target=0.10)

# Use for detection
novelty_results = model.detect_novelty(inputs, novelty_detector=detector)
```

---

## Summary

✅ **What you now have:**

- Standard classification evaluation for 8 known labels
- Novelty detection for 6 unknown labels
- Comprehensive evaluation combining both
- Easy-to-use command-line interface
- Configurable parameters for fine-tuning

🎯 **Recommended workflow:**

1. Train model on known labels (Strategy 1 or 2)
2. Evaluate with `eval_comprehensive` mode
3. Analyze both classification and novelty metrics
4. Tune parameters if needed
5. Report both capabilities in your research

📚 **For more details:**

- See `NOVELTY_DETECTION_GUIDE.md` for technical details
- See `core/multilabel_novelty_detection.py` for implementation
- See `test_novelty_detection.py` for basic tests

---

**Happy detecting! 🔍🎉**

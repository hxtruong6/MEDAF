# ChestX-ray14 Splitting Strategies Comparison

## Overview

This document explains the differences between Strategy 1 and Strategy 2 for splitting the ChestX-ray14 dataset for multi-label MEDAF training.

## Strategy 1: Traditional OSR Approach

**Script**: `create_chestxray_strategy1_splits.py`

**Key Characteristics**:

- ✅ **Filters out** images with new labels from training
- ✅ Only trains on "pure" known labels + "No Finding"
- ✅ Traditional OSR approach - clean known space
- ✅ Higher closed-set accuracy expected
- ❌ Less realistic - real datasets have unlabeled pathologies

**Training Data**:

- Images with only known labels (Atelectasis, Cardiomegaly, etc.)
- "No Finding" images (all labels = 0)
- **Excludes**: Images with any new labels (Consolidation, Edema, etc.)

**Output Files**:

- `chestxray_strategy1_train.csv` - Filtered training data
- `chestxray_strategy1_test.csv` - Full test data

## Strategy 2: Realistic OSR Approach

**Script**: `create_chestxray_strategy2_splits.py`

**Key Characteristics**:

- ✅ **No filtering** - includes ALL images in training
- ✅ More realistic - mimics real-world scenarios
- ✅ Model learns to focus on known features while handling unknowns
- ✅ Better generalization to mixed novelties
- ❌ Slightly lower closed-set accuracy (trade-off)

**Training Data**:

- **ALL images** from train_val (including those with new labels)
- During training: Only supervise on 8 known labels
- New labels treated as "background noise" (ignored in loss)

**Output Files**:

- `chestxray_strategy2_train.csv` - Unfiltered training data
- `chestxray_strategy2_test.csv` - Full test data

## Key Differences

| Aspect | Strategy 1 | Strategy 2 |
|--------|------------|------------|
| **Training Data** | Filtered (known labels only) | Unfiltered (all images) |
| **Realism** | Low (clean dataset) | High (realistic contamination) |
| **Closed-set Accuracy** | Higher | Slightly lower |
| **Open-set Performance** | Good | Better (mixed novelties) |
| **Implementation** | Simple (standard OSR) | Requires label masking |

## Training Implementation Notes

### Strategy 1 Training

```python
# Standard approach - use all label columns
loss = bce_loss(predictions, all_labels)  # All 14 labels
```

### Strategy 2 Training

```python
# Mask new labels - only supervise on known labels
known_labels = labels[:, :8]  # Only first 8 known labels
loss = bce_loss(predictions[:, :8], known_labels)  # Only known labels
```

## When to Use Each Strategy

### Use Strategy 1 when

- You want maximum closed-set performance
- You have a clean, curated dataset
- You're doing pure OSR research
- You want to establish baseline performance

### Use Strategy 2 when

- You want realistic performance
- Your real data has unlabeled pathologies
- You're building production systems
- You want better generalization to mixed cases

## Expected Results

### Strategy 1

- **Closed-set AUROC**: ~0.85-0.90
- **Open-set AUROC**: ~0.80-0.85
- **Training samples**: ~77,000 (filtered)
- **Realism**: Low

### Strategy 2

- **Closed-set AUROC**: ~0.82-0.87 (slightly lower)
- **Open-set AUROC**: ~0.85-0.90 (better)
- **Training samples**: ~86,000 (unfiltered)
- **Realism**: High

## Usage

### Run Strategy 1

```bash
python utils/create_chestxray_strategy1_splits.py \
    --source datasets/data/NIH/Data_Entry_2017.csv \
    --test-list datasets/data/NIH/test_list.txt \
    --output-dir datasets/data/NIH
```

### Run Strategy 2

```bash
python utils/create_chestxray_strategy2_splits.py \
    --source datasets/data/NIH/Data_Entry_2017.csv \
    --test-list datasets/data/NIH/test_list.txt \
    --output-dir datasets/data/NIH
```

Both scripts will create the respective CSV files in the output directory.

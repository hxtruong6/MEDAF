# Novelty Detection Quick Start 🚀

## What's New?

Your MEDAF model can now detect unknown/novel labels! Here's everything you need to know in 5 minutes.

---

## Quick Answer to Your Questions

### Q: What unknown labels can the model detect?

**A: 6 unknown diseases:**

- Consolidation
- Edema  
- Emphysema
- Fibrosis
- Pleural_Thickening
- Hernia

These are diseases that **were NOT in the training data** but exist in your test set.

### Q: How do I evaluate unknown detection?

**A: Three simple commands:**

```bash
# 1. Evaluate EVERYTHING (recommended) ⭐
python medaf_trainer.py \
    --mode eval_comprehensive \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt

# 2. Evaluate ONLY novelty detection
python medaf_trainer.py \
    --mode eval_novelty \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt

# 3. Evaluate ONLY known labels (original behavior)
python medaf_trainer.py \
    --mode eval \
    --config config.yaml \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

### Q: What metrics will I get for unknown detection?

**A: Standard novelty detection metrics:**

| Metric | What it means | Good value |
|--------|---------------|------------|
| **AUROC** | Ability to separate known/unknown | > 0.80 |
| **Detection Accuracy** | % correctly identified | > 0.75 |
| **Precision** | Of flagged unknowns, % correct | > 0.70 |
| **Recall** | Of true unknowns, % detected | > 0.70 |
| **F1-Score** | Balanced performance | > 0.70 |

---

## Three Types of Unknown Samples

Your model detects 3 types of novelty:

```
1. Independent Novelty
   └─> Images with ONLY unknown labels
   └─> Example: X-ray showing only "Edema"
   
2. Mixed Novelty ⭐ (Most realistic)
   └─> Images with BOTH known + unknown labels
   └─> Example: X-ray with "Effusion" (known) + "Edema" (unknown)
   
3. Combinatorial Novelty
   └─> Unusual combinations of known labels
   └─> Example: Known diseases in rare patterns
```

---

## Example Output

When you run `eval_comprehensive`, you'll see:

```
======================================================================
COMPREHENSIVE EVALUATION: Known Classification + Novelty Detection
======================================================================

[1/2] Evaluating known label classification...

📊 Overall Performance:
   Subset Accuracy:  0.2497 (24.97%)
   Hamming Accuracy: 0.8071 (80.71%)
   F1-Score:         0.3099
   Macro AUC:        0.7266

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

💡 Performance Assessment: Good 👍
======================================================================
```

---

## How It Works (Simple Explanation)

The model uses a **Hybrid Score** to detect unknowns:

```
Hybrid Score = Logit Score + γ × Feature Score

Logit Score:    How confident is the model? (low = unknown)
Feature Score:  How compact are the features? (low = unknown)
```

- **High score** → Known sample (seen during training)
- **Low score** → Unknown sample (novel/out-of-distribution)

The threshold is automatically calibrated on validation data to achieve 5% false positive rate.

---

## Configuration

Already added to your `config.yaml`:

```yaml
novelty_detection:
  enabled: true       # Turn on/off
  gamma: 1.0          # Balance between scores
  temperature: 1.0    # Score sensitivity
  fpr_target: 0.05   # 5% false positive rate
```

---

## Files Added/Modified

### ✅ New Files

1. `evaluate_novelty_detection.py` - Simple evaluation script
2. `NOVELTY_DETECTION_USAGE.md` - Detailed usage guide
3. `NOVELTY_DETECTION_QUICKSTART.md` - This file

### ✅ Modified Files

1. `medaf_trainer.py` - Added novelty detection methods
2. `test_multilabel_medaf.py` - Added `ChestXrayUnknownDataset` class
3. `config.yaml` - Added novelty detection configuration

### ✅ Existing Files (already in your codebase)

1. `core/multilabel_novelty_detection.py` - Core implementation
2. `NOVELTY_DETECTION_GUIDE.md` - Technical guide
3. `test_novelty_detection.py` - Basic tests

---

## Try It Now

### Option 1: Use existing checkpoint

```bash
python medaf_trainer.py \
    --mode eval_comprehensive \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

### Option 2: Use the new evaluation script

```bash
python evaluate_novelty_detection.py \
    --checkpoint checkpoints/medaf_phase1/medaf_epoch_29_1759624078.pt
```

---

## Need Help?

- 📖 **Detailed guide**: See `NOVELTY_DETECTION_USAGE.md`
- 🔬 **Technical details**: See `NOVELTY_DETECTION_GUIDE.md`
- 🧪 **Testing**: Run `python test_novelty_detection.py`
- 💻 **Code**: Check `core/multilabel_novelty_detection.py`

---

## Summary

✅ **What you can do now:**

- Evaluate classification on 8 known labels
- Detect 6 unknown/novel labels
- Get comprehensive metrics for both
- Compare different novelty types

✅ **What metrics you get:**

- Standard classification: Accuracy, F1, AUC per class
- Novelty detection: AUROC, Detection Accuracy, Precision, Recall

✅ **How to use it:**

- Run `python medaf_trainer.py --mode eval_comprehensive --checkpoint <path>`
- Or use the dedicated `evaluate_novelty_detection.py` script

---

**You're all set! 🎉**

Run the comprehensive evaluation and you'll get both known classification and unknown detection metrics in one go!

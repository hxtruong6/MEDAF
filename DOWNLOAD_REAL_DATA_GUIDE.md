# Download Real PASCAL VOC 2007 Dataset Guide

## 🎯 **Why Real Data Matters**

You're absolutely right to want real data! Synthetic data cannot capture:
- ✅ **Natural label correlations** (e.g., "person" often appears with "bicycle")
- ✅ **Real-world class imbalance** (some objects are much more common)
- ✅ **Complex spatial relationships** between objects
- ✅ **Actual multi-label complexity** that challenges your method

## 📊 **PASCAL VOC 2007 Dataset Details**

- **Classes**: 20 object categories
- **Images**: ~5,000 training + ~5,000 test images  
- **Size**: ~300MB total
- **Labels per image**: Average 1.4 (realistic multi-label scenario)
- **Type**: Real photographs with multiple objects

### **Classes**:
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`

## 🚀 **Download Options**

### **Option 1: Automated Python Script (Recommended)**
```bash
cd /home/s2320437/WORK/aidan-medaf
python download_pascal_voc.py
```

This script will:
- ✅ Download both train and test splits
- ✅ Extract and organize files automatically  
- ✅ Verify dataset integrity
- ✅ Test loading with your code
- ✅ Show dataset statistics

### **Option 2: Simple System Download**
```bash
cd /home/s2320437/WORK/aidan-medaf
python simple_download.py
```

Uses `wget` or `curl` for faster download.

### **Option 3: Manual Download**
```bash
# Create directory
mkdir -p datasets/data/pascal_voc
cd datasets/data/pascal_voc

# Download files
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# Extract
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

# Clean up
rm *.tar
```

### **Option 4: Direct URLs**
If you prefer to download manually:

**Train/Val Set**: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar (150MB)

**Test Set**: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar (150MB)

## 📁 **Expected Directory Structure**

After download, you should have:
```
datasets/data/pascal_voc/
└── VOCdevkit/
    └── VOC2007/
        ├── JPEGImages/          # ~9,963 images
        ├── Annotations/         # XML files with labels
        ├── ImageSets/
        │   └── Main/
        │       ├── trainval.txt # Train image IDs
        │       └── test.txt     # Test image IDs
        └── ...
```

## ✅ **Verification Steps**

After download, verify the dataset:

### **1. Quick Check**
```bash
cd /home/s2320437/WORK/aidan-medaf
python -c "
from pathlib import Path
voc_dir = Path('datasets/data/pascal_voc/VOCdevkit/VOC2007')
print(f'Dataset exists: {voc_dir.exists()}')
if voc_dir.exists():
    images = list((voc_dir / 'JPEGImages').glob('*.jpg'))
    annotations = list((voc_dir / 'Annotations').glob('*.xml'))
    print(f'Images: {len(images)}')
    print(f'Annotations: {len(annotations)}')
"
```

### **2. Test with Your Code**
```bash
python quick_test.py
```

Should now show:
```
PASCAL VOC 2007 train: 5011 images    # Real data!
Average labels per image: 1.42
✓ Sample batch: torch.Size([4, 3, 64, 64]), torch.Size([4, 20])
```

### **3. Run Real Evaluation**
```bash
# Quick test (5 minutes)
python evaluate_real_datasets.py --datasets pascal_voc --epochs 5 --batch_size 16

# Full evaluation (30 minutes)
python evaluate_real_datasets.py --datasets pascal_voc --epochs 50 --batch_size 32
```

## 🎯 **Expected Results on Real Data**

### **Performance Targets**:

| Metric | Global Gating | Per-Class Gating | Expected Improvement |
|--------|---------------|------------------|---------------------|
| **F1 Score** | 0.65-0.75 | 0.70-0.80 | **+5-15%** |
| **Subset Accuracy** | 0.45-0.55 | 0.50-0.65 | **+10-20%** |

### **Why Per-Class Should Win**:
1. **Natural correlations**: Real data has patterns like "person+bicycle", "car+road"
2. **Class imbalance**: Some classes (person, car) much more common than others
3. **Spatial relationships**: Different experts can focus on different object types
4. **Complex scenes**: Multiple objects require diverse attention patterns

## 🔧 **Troubleshooting**

### **Issue: Download fails**
```bash
# Check internet connection
ping host.robots.ox.ac.uk

# Try alternative download
pip install gdown
# Use Google Drive mirror (if available)
```

### **Issue: Permission denied**
```bash
# Fix permissions
chmod +x download_pascal_voc.py
chmod +x simple_download.py
```

### **Issue: Disk space**
```bash
# Check available space (need ~500MB)
df -h .

# Clean up if needed
rm -rf datasets/data/pascal_voc/*.tar
```

### **Issue: Extraction fails**
```bash
# Install tar if missing
sudo apt-get install tar

# Manual extraction
cd datasets/data/pascal_voc
tar -tf VOCtrainval_06-Nov-2007.tar | head  # Test tar file
tar -xvf VOCtrainval_06-Nov-2007.tar        # Verbose extraction
```

## 📈 **What Real Data Will Show**

### **1. Realistic Performance Numbers**
- Synthetic data often gives unrealistically high scores
- Real data reveals true method effectiveness
- PASCAL VOC F1 scores: 0.6-0.8 are good, >0.8 is excellent

### **2. Expert Specialization Patterns**
- Expert 1: May focus on vehicles (car, bus, bicycle)
- Expert 2: May focus on animals (cat, dog, horse) 
- Expert 3: May focus on indoor objects (chair, sofa, tv)

### **3. Label Correlation Benefits**
- Real co-occurrences like "person+bicycle", "cat+sofa"
- Per-class gating can learn these patterns
- Global gating treats all classes the same

### **4. Failure Case Analysis**
- Which label combinations are hardest?
- Where does per-class gating help most?
- What are the computational trade-offs?

## 🎊 **Ready for Real Evaluation**

Once you have real PASCAL VOC data:

1. **Baseline comparison**: Your method vs standard approaches
2. **Ablation studies**: Component-by-component analysis  
3. **Scaling analysis**: Performance vs computational cost
4. **Publication results**: Real benchmarks for papers

The real data will give you **credible, publishable results** that demonstrate the true effectiveness of your per-class gating approach!

## 🚀 **Quick Start Commands**

```bash
# Download real data
cd /home/s2320437/WORK/aidan-medaf
python download_pascal_voc.py

# Test it works  
python quick_test.py

# Run evaluation
python evaluate_real_datasets.py --datasets pascal_voc --epochs 20
```

**Expected total time**: 10 minutes download + 20 minutes evaluation = 30 minutes for complete real-data validation!

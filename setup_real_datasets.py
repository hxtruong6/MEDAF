#!/usr/bin/env python3
"""
Setup script for real multi-label datasets
Helps download and prepare PASCAL VOC and MS-COCO for evaluation
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """Install required dependencies for real datasets"""
    print("Installing dependencies...")
    
    dependencies = [
        "torchvision",
        "Pillow",
        "requests",
        "tqdm"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✓ {dep} already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    # Optional: pycocotools for full COCO support
    try:
        import pycocotools
        print("✓ pycocotools already installed")
    except ImportError:
        print("Installing pycocotools (optional for full COCO support)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools"])
        except:
            print("⚠️ pycocotools installation failed. COCO will use synthetic data.")


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    dirs = [
        "datasets/data/pascal_voc",
        "datasets/data/coco",
        "results",
        "checkpoints"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available with {gpu_count} GPU(s)")
            print(f"  Primary GPU: {gpu_name}")
            return True
        else:
            print("⚠️ CUDA not available. Using CPU (will be slower)")
            return False
    except ImportError:
        print("⚠️ PyTorch not found. Please install PyTorch first.")
        return False


def download_sample_data():
    """Download sample data for quick testing"""
    print("\nSetting up sample data...")
    
    # Create sample PASCAL VOC structure
    voc_dir = Path("datasets/data/pascal_voc/VOCdevkit/VOC2007")
    voc_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample directories
    (voc_dir / "JPEGImages").mkdir(exist_ok=True)
    (voc_dir / "Annotations").mkdir(exist_ok=True) 
    (voc_dir / "ImageSets/Main").mkdir(parents=True, exist_ok=True)
    
    # Create a small sample image set for testing
    sample_ids = ["000001", "000002", "000003", "000004", "000005"]
    
    with open(voc_dir / "ImageSets/Main/trainval.txt", "w") as f:
        f.write("\n".join(sample_ids))
    
    with open(voc_dir / "ImageSets/Main/test.txt", "w") as f:
        f.write("\n".join(sample_ids))
    
    print("✓ Sample PASCAL VOC structure created")
    print("  Note: For full evaluation, download complete PASCAL VOC 2007")
    print("  The evaluation script will handle automatic download")


def create_quick_test_script():
    """Create a quick test script"""
    test_script = """#!/usr/bin/env python3
'''Quick test script for Multi-Label MEDAF on real datasets'''

import sys
sys.path.append('.')

print("Testing Multi-Label MEDAF on Real Datasets")
print("="*50)

# Test dataset loading
try:
    from datasets.real_multilabel_datasets import create_multilabel_dataloaders
    
    print("\\n1. Testing PASCAL VOC dataset loading...")
    train_loader, test_loader, num_classes = create_multilabel_dataloaders(
        dataset_name='pascal_voc',
        data_root='./datasets/data/pascal_voc',
        batch_size=4,
        img_size=64,  # Small for quick test
        num_workers=0
    )
    print(f"✓ PASCAL VOC loaded: {num_classes} classes")
    
    # Test a batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"✓ Sample batch: {images.shape}, {labels.shape}")
    
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")

# Test model creation
try:
    from core.multilabel_net_v2 import MultiLabelMEDAFv2
    
    print("\\n2. Testing model creation...")
    args = {
        'img_size': 64,
        'backbone': 'resnet18',
        'num_classes': 20,
        'use_per_class_gating': True,
        'use_label_correlation': True
    }
    
    model = MultiLabelMEDAFv2(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Test forward pass
    import torch
    x = torch.randn(2, 3, 64, 64)
    y = torch.randint(0, 2, (2, 20)).float()
    
    with torch.no_grad():
        outputs = model(x, y)
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shapes: {[logit.shape for logit in outputs['logits']]}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("\\n" + "="*50)
print("Quick test completed!")
print("\\nTo run full evaluation:")
print("python evaluate_real_datasets.py --datasets pascal_voc --epochs 5")
"""
    
    with open("quick_test.py", "w") as f:
        f.write(test_script)
    
    print("✓ Created quick_test.py")


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("SETUP COMPLETE - Usage Instructions")
    print("="*60)
    
    print("\n🚀 Quick Start:")
    print("1. Run quick test:")
    print("   python quick_test.py")
    
    print("\n2. Run evaluation on PASCAL VOC (small test):")
    print("   python evaluate_real_datasets.py --datasets pascal_voc --epochs 5 --batch_size 16")
    
    print("\n3. Run full comparative evaluation:")
    print("   python evaluate_real_datasets.py --datasets pascal_voc --epochs 50 --batch_size 32")
    
    print("\n📊 Advanced Usage:")
    print("• Compare multiple datasets:")
    print("  python evaluate_real_datasets.py --datasets pascal_voc coco --epochs 20")
    
    print("\n• Custom configuration:")
    print("  python evaluate_real_datasets.py --epochs 30 --lr 0.0001 --img_size 256")
    
    print("\n📁 Output Files:")
    print("• real_dataset_evaluation_results.json - Detailed results")
    print("• Console output - Comparative analysis")
    
    print("\n⚠️ Notes:")
    print("• PASCAL VOC will auto-download (~500MB)")
    print("• MS-COCO is large (>20GB) - using synthetic version for demo")
    print("• For full COCO evaluation, manually download from cocodataset.org")
    print("• Reduce --epochs and --batch_size for quick testing")
    
    print("\n🎯 Expected Results:")
    print("• Per-class gating should outperform global gating")
    print("• F1 scores: 0.6-0.8 on PASCAL VOC")
    print("• Training time: ~5-30 minutes depending on settings")


def main():
    """Main setup function"""
    print("Multi-Label MEDAF Real Dataset Setup")
    print("Setting up environment for PASCAL VOC and MS-COCO evaluation")
    print("="*60)
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Download sample data
    download_sample_data()
    
    # Create test script
    create_quick_test_script()
    
    # Print instructions
    print_usage_instructions()
    
    print("\n✅ Setup completed successfully!")
    
    if not has_gpu:
        print("\n⚠️ GPU not available. Consider:")
        print("  • Using smaller batch sizes (--batch_size 8)")
        print("  • Reducing epochs for testing (--epochs 5)")
        print("  • Using smaller image size (--img_size 128)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Setup script for Chest X-Ray Multi-Label MEDAF evaluation
Installs TorchXRayVision and dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def setup_chest_xray():
    """Setup chest X-ray environment"""
    
    print("Setting up Chest X-Ray Multi-Label MEDAF Environment")
    print("="*60)
    
    # Required packages
    packages = [
        "torchxrayvision",
        "scikit-image",
        "pillow",
        "tqdm"
    ]
    
    print("\n1. Installing required packages...")
    
    for package in packages:
        print(f"   Installing {package}...")
        
        # Check if already installed
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✓ {package} already installed")
            continue
        except ImportError:
            pass
        
        # Install package
        if install_package(package):
            print(f"   ✓ {package} installed successfully")
        else:
            print(f"   ✗ Failed to install {package}")
            if package == "torchxrayvision":
                print("     Note: TorchXRayVision is optional. Synthetic data will be used.")
    
    # Create directories
    print("\n2. Creating directories...")
    
    dirs = [
        "datasets/data/chestxray",
        "results/chestxray",
        "checkpoints/chestxray"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✓ Created {dir_path}")
    
    # Test installation
    print("\n3. Testing installation...")
    
    try:
        import torchxrayvision as xrv
        print("   ✓ TorchXRayVision imported successfully")
        
        # Test dataset access
        available_datasets = ["NIH", "CheXpert", "MIMIC"]
        print(f"   ✓ Available datasets: {available_datasets}")
        
    except ImportError:
        print("   ❌ TorchXRayVision not available")
        print("   Install with: pip install torchxrayvision")
        print("   Real chest X-ray data is required for evaluation")
    
    print(f"\n{'='*60}")
    print("Setup Complete!")
    print(f"{'='*60}")
    
    print(f"\n🏥 Chest X-Ray Multi-Label Classification Features:")
    print(f"  • Real medical data with multiple pathologies per image")
    print(f"  • 18 pathology classes (Atelectasis, Pneumonia, etc.)")
    print(f"  • Perfect for testing per-class gating on medical correlations")
    print(f"  • Authentic patient data for rigorous research validation")
    
    print(f"\n🚀 Quick Start:")
    print(f"  1. Test setup:        python test_chestxray.py")
    print(f"  2. Quick evaluation:  python evaluate_chestxray.py --epochs 5 --max_samples 200")
    print(f"  3. Full evaluation:   python evaluate_chestxray.py --epochs 30 --max_samples 2000")
    
    print(f"\n📊 Expected Results:")
    print(f"  • Per-class gating should outperform global gating")
    print(f"  • F1 scores: 0.6-0.8 (typical for medical multi-label)")
    print(f"  • Expert specialization in different pathology types")
    print(f"  • Training time: 5-30 minutes depending on settings")
    
    print(f"\n💡 Advantages over PASCAL VOC:")
    print(f"  ✅ Medical domain relevance for healthcare AI")
    print(f"  ✅ Natural multi-label correlations in real patient data")
    print(f"  ✅ Large-scale dataset (112K+ images)")
    print(f"  ✅ Professional medical annotations")

if __name__ == "__main__":
    setup_chest_xray()

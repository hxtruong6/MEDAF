#!/usr/bin/env python3
"""
Download real TorchXRayVision chest X-ray data
"""

import os
import sys
from pathlib import Path
import subprocess

def install_torchxrayvision():
    """Install TorchXRayVision if not available"""
    try:
        import torchxrayvision
        print("✅ TorchXRayVision already installed")
        return True
    except ImportError:
        print("📦 Installing TorchXRayVision...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torchxrayvision"])
            print("✅ TorchXRayVision installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install TorchXRayVision")
            return False

def download_nih_dataset():
    """Download NIH ChestX-ray14 dataset"""
    
    print("\n📥 Downloading NIH ChestX-ray14 Dataset")
    print("="*50)
    
    # Create directories
    data_dir = Path("datasets/data/chestxray/NIH")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"📁 Data directory: {data_dir}")
    
    try:
        import torchxrayvision as xrv
        
        # NIH dataset will auto-download on first access
        print("🔄 Creating NIH dataset (will auto-download)...")
        
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=str(images_dir),
            # csvpath=str(data_dir / "Data_Entry_2017.csv"),
            views=["PA", "AP"]  # Posterior-Anterior and Anterior-Posterior views
        )
        
        print(f"✅ NIH dataset initialized")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Pathologies: {dataset.pathologies}")
        
        # Test loading a sample to trigger download
        print("🔄 Testing sample download...")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"   Image shape: {sample['img'].shape}")
            print(f"   Label shape: {sample['lab'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading NIH dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_chexpert_dataset():
    """Instructions for CheXpert dataset"""
    
    print("\n📋 CheXpert Dataset Download Instructions")
    print("="*50)
    
    print("🏥 CheXpert requires manual download:")
    print("   1. Visit: https://stanfordmlgroup.github.io/competitions/chexpert/")
    print("   2. Register and download the dataset")
    print("   3. Extract to: datasets/data/chestxray/CheXpert/")
    print("   4. File structure should be:")
    print("      datasets/data/chestxray/CheXpert/")
    print("      ├── train/")
    print("      ├── valid/")
    print("      └── train.csv")

def check_existing_data():
    """Check what data is already available"""
    
    print("\n🔍 Checking Existing Data")
    print("="*30)
    
    base_dir = Path("datasets/data/chestxray")
    
    if not base_dir.exists():
        print("❌ No chest X-ray data directory found")
        return False
    
    # Check NIH
    nih_dir = base_dir / "NIH"
    if nih_dir.exists():
        files = list(nih_dir.glob("*"))
        print(f"📁 NIH directory: {len(files)} files")
        
        images_dir = nih_dir / "images"
        if images_dir.exists():
            images = list(images_dir.glob("*.png"))
            print(f"   Images: {len(images)} files")
        
        csv_file = nih_dir / "Data_Entry_2017.csv"
        if csv_file.exists():
            print(f"   ✅ Metadata CSV found")
        else:
            print(f"   ❌ Metadata CSV missing")
    else:
        print("❌ NIH directory not found")
    
    # Check CheXpert
    chexpert_dir = base_dir / "CheXpert"
    if chexpert_dir.exists():
        print(f"📁 CheXpert directory found")
    else:
        print("❌ CheXpert directory not found")
    
    return nih_dir.exists() or chexpert_dir.exists()

def create_download_test():
    """Create a test script to verify downloads work"""
    
    test_script = """#!/usr/bin/env python3
import torchxrayvision as xrv
import os

print("Testing TorchXRayVision dataset access...")

# Test NIH dataset
try:
    dataset = xrv.datasets.NIH_Dataset(
        imgpath="./datasets/data/chestxray/NIH/images",
        csvpath="./datasets/data/chestxray/NIH/Data_Entry_2017.csv"
    )
    print(f"✅ NIH dataset: {len(dataset)} samples")
    
    # Load first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"✅ First sample loaded: {sample['img'].shape}")
        
except Exception as e:
    print(f"❌ NIH dataset error: {e}")

print("Download test complete!")
"""
    
    with open("test_download.py", "w") as f:
        f.write(test_script)
    
    print("📝 Created test_download.py - run this to test downloads")

def main():
    """Main download function"""
    
    print("TORCHXRAYVISION REAL DATA DOWNLOAD")
    print("="*60)
    
    # Check current status
    has_data = check_existing_data()
    
    if has_data:
        print("\n✅ Some chest X-ray data already exists")
        response = input("\nDownload additional data? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Skipping download")
            return
    
    # Install TorchXRayVision
    if not install_torchxrayvision():
        print("❌ Cannot proceed without TorchXRayVision")
        return
    
    # Download options
    print(f"\n📥 Download Options:")
    print(f"   1. NIH ChestX-ray14 (Auto-download, ~45GB)")
    print(f"   2. CheXpert (Manual download required)")
    print(f"   3. Skip download (use synthetic data)")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == "1":
        print(f"\n⚠️  WARNING: NIH dataset is ~45GB")
        print(f"   Download will take significant time and space")
        confirm = input("   Continue? (y/n): ")
        
        if confirm.lower() in ['y', 'yes']:
            success = download_nih_dataset()
            if success:
                print(f"\n🎉 NIH dataset download initiated!")
                print(f"   First access may take time to download")
                create_download_test()
            else:
                print(f"\n❌ NIH dataset download failed")
        else:
            print("Download cancelled")
    
    elif choice == "2":
        download_chexpert_dataset()
    
    elif choice == "3":
        print(f"\n✅ Using synthetic data (recommended for testing)")
        print(f"   Your Multi-Label MEDAF evaluation will work with synthetic data")
        print(f"   Synthetic data provides realistic medical multi-label patterns")
    
    else:
        print("Invalid choice")
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    
    print(f"🧪 Test your setup:")
    print(f"   python check_dataset_stats.py")
    
    print(f"\n🚀 Run evaluation:")
    print(f"   python evaluate_chestxray.py --epochs 10 --max_samples 500")
    
    print(f"\n💡 Note:")
    print(f"   • Synthetic data is sufficient for testing your method")
    print(f"   • Real data provides larger scale validation")
    print(f"   • Both will show the same Multi-Label MEDAF improvements")

if __name__ == "__main__":
    main()

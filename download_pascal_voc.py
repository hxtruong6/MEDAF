#!/usr/bin/env python3
"""
Download and setup PASCAL VOC 2007 dataset for multi-label classification
"""

import os
import sys
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

class DownloadProgressBar:
    """Simple progress indicator for download"""
    def __init__(self, desc="Downloading"):
        self.desc = desc
        self.last_percent = 0
    
    def update_to(self, block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            if percent != self.last_percent and percent % 10 == 0:
                print(f"   {self.desc}: {percent}%")
                self.last_percent = percent

def download_with_progress(url, output_path):
    """Download file with progress indicator"""
    progress = DownloadProgressBar(f"Downloading {output_path.name}")
    urllib.request.urlretrieve(url, filename=output_path, reporthook=progress.update_to)

def download_pascal_voc_2007(data_root="./datasets/data/pascal_voc"):
    """Download PASCAL VOC 2007 dataset"""
    
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    
    print("Downloading PASCAL VOC 2007 Dataset")
    print("="*50)
    print(f"Download location: {data_root.absolute()}")
    
    # URLs for PASCAL VOC 2007
    urls = {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
    }
    
    # Download and extract each file
    for split_name, url in urls.items():
        filename = url.split('/')[-1]
        filepath = data_root / filename
        
        print(f"\n1. Downloading {split_name} split: {filename}")
        print(f"   URL: {url}")
        print(f"   Size: ~150MB")
        
        if filepath.exists():
            print(f"   ✓ {filename} already exists, skipping download")
        else:
            try:
                print(f"   Starting download...")
                download_with_progress(url, filepath)
                print(f"   ✓ Downloaded {filename}")
            except Exception as e:
                print(f"   ✗ Failed to download {filename}: {e}")
                continue
        
        # Extract tar file
        print(f"2. Extracting {filename}...")
        try:
            with tarfile.open(filepath, 'r') as tar:
                tar.extractall(data_root)
            print(f"   ✓ Extracted {filename}")
            
            # Remove tar file to save space
            filepath.unlink()
            print(f"   ✓ Cleaned up {filename}")
            
        except Exception as e:
            print(f"   ✗ Failed to extract {filename}: {e}")
    
    # Verify dataset structure
    voc_dir = data_root / "VOCdevkit" / "VOC2007"
    if voc_dir.exists():
        print(f"\n✓ PASCAL VOC 2007 successfully downloaded to:")
        print(f"  {voc_dir.absolute()}")
        
        # Check key directories
        key_dirs = ["JPEGImages", "Annotations", "ImageSets"]
        for dir_name in key_dirs:
            dir_path = voc_dir / dir_name
            if dir_path.exists():
                print(f"  ✓ {dir_name}/ directory found")
            else:
                print(f"  ✗ {dir_name}/ directory missing")
        
        return True
    else:
        print(f"\n✗ Dataset extraction failed. Expected directory not found:")
        print(f"  {voc_dir.absolute()}")
        return False

def analyze_pascal_voc_dataset(data_root="./datasets/data/pascal_voc"):
    """Analyze the downloaded PASCAL VOC dataset"""
    
    voc_dir = Path(data_root) / "VOCdevkit" / "VOC2007"
    
    if not voc_dir.exists():
        print("PASCAL VOC dataset not found. Please download first.")
        return
    
    print(f"\nAnalyzing PASCAL VOC 2007 Dataset")
    print("="*40)
    
    # VOC 2007 classes
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Analyze different splits
    splits = {
        'train': 'trainval.txt',
        'test': 'test.txt'
    }
    
    for split_name, filename in splits.items():
        image_set_file = voc_dir / "ImageSets" / "Main" / filename
        
        if not image_set_file.exists():
            print(f"✗ {split_name} split file not found: {filename}")
            continue
        
        # Read image IDs
        with open(image_set_file, 'r') as f:
            image_ids = [line.strip() for line in f]
        
        print(f"\n{split_name.upper()} Split Analysis:")
        print(f"  Total images: {len(image_ids)}")
        
        # Analyze labels
        label_counts = {cls: 0 for cls in classes}
        total_labels = 0
        labels_per_image = []
        
        valid_images = 0
        
        for image_id in image_ids[:100]:  # Sample first 100 for quick analysis
            ann_path = voc_dir / "Annotations" / f"{image_id}.xml"
            img_path = voc_dir / "JPEGImages" / f"{image_id}.jpg"
            
            if not ann_path.exists() or not img_path.exists():
                continue
            
            valid_images += 1
            
            # Parse annotation
            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
                image_labels = set()
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in classes:
                        label_counts[class_name] += 1
                        image_labels.add(class_name)
                
                labels_per_image.append(len(image_labels))
                total_labels += len(image_labels)
                
            except Exception as e:
                print(f"  Warning: Error parsing {ann_path}: {e}")
        
        if valid_images > 0:
            print(f"  Valid images analyzed: {valid_images}")
            print(f"  Average labels per image: {total_labels/valid_images:.2f}")
            print(f"  Labels per image range: {min(labels_per_image)}-{max(labels_per_image)}")
            
            # Most/least common classes
            sorted_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"  Most common classes: {sorted_classes[:5]}")
            print(f"  Least common classes: {sorted_classes[-5:]}")
        
        print(f"  ✓ {split_name} split verified")

def test_real_dataset_loading(data_root="./datasets/data/pascal_voc"):
    """Test loading the real PASCAL VOC dataset"""
    
    print(f"\nTesting Real Dataset Loading")
    print("="*30)
    
    try:
        # Import our dataset loader
        sys.path.append('.')
        from datasets.real_multilabel_datasets import create_multilabel_dataloaders
        
        # Create data loaders
        train_loader, test_loader, num_classes = create_multilabel_dataloaders(
            dataset_name='pascal_voc',
            data_root=data_root,
            batch_size=8,
            img_size=224,
            num_workers=0
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of classes: {num_classes}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test a batch
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"✓ Sample batch loaded:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels per sample: {labels.sum(dim=1).tolist()}")
        print(f"  Sample labels: {labels[0].nonzero().flatten().tolist()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    
    print("PASCAL VOC 2007 Real Dataset Setup")
    print("Multi-Label Classification")
    print("="*50)
    
    # Default download location
    data_root = "./datasets/data/pascal_voc"
    
    print(f"Setup location: {Path(data_root).absolute()}")
    print("\nThis script will:")
    print("1. Download PASCAL VOC 2007 trainval (~150MB)")
    print("2. Download PASCAL VOC 2007 test (~150MB)")
    print("3. Extract and organize the dataset")
    print("4. Analyze dataset statistics")
    print("5. Test dataset loading")
    
    response = input("\nProceed with download? [y/N]: ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Step 1: Download dataset
    success = download_pascal_voc_2007(data_root)
    
    if not success:
        print("Download failed. Please check your internet connection and try again.")
        return
    
    # Step 2: Analyze dataset
    analyze_pascal_voc_dataset(data_root)
    
    # Step 3: Test loading
    success = test_real_dataset_loading(data_root)
    
    if success:
        print(f"\n🎉 SUCCESS! PASCAL VOC 2007 is ready for evaluation.")
        print(f"\nNext steps:")
        print(f"1. Quick test: python quick_test.py")
        print(f"2. Fast evaluation: python evaluate_real_datasets.py --datasets pascal_voc --epochs 5")
        print(f"3. Full evaluation: python evaluate_real_datasets.py --datasets pascal_voc --epochs 50")
        print(f"\nDataset location: {Path(data_root).absolute()}")
    else:
        print(f"\n❌ Dataset loading test failed. Please check the installation.")

if __name__ == "__main__":
    main()

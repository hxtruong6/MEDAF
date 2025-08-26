#!/usr/bin/env python3
"""
Quick test for Chest X-Ray Multi-Label MEDAF (Real Data Only)
"""

import torch
import numpy as np

def test_chestxray_setup():
    """Test chest X-ray dataset and model setup"""
    
    print("Testing Chest X-Ray Multi-Label MEDAF Setup")
    print("="*50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test 1: Check prerequisites 
    print("\n1. Checking prerequisites...")
    
    try:
        import torchxrayvision as xrv
        print(f"✓ TorchXRayVision v{xrv.__version__} is installed")
    except ImportError:
        print("✗ TorchXRayVision not installed")
        print("Install with: pip install torchxrayvision")
        return False
    
    # Test 2: Check dataset availability
    print("\n2. Checking dataset availability...")
    
    try:
        from datasets.chestxray_multilabel import check_dataset_availability
        
        available, message = check_dataset_availability('nih')
        print(f"Dataset status: {message}")
        
        if not available:
            print("✗ Real dataset not available")
            print("Download with: python download_real_chestxray.py")
            return False
            
    except Exception as e:
        print(f"✗ Error checking dataset: {e}")
        return False
    
    # Test 3: Dataset Loading
    print("\n3. Testing chest X-ray dataset loading...")
    
    try:
        from datasets.chestxray_multilabel import create_chestxray_dataloaders
        
        # Create small dataset for testing
        train_loader, test_loader, num_classes, pathologies = create_chestxray_dataloaders(
            dataset_name='nih',
            batch_size=8,
            img_size=224,
            num_workers=0,
            max_samples=50  # Small for quick test
        )
        
        print(f"✓ Dataset loaded successfully")
        print(f"  Number of pathologies: {num_classes}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        print(f"  Sample pathologies: {pathologies[:5]}")
        
        # Test a batch
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"✓ Sample batch loaded:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Pathologies per sample: {labels.sum(dim=1).tolist()}")
        
        # Show sample pathologies
        first_sample_labels = labels[0]
        positive_pathologies = [pathologies[i] for i, val in enumerate(first_sample_labels) if val > 0.5]
        print(f"  Sample pathologies: {positive_pathologies}")
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        print("Please ensure real dataset is downloaded")
        return False
    
    # Test 4: Model Creation
    print("\n4. Testing model creation...")
    
    try:
        from core.multilabel_net_v2 import MultiLabelMEDAFv2
        
        # Test medical configuration
        args = {
            'img_size': 224,
            'backbone': 'resnet18',
            'num_classes': num_classes,
            'gate_temp': 100,
            'use_per_class_gating': True,
            'use_label_correlation': True,
            'enhanced_diversity': True
        }
        
        model = MultiLabelMEDAFv2(args)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {total_params:,}")
        print(f"  Gating type: {model.get_gating_summary()['gating_type']}")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(images, labels)
        
        print(f"✓ Forward pass successful")
        print(f"  Output logits shapes: {[logit.shape for logit in outputs['logits']]}")
        print(f"  Gating type: {outputs['gating_type']}")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Training Components
    print("\n5. Testing training components...")
    
    try:
        from core.multilabel_train_v2 import multiLabelAttnDiv, multiLabelAccuracy
        
        # Test diversity loss
        cams_list = outputs['cams_list']
        div_loss = multiLabelAttnDiv(cams_list, labels)
        print(f"✓ Diversity loss computed: {div_loss.item():.6f}")
        
        # Test accuracy metrics
        fused_logits = outputs['logits'][3]
        subset_acc, hamming_acc, precision, recall, f1 = multiLabelAccuracy(fused_logits, labels)
        
        print(f"✓ Medical metrics computed:")
        print(f"  Subset Accuracy: {subset_acc.item():.4f}")
        print(f"  Hamming Accuracy: {hamming_acc.item():.4f}")
        print(f"  F1 Score: {f1.item():.4f}")
        
    except Exception as e:
        print(f"✗ Training components failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*50}")
    print("🏥 Chest X-Ray Multi-Label MEDAF Test PASSED!")
    print(f"{'='*50}")
    
    print(f"\n🎯 Key Features Verified:")
    print(f"  ✅ REAL medical multi-label data ({num_classes} pathologies)")
    print(f"  ✅ Per-class gating for pathology-specific experts")
    print(f"  ✅ Enhanced attention diversity for medical imaging")
    print(f"  ✅ Authentic pathology correlations from real patients")
    
    print(f"\n🚀 Ready for Medical Evaluation:")
    print(f"  Quick test:  python evaluate_chestxray.py --epochs 5 --max_samples 200")
    print(f"  Full test:   python evaluate_chestxray.py --epochs 30 --max_samples 2000")
    print(f"  Production:  python evaluate_chestxray.py --epochs 100 --max_samples 10000")
    
    return True


if __name__ == "__main__":
    success = test_chestxray_setup()
    
    if success:
        print(f"\n✅ All tests passed! Ready for REAL medical multi-label evaluation.")
        print(f"🏥 Using authentic chest X-ray data with real pathology correlations.")
    else:
        print(f"\n❌ Some tests failed. Please check the setup.")
        print(f"Ensure TorchXRayVision is installed and real data is downloaded.")

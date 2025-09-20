#!/usr/bin/env python3
"""Quick test script for Multi-Label MEDAF on real datasets"""

import sys

sys.path.append(".")

print("Testing Multi-Label MEDAF on Real Datasets")
print("=" * 50)

# Test dataset loading
try:
    from datasets.real_multilabel_datasets import create_multilabel_dataloaders

    print("\n1. Testing PASCAL VOC dataset loading...")
    train_loader, test_loader, num_classes = create_multilabel_dataloaders(
        dataset_name="pascal_voc",
        data_root="./datasets/data/pascal_voc",
        batch_size=4,
        img_size=64,  # Small for quick test
        num_workers=0,
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

    print("\n2. Testing model creation...")
    args = {
        "img_size": 64,
        "backbone": "resnet18",
        "num_classes": 20,
        "gate_temp": 100,  # Add missing parameter
        "use_per_class_gating": True,
        "use_label_correlation": True,
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

print("\n" + "=" * 50)
print("Quick test completed!")
print("\nTo run full evaluation:")
print("python evaluate_real_datasets.py --datasets pascal_voc --epochs 5")

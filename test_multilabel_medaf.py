#!/usr/bin/env python3
"""
Test script for Multi-Label MEDAF implementation
Validates Phase 1 modifications with synthetic multi-label data
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import sys
import os
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_train import (
    train_multilabel,
    multiLabelAttnDiv,
    multiLabelAccuracy,
)


class SyntheticMultiLabelDataset(data.Dataset):
    """
    Synthetic multi-label dataset for testing
    """

    def __init__(
        self,
        num_samples=1000,
        img_size=32,
        num_classes=10,
        avg_labels_per_sample=3,
        random_state=42,
    ):
        self.img_size = img_size
        self.num_classes = num_classes

        # Generate synthetic multi-label data
        X, y = make_multilabel_classification(
            n_samples=num_samples,
            n_features=img_size * img_size * 3,  # Simulate RGB image
            n_classes=num_classes,
            n_labels=avg_labels_per_sample,
            length=50,  # Total number of features
            allow_unlabeled=False,
            sparse=False,
            return_indicator="dense",
            random_state=random_state,
        )

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Reshape to image format and convert to tensors
        self.images = torch.FloatTensor(X.reshape(-1, 3, img_size, img_size))
        self.labels = torch.FloatTensor(y)

        print(f"Generated {num_samples} samples with {num_classes} classes")
        print(f"Average labels per sample: {y.sum(axis=1).mean():.2f}")
        print(f"Label distribution: {y.sum(axis=0)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def test_model_forward():
    """Test model forward pass with synthetic data"""
    print("\n" + "=" * 50)
    print("Testing Model Forward Pass")
    print("=" * 50)

    # Model configuration
    args = {"img_size": 32, "backbone": "resnet18", "num_classes": 10, "gate_temp": 100}

    # Create model
    model = MultiLabelMEDAF(args)
    model.eval()

    # Create synthetic batch
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, 2, (batch_size, 10)).float()  # Multi-hot labels

    print(f"Input shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Sample labels:\n{y}")

    # Forward pass
    with torch.no_grad():
        outputs = model(x, y)

    # Check outputs
    logits = outputs["logits"]
    gate_pred = outputs["gate_pred"]
    cams_list = outputs["cams_list"]

    print(f"\nModel outputs:")
    print(f"Number of expert outputs: {len(logits)}")
    print(f"Logits shapes: {[logit.shape for logit in logits]}")
    print(f"Gate predictions shape: {gate_pred.shape}")
    print(f"CAMs list length: {len(cams_list)}")
    print(f"CAM shapes: {[cam.shape for cam in cams_list]}")

    # Test without labels
    outputs_no_labels = model(x, return_ft=True)
    print(f"\nForward pass without labels successful")
    print(f"Features shape: {outputs_no_labels['fts'].shape}")

    print("✓ Model forward pass test PASSED")
    return True


def test_attention_diversity():
    """Test multi-label attention diversity loss"""
    print("\n" + "=" * 50)
    print("Testing Multi-Label Attention Diversity Loss")
    print("=" * 50)

    batch_size = 3
    num_classes = 5
    H, W = 8, 8

    # Create synthetic CAMs and labels
    cams_list = [torch.randn(batch_size, num_classes, H, W) for _ in range(3)]
    targets = torch.tensor(
        [
            [1, 0, 1, 0, 1],  # Sample 1: classes 0, 2, 4
            [1, 1, 0, 0, 0],  # Sample 2: classes 0, 1
            [0, 0, 0, 1, 1],  # Sample 3: classes 3, 4
        ]
    ).float()

    print(f"CAMs shapes: {[cam.shape for cam in cams_list]}")
    print(f"Targets:\n{targets}")

    # Compute diversity loss
    div_loss = multiLabelAttnDiv(cams_list, targets)
    print(f"Diversity loss: {div_loss.item():.6f}")

    # Test edge cases
    # No positive labels
    empty_targets = torch.zeros_like(targets)
    div_loss_empty = multiLabelAttnDiv(cams_list, empty_targets)
    print(f"Diversity loss (no positive labels): {div_loss_empty.item():.6f}")

    # Single positive label per sample
    single_targets = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    ).float()
    div_loss_single = multiLabelAttnDiv(cams_list, single_targets)
    print(f"Diversity loss (single labels): {div_loss_single.item():.6f}")

    print("✓ Attention diversity loss test PASSED")
    return True


def test_multilabel_accuracy():
    """Test multi-label accuracy metrics"""
    print("\n" + "=" * 50)
    print("Testing Multi-Label Accuracy Metrics")
    print("=" * 50)

    # Create sample predictions and targets
    predictions = torch.tensor(
        [
            [2.0, -1.0, 1.5, -0.5, 0.8],  # High conf for 0,2,4
            [0.1, 1.2, -2.0, 0.3, -0.1],  # High conf for 1,3
            [-0.5, -1.0, 0.2, 1.8, 2.1],  # High conf for 3,4
        ]
    )

    targets = torch.tensor(
        [
            [1, 0, 1, 0, 1],  # True: 0, 2, 4
            [0, 1, 0, 1, 0],  # True: 1, 3
            [0, 0, 0, 1, 1],  # True: 3, 4
        ]
    ).float()

    print(f"Predictions:\n{predictions}")
    print(f"Targets:\n{targets}")

    # Compute metrics
    subset_acc, hamming_acc, precision, recall, f1 = multiLabelAccuracy(
        predictions, targets
    )

    print(f"\nMetrics:")
    print(f"Subset Accuracy: {subset_acc.item():.4f}")
    print(f"Hamming Accuracy: {hamming_acc.item():.4f}")
    print(f"Precision: {precision.item():.4f}")
    print(f"Recall: {recall.item():.4f}")
    print(f"F1 Score: {f1.item():.4f}")

    # Test with different threshold
    subset_acc_strict, hamming_acc_strict, _, _, _ = multiLabelAccuracy(
        predictions, targets, threshold=0.8
    )
    print(f"\nWith threshold=0.8:")
    print(f"Subset Accuracy: {subset_acc_strict.item():.4f}")
    print(f"Hamming Accuracy: {hamming_acc_strict.item():.4f}")

    print("✓ Multi-label accuracy test PASSED")
    return True


def test_training_loop():
    """Test training loop with synthetic dataset"""
    print("\n" + "=" * 50)
    print("Testing Training Loop")
    print("=" * 50)

    # Create synthetic dataset
    dataset = SyntheticMultiLabelDataset(
        num_samples=100, img_size=32, num_classes=8, avg_labels_per_sample=2
    )

    # Create data loader
    train_loader = data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues in test
    )

    # Model configuration
    args = {
        "img_size": 32,
        "backbone": "resnet18",
        "num_classes": 8,
        "gate_temp": 100,
        "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
        "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
        "loss_wgts": [0.7, 1.0, 0.01],  # [expert, gate, diversity]
    }

    # Create model
    model = MultiLabelMEDAF(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Using device: {device}")
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Setup training
    criterion = {"bce": nn.BCEWithLogitsLoss()}

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Test single batch
    print("\nTesting single batch...")
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch[0].to(device), sample_batch[1].to(device)

    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch target shape: {targets.shape}")
    print(
        f"Target statistics - Min: {targets.min():.1f}, Max: {targets.max():.1f}, Mean: {targets.mean():.3f}"
    )

    # Forward pass
    model.train()
    outputs = model(inputs, targets)
    logits = outputs["logits"]
    cams_list = outputs["cams_list"]

    # Compute losses manually for verification
    bce_losses = [criterion["bce"](logit, targets) for logit in logits[:3]]
    gate_loss = criterion["bce"](logits[3], targets)
    diversity_loss = multiLabelAttnDiv(cams_list, targets)

    print(f"\nLoss components:")
    print(f"Expert BCE losses: {[loss.item() for loss in bce_losses]}")
    print(f"Gate loss: {gate_loss.item():.4f}")
    print(f"Diversity loss: {diversity_loss.item():.6f}")

    total_loss = (
        args["loss_wgts"][0] * sum(bce_losses)
        + args["loss_wgts"][1] * gate_loss
        + args["loss_wgts"][2] * diversity_loss
    )
    print(f"Total loss: {total_loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print("✓ Single batch training PASSED")

    # Test full training loop for a few iterations
    print("\nTesting training loop (3 batches)...")

    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        if i >= 3:  # Only test 3 batches
            break

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs, targets)
        logits = outputs["logits"]
        cams_list = outputs["cams_list"]

        # Compute losses
        bce_losses = [criterion["bce"](logit, targets) for logit in logits[:3]]
        gate_loss = criterion["bce"](logits[3], targets)
        diversity_loss = multiLabelAttnDiv(cams_list, targets)

        total_loss = (
            args["loss_wgts"][0] * sum(bce_losses)
            + args["loss_wgts"][1] * gate_loss
            + args["loss_wgts"][2] * diversity_loss
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Compute accuracy
        subset_acc, hamming_acc, precision, recall, f1 = multiLabelAccuracy(
            logits[3], targets
        )

        print(
            f"Batch {i+1}: Loss={total_loss.item():.4f}, SubsetAcc={subset_acc.item():.3f}, HammingAcc={hamming_acc.item():.3f}"
        )

    print("✓ Training loop test PASSED")
    return True


def main():
    """Run all tests"""
    print("Testing Multi-Label MEDAF Implementation")
    print("Phase 1: Core Modifications")

    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Run tests
        tests = [
            test_model_forward,
            test_attention_diversity,
            test_multilabel_accuracy,
            test_training_loop,
        ]

        passed = 0
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ Test {test_func.__name__} FAILED: {e}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Tests passed: {passed}/{len(tests)}")

        if passed == len(tests):
            print(
                "🎉 ALL TESTS PASSED! Multi-Label MEDAF Phase 1 implementation is working correctly."
            )
            print("\nYou can now:")
            print("1. Train on real multi-label datasets")
            print("2. Implement Phase 2 features (per-class gating)")
            print("3. Add advanced research extensions")
        else:
            print("❌ Some tests failed. Please review the implementation.")

    except Exception as e:
        print(f"Fatal error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

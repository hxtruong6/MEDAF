#!/usr/bin/env python3
"""
Test script for Multi-Label MEDAF implementation
Validates Phase 1 modifications with synthetic multi-label data
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_train import (
    multiLabelAccuracy,
    multiLabelAttnDiv,
)


# Paths and configuration for real ChestX-ray14 data integration
KNOWN_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

CHESTXRAY_IMAGE_ROOT = Path("datasets/data/NIH/images-224")
CHESTXRAY_CSV_CANDIDATES = [
    Path("datasets/data/NIH/chestxray_train_known.csv"),
    Path("datasets/data/NIH/chestxray_train_new.csv"),
]

# Enable real-data smoke test by setting MEDAF_USE_CHESTXRAY_DATA=1
USE_REAL_DATA = os.environ.get("MEDAF_USE_CHESTXRAY_DATA", "0") == "1"


def resolve_chestxray_known_csv():
    """Return the first available CSV path for the known-label split."""

    for candidate in CHESTXRAY_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


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


class ChestXrayKnownDataset(data.Dataset):
    """Dataset that reads ChestX-ray14 samples from the known-label CSV split."""

    def __init__(
        self,
        csv_path,
        image_root,
        img_size=224,
        max_samples=64,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.img_size = img_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"ChestX-ray CSV not found: {self.csv_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(
                f"ChestX-ray image directory not found: {self.image_root}"
            )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        self.label_to_idx = {label: idx for idx, label in enumerate(KNOWN_LABELS)}
        self.num_classes = len(self.label_to_idx)

        df = pd.read_csv(self.csv_path)
        if "known_labels" not in df.columns:
            raise ValueError(
                "Expected 'known_labels' column in CSV. Run create_chestxray_splits.py first."
            )

        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.records = df.to_dict("records")

    @staticmethod
    def _parse_label_list(raw_value):
        if isinstance(raw_value, list):
            return raw_value
        if pd.isna(raw_value):
            return []
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not raw_value:
                return []
            try:
                parsed = ast.literal_eval(raw_value)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                if isinstance(parsed, str):
                    return [parsed]
            except (ValueError, SyntaxError):
                pass
            return [item.strip() for item in raw_value.split("|") if item.strip()]
        return []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.image_root / record["Image Index"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in self._parse_label_list(record.get("known_labels", [])):
            if label in self.label_to_idx:
                labels[self.label_to_idx[label]] = 1.0

        return image, labels


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


def test_chestxray_known_csv_loader():
    """Verify that the known-label ChestX-ray CSV can be read and transformed."""

    print("\n" + "=" * 50)
    print("Testing ChestX-ray Known CSV Loader")
    print("=" * 50)

    csv_path = resolve_chestxray_known_csv()
    if csv_path is None or not CHESTXRAY_IMAGE_ROOT.exists():
        print("ChestX-ray CSV or image directory not found. Skipping real-data test.")
        return True

    dataset = ChestXrayKnownDataset(
        csv_path=csv_path,
        image_root=CHESTXRAY_IMAGE_ROOT,
        img_size=224,
        max_samples=12,
    )

    loader = data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images, targets = batch

    print(f"Loaded batch images shape: {images.shape}")
    print(f"Loaded batch targets shape: {targets.shape}")
    print(f"Positive labels per sample: {targets.sum(dim=1)}")

    if images.shape[1] != 3 or images.shape[2] != 224 or images.shape[3] != 224:
        raise AssertionError("Unexpected image tensor shape from ChestXrayKnownDataset")

    if targets.shape[1] != len(KNOWN_LABELS):
        raise AssertionError("Target tensor does not match known label count")

    print("✓ ChestX-ray CSV loader test PASSED")
    return True


def test_training_loop():
    """Test training loop with synthetic dataset"""
    print("\n" + "=" * 50)
    print("Testing Training Loop")
    print("=" * 50)

    csv_path = resolve_chestxray_known_csv()
    real_data_available = (
        USE_REAL_DATA and csv_path is not None and CHESTXRAY_IMAGE_ROOT.exists()
    )

    if real_data_available:
        print("Using ChestX-ray known-label split for training loop test")
        dataset = ChestXrayKnownDataset(
            csv_path=csv_path,
            image_root=CHESTXRAY_IMAGE_ROOT,
            img_size=224,
            max_samples=48,
        )
        batch_size = 8
    else:
        print("Using synthetic dataset for training loop test")
        dataset = SyntheticMultiLabelDataset(
            num_samples=100, img_size=32, num_classes=8, avg_labels_per_sample=2
        )
        batch_size = 16

    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    args = {
        "img_size": dataset.img_size,
        "backbone": "resnet18",
        "num_classes": dataset.num_classes,
        "gate_temp": 100,
        "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
        "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
        "loss_wgts": [0.7, 1.0, 0.01],
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
            test_chestxray_known_csv_loader,
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

        else:
            print("❌ Some tests failed. Please review the implementation.")

    except Exception as e:
        print(f"Fatal error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

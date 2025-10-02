"""
Test script for Multi-Label Novelty Detection with MEDAF

This script demonstrates how to use the novelty detection functionality
with your existing MEDAF model.

Usage:
    python test_novelty_detection.py
"""

import torch
import torch.utils.data as data
from pathlib import Path
import numpy as np

# Import your existing modules
from core.multilabel_net import MultiLabelMEDAF
from test_multilabel_medaf import ChestXrayKnownDataset


def test_novelty_detection():
    """
    Test the novelty detection functionality with a simple example.
    """
    print("🧪 Testing Multi-Label Novelty Detection")
    print("=" * 50)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Create a simple test dataset
    print("\n📊 Creating test dataset...")

    # Use your existing dataset class
    csv_path = "datasets/data/chestxray/NIH/chestxray_train_known.csv"
    image_root = "datasets/data/chestxray/NIH/images-224"

    try:
        dataset = ChestXrayKnownDataset(
            csv_path=csv_path,
            image_root=image_root,
            img_size=224,
            max_samples=100,  # Use only 100 samples for quick testing
        )
        print(f"   ✅ Dataset loaded: {len(dataset)} samples")

        # Create data loader
        test_loader = data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=1
        )

        # Create a simple model (for testing purposes)
        print("\n🏗️  Creating model...")
        args = {
            "img_size": 224,
            "backbone": "resnet18",
            "num_classes": dataset.num_classes,
            "gate_temp": 100,
        }

        model = MultiLabelMEDAF(args)
        model.to(device)
        model.eval()
        print(
            f"   ✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        print("   ⚠️  Note: This is an untrained model for testing purposes only")
        print("   📝 For real novelty detection, use a trained model checkpoint")

        # Test novelty detection
        print("\n🔍 Testing novelty detection...")

        # Get a batch of test data
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx >= 2:  # Test only first 2 batches
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            print(f"\n   Batch {batch_idx + 1}:")
            print(f"   Input shape: {inputs.shape}")
            print(f"   Target shape: {targets.shape}")

            # Test novelty detection
            novelty_results = model.detect_novelty(inputs)

            # Display results
            is_novel = novelty_results["is_novel"]
            novelty_scores = novelty_results["novelty_scores"]
            novelty_types = novelty_results["novelty_types"]
            predictions = novelty_results["predictions"]

            print(f"   Novelty detection results:")
            print(f"     Novel samples: {is_novel.sum().item()}/{len(is_novel)}")
            print(f"     Novelty scores: {novelty_scores.cpu().numpy()}")
            print(f"     Novelty types: {novelty_types}")

            # Show prediction details
            for i in range(len(inputs)):
                sample_predictions = predictions[i].cpu().numpy()
                sample_targets = targets[i].cpu().numpy()
                predicted_labels = np.where(sample_predictions > 0.5)[0]
                true_labels = np.where(sample_targets > 0.5)[0]

                print(f"     Sample {i}:")
                print(f"       True labels: {true_labels}")
                print(f"       Predicted labels: {predicted_labels}")
                print(f"       Is novel: {is_novel[i].item()}")
                print(f"       Novelty type: {novelty_types[i]}")
                print(f"       Novelty score: {novelty_scores[i].item():.4f}")

        print("\n✅ Novelty detection test completed successfully!")

        # Test calibration
        print("\n🎯 Testing detector calibration...")

        # Create a small validation loader for calibration
        val_size = min(20, len(dataset))
        val_dataset, _ = data.random_split(
            dataset,
            [val_size, len(dataset) - val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = data.DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Calibrate detector
        detector = model.calibrate_novelty_detector(val_loader, device, fpr_target=0.05)

        print(f"   ✅ Detector calibrated with threshold: {detector.threshold:.4f}")

        # Test with calibrated detector
        print("\n🔍 Testing with calibrated detector...")

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx >= 1:  # Test only first batch
                break

            inputs = inputs.to(device)

            # Use calibrated detector
            novelty_results = model.detect_novelty(inputs, novelty_detector=detector)

            is_novel = novelty_results["is_novel"]
            novelty_scores = novelty_results["novelty_scores"]

            print(f"   Novel samples detected: {is_novel.sum().item()}/{len(is_novel)}")
            print(f"   Novelty scores: {novelty_scores.cpu().numpy()}")

        print("\n🎉 All tests completed successfully!")
        print("\n📝 Key insights:")
        print("   - The hybrid score combines logit confidence and CAM diversity")
        print("   - Lower scores indicate more novel/unknown samples")
        print("   - The threshold separates known from unknown samples")
        print("   - Calibration is important for reliable detection")
        print("\n⚠️  Expected behavior with untrained model:")
        print("   - All samples may be classified as 'Known' (no novelty detected)")
        print("   - Novelty scores will be similar across samples")
        print("   - This is normal for an untrained model")
        print("   - Use a trained model for meaningful novelty detection")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Make sure you have:")
        print("   - Proper data paths for ChestX-ray dataset")
        print("   - All required dependencies installed")
        print("   - Sufficient memory for model loading")
        return False


def main():
    """
    Main function to run the novelty detection test.
    """
    print("🚀 Starting Multi-Label Novelty Detection Test")
    print("=" * 60)

    success = test_novelty_detection()

    if success:
        print(f"\n✅ Test completed successfully!")
        print(f"   Your MEDAF model now has novelty detection capabilities!")
        print(f"   You can use model.detect_novelty() to detect unknown samples.")
    else:
        print(f"\n❌ Test failed - check error messages above")


if __name__ == "__main__":
    main()

"""
Example: Multi-Label Novelty Detection with MEDAF

This script demonstrates how to integrate the novelty detection system
with your existing MEDAF model for detecting unknown samples in multi-label classification.

Usage:
    python example_novelty_detection.py
"""

import torch
import torch.utils.data as data
from pathlib import Path
import json

# Import your existing modules
from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_novelty_detection import (
    MultiLabelNoveltyDetector,
    evaluate_novelty_detection,
)
from test_multilabel_medaf import ChestXrayKnownDataset


def load_trained_model(checkpoint_path: str, device: torch.device):
    """
    Load a trained MEDAF model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load model on

    Returns:
        Loaded model and args
    """
    print(f"📁 Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    if not args:
        raise ValueError("Checkpoint missing 'args'")

    # Create model
    model = MultiLabelMEDAF(args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, args


def create_data_loaders(known_csv: str, image_root: str, batch_size: int = 32):
    """
    Create data loaders for known samples (for calibration and testing).

    Args:
        known_csv: Path to CSV with known samples
        image_root: Path to image directory
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"📊 Creating data loaders...")

    # Create dataset
    dataset = ChestXrayKnownDataset(
        csv_path=known_csv,
        image_root=image_root,
        img_size=224,
        max_samples=None,  # Use all samples
    )

    # Split into train/val/test (80/10/10)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def demonstrate_novelty_detection_workflow():
    """
    Complete workflow for novelty detection with MEDAF.

    This demonstrates:
    1. Loading a trained model
    2. Creating data loaders
    3. Calibrating the novelty detector
    4. Detecting novelty in test samples
    5. Evaluating performance
    """
    print("🔍 Multi-Label Novelty Detection Workflow")
    print("=" * 60)

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Paths (adjust these to your actual paths)
    checkpoint_path = "checkpoints/medaf_phase1/medaf_phase1_chestxray.pt"
    known_csv = "datasets/data/chestxray/NIH/chestxray_train_known.csv"
    image_root = "datasets/data/chestxray/NIH/images-224"

    try:
        # Step 1: Load trained model
        print("\n📥 Step 1: Loading trained model...")
        model, args = load_trained_model(checkpoint_path, device)

        # Step 2: Create data loaders
        print("\n📊 Step 2: Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            known_csv, image_root, batch_size=16
        )

        # Step 3: Create and calibrate novelty detector
        print("\n🎯 Step 3: Creating and calibrating novelty detector...")
        detector = MultiLabelNoveltyDetector(gamma=1.0, temperature=1.0)

        # Calibrate threshold on validation data (known samples only)
        print("   Calibrating threshold on validation data...")
        threshold = detector.calibrate_threshold(
            model, val_loader, device, fpr_target=0.05
        )
        print(f"   ✅ Threshold calibrated: {threshold:.4f}")

        # Step 4: Detect novelty in test samples
        print("\n🔍 Step 4: Detecting novelty in test samples...")
        model.eval()

        all_predictions = []
        all_novelty_scores = []
        all_is_novel = []
        all_novelty_types = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                output_dict = model(inputs, targets)
                gate_logits = output_dict["logits"][-1]
                cams_list = output_dict["cams_list"]

                # Convert to predictions
                probs = torch.sigmoid(gate_logits)
                predicted_labels = (probs > 0.5).float()

                # Detect novelty
                is_novel, novelty_scores = detector.detect_novelty(
                    gate_logits, cams_list, predicted_labels
                )

                # Classify novelty types
                novelty_types = detector.classify_novelty_type(
                    predicted_labels, is_novel
                )

                # Store results
                all_predictions.append(predicted_labels.cpu())
                all_novelty_scores.append(novelty_scores.cpu())
                all_is_novel.append(is_novel.cpu())
                all_novelty_types.extend(novelty_types)

                if batch_idx % 5 == 0:
                    print(f"   Processed batch {batch_idx + 1}")

        # Step 5: Analyze results
        print("\n📈 Step 5: Analyzing results...")

        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_novelty_scores = torch.cat(all_novelty_scores, dim=0)
        all_is_novel = torch.cat(all_is_novel, dim=0)

        # Compute statistics
        num_novel = all_is_novel.sum().item()
        num_known = len(all_is_novel) - num_novel
        novel_ratio = num_novel / len(all_is_novel)

        print(f"   Total samples: {len(all_is_novel)}")
        print(f"   Known samples: {num_known} ({100*(1-novel_ratio):.1f}%)")
        print(f"   Novel samples: {num_novel} ({100*novel_ratio:.1f}%)")

        # Analyze novelty types
        type_counts = {}
        for novelty_type in all_novelty_types:
            type_counts[novelty_type] = type_counts.get(novelty_type, 0) + 1

        print(f"\n   Novelty type distribution:")
        for novelty_type, count in type_counts.items():
            percentage = 100 * count / len(all_novelty_types)
            print(f"     {novelty_type}: {count} ({percentage:.1f}%)")

        # Show score statistics
        print(f"\n   Novelty score statistics:")
        print(f"     Mean score: {all_novelty_scores.mean():.4f}")
        print(f"     Std score: {all_novelty_scores.std():.4f}")
        print(f"     Min score: {all_novelty_scores.min():.4f}")
        print(f"     Max score: {all_novelty_scores.max():.4f}")

        # Step 6: Save results
        print("\n💾 Step 6: Saving results...")

        results = {
            "threshold": float(threshold),
            "num_samples": len(all_is_novel),
            "num_known": int(num_known),
            "num_novel": int(num_novel),
            "novel_ratio": float(novel_ratio),
            "novelty_types": type_counts,
            "score_stats": {
                "mean": float(all_novelty_scores.mean()),
                "std": float(all_novelty_scores.std()),
                "min": float(all_novelty_scores.min()),
                "max": float(all_novelty_scores.max()),
            },
        }

        results_path = "novelty_detection_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"   ✅ Results saved to: {results_path}")

        print("\n🎉 Novelty detection workflow completed successfully!")
        print("\n📝 Key insights:")
        print("   - The hybrid score combines logit confidence and CAM diversity")
        print("   - Lower scores indicate more novel/unknown samples")
        print("   - The threshold separates known from unknown samples")
        print("   - Mixed novelty is the most challenging to detect")

        return results

    except Exception as e:
        print(f"❌ Error in novelty detection workflow: {e}")
        print("💡 Make sure you have:")
        print("   - A trained MEDAF model checkpoint")
        print("   - Proper data paths for ChestX-ray dataset")
        print("   - All required dependencies installed")
        return None


def main():
    """
    Main function to run the novelty detection demonstration.
    """
    print("🚀 Starting Multi-Label Novelty Detection Demo")
    print("=" * 60)

    # Run the complete workflow
    results = demonstrate_novelty_detection_workflow()

    if results:
        print(f"\n✅ Demo completed successfully!")
        print(
            f"   Detected {results['num_novel']} novel samples out of {results['num_samples']} total"
        )
        print(f"   Novelty ratio: {results['novel_ratio']:.2%}")
    else:
        print(f"\n❌ Demo failed - check error messages above")


if __name__ == "__main__":
    main()

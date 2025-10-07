#!/usr/bin/env python3
"""
Enhanced MEDAF Training Script with Class Weighting and Threshold Optimization

This script demonstrates the improved training approach for handling class imbalance
in medical multi-label classification.

Key improvements:
1. Class-weighted BCE loss for imbalanced data
2. Per-class threshold optimization
3. Enhanced evaluation metrics
4. Better training configuration for medical data
"""

import os
import sys

# Set working directory
CURRENT_DIR = "/home/s2320437/WORK/aidan-medaf/"
os.chdir(CURRENT_DIR)
sys.path.append(CURRENT_DIR)

# Import the enhanced training
from multilabel_medaf import main, evaluation, config


def run_enhanced_training():
    """Run enhanced training with class weighting"""
    print("🚀 ENHANCED MEDAF TRAINING")
    print("=" * 60)
    print("Key improvements:")
    print("✅ Class-weighted BCE loss for imbalanced data")
    print("✅ Per-class threshold optimization")
    print("✅ Enhanced evaluation with medical metrics")
    print("✅ Improved training configuration")
    print("=" * 60)

    # Show current configuration
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Class Weighting: {config.get('use_class_weights', False)}")
    print(f"   Optimal Thresholds: {config.get('use_optimal_thresholds', False)}")
    print(f"   Enhanced Training: {config.get('enhanced_training', False)}")

    # Ask user for confirmation
    response = input(f"\n🤔 Start enhanced training? (y/n): ").lower().strip()
    if response != "y":
        print("Training cancelled.")
        return

    # Run training
    print(f"\n🎯 Starting enhanced training...")
    main()


def run_enhanced_evaluation():
    """Run enhanced evaluation with optimal thresholds"""
    print("🔍 ENHANCED MEDAF EVALUATION")
    print("=" * 60)
    print("Key improvements:")
    print("✅ Per-class threshold optimization")
    print("✅ Comprehensive medical evaluation metrics")
    print("✅ Class-specific performance analysis")
    print("=" * 60)

    # Run evaluation
    print(f"\n📊 Starting enhanced evaluation...")
    checkpoint_path = "/home/s2320437/WORK/aidan-medaf/checkpoints/medaf_phase1/medaf_phase1_chestxray_enhanced_epoch_29_1759726840.pt"
    evaluation(checkpoint_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced MEDAF Training and Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "both"],
        default="both",
        help="Mode: train, eval, or both",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with limited samples"
    )

    args = parser.parse_args()

    # Quick test configuration
    if args.quick:
        config["max_samples"] = 500  # Limit samples for quick testing
        config["num_epochs"] = 5  # Fewer epochs for testing
        print("🏃 Quick test mode enabled (limited samples and epochs)")

    if args.mode in ["train", "both"]:
        run_enhanced_training()

    if args.mode in ["eval", "both"]:
        run_enhanced_evaluation()

    print(
        f"\n🎉 Enhanced MEDAF {'training and evaluation' if args.mode == 'both' else args.mode} complete!"
    )

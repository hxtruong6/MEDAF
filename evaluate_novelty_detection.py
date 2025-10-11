#!/usr/bin/env python3
"""
Example script for evaluating MEDAF model with novelty detection.

This script demonstrates three evaluation modes:
1. Standard evaluation: Evaluate classification on known labels only
2. Novelty detection: Evaluate ability to detect unknown/novel samples
3. Comprehensive: Combine both evaluations for complete analysis

Usage:
    # Standard evaluation (known labels only)
    python evaluate_novelty_detection.py --mode eval --checkpoint path/to/checkpoint.pt

    # Novelty detection only
    python evaluate_novelty_detection.py --mode eval_novelty --checkpoint path/to/checkpoint.pt

    # Comprehensive evaluation (both)
    python evaluate_novelty_detection.py --mode eval_comprehensive --checkpoint path/to/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

from medaf_trainer import MEDAFTrainer


def main():
    parser = argparse.ArgumentParser(
        description="MEDAF Novelty Detection Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint for evaluation",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "eval_novelty", "eval_comprehensive"],
        default="eval_comprehensive",
        help=(
            "Evaluation mode:\n"
            "  eval: Standard evaluation on known labels\n"
            "  eval_novelty: Novelty detection on unknown samples\n"
            "  eval_comprehensive: Both known and novelty evaluation (default)"
        ),
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 70)
    print("🚀 MEDAF NOVELTY DETECTION EVALUATION")
    print("=" * 70)
    print(f"📁 Config: {args.config}")
    print(f"💾 Checkpoint: {args.checkpoint}")
    print(f"🎯 Mode: {args.mode}")
    print("=" * 70)

    try:
        # Create trainer
        trainer = MEDAFTrainer(args.config)

        # Run evaluation based on mode
        if args.mode == "eval":
            print("\n📊 Running standard evaluation on known labels...")
            results = trainer.evaluate(args.checkpoint)

        elif args.mode == "eval_novelty":
            print("\n🔍 Running novelty detection evaluation...")
            results = trainer.evaluate_novelty_detection(args.checkpoint)

        elif args.mode == "eval_comprehensive":
            print("\n📋 Running comprehensive evaluation (known + novelty)...")
            results = trainer.evaluate_comprehensive(args.checkpoint)

        print("\n✅ Evaluation completed successfully!")

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

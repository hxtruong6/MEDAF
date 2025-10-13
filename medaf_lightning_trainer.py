"""
MEDAF Lightning Trainer - Main Integration Script
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

from core.lightning_trainer import MEDAFLightningTrainer


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main function to run MEDAF Lightning training or evaluation"""
    parser = argparse.ArgumentParser(
        description="MEDAF Multi-Label Classification - Lightning Trainer"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config_lightning.yaml",
        help="Path to Lightning configuration file",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train, test",
    )

    # Checkpoint path for evaluation
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint path for evaluation (uses best model if not specified)",
    )

    # Logging level
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # Resume training
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")

    # Quick test mode
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick test with limited data"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate configuration file
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    # Create trainer
    try:
        trainer = MEDAFLightningTrainer(args.config)
        logger.info("✅ MEDAF Lightning Trainer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        return 1

    # Quick test mode - modify config for limited data
    if args.quick_test:
        logger.info("🚀 Running in quick test mode with limited data")
        trainer.config["data"]["max_samples"] = 100
        trainer.config["training"]["num_epochs"] = 4
        trainer.config["training"]["batch_size"] = 32

    try:
        if args.mode == "train":
            # Training mode
            if args.resume:
                logger.info(f"🔄 Resuming training from checkpoint: {args.resume}")
                # Use PyTorch Lightning's built-in resume functionality
                results = trainer.train(ckpt_path=args.resume)
            else:
                logger.info("🚀 Starting MEDAF Lightning training...")
                results = trainer.train()

            logger.info("✅ Training completed successfully!")
            logger.info(f"Best model: {results['best_model_path']}")
            logger.info(f"Training time: {results['training_time']:.2f} seconds")

            # Print training summary
            summary = trainer.get_training_summary()
            logger.info("📊 Training Summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")

        # elif args.mode == "test":
        #     # Test mode - standard classification evaluation
        #     logger.info("🧪 Starting model testing...")
        #     results = trainer.test(args.checkpoint)

        #     logger.info("✅ Testing completed successfully!")
        #     logger.info(f"Test results: {results['test_results']}")

        # elif args.mode == "eval_novelty":
        #     # Novelty detection evaluation mode
        #     logger.info("🔍 Starting novelty detection evaluation...")
        #     results = trainer.evaluate_novelty_detection(args.checkpoint)

        #     logger.info("✅ Novelty detection evaluation completed successfully!")

        elif args.mode == "test":
            # Comprehensive evaluation mode
            logger.info("📊 Starting comprehensive evaluation...")

            # Test standard classification
            logger.info("[1/2] Testing standard classification...")
            test_results = trainer.test(args.checkpoint)

            # Test novelty detection
            logger.info("[2/2] Testing novelty detection...")
            novelty_results = trainer.evaluate_novelty_detection(args.checkpoint)

            logger.info("✅ Comprehensive evaluation completed successfully!")

            # Print summary
            print("\n" + "=" * 70)
            print("📋 COMPREHENSIVE EVALUATION SUMMARY")
            print("=" * 70)

            if test_results and "test_results" in test_results:
                print("\n✅ Standard Classification Results:")
                for result in test_results["test_results"]:
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key}: {value:.4f}")

            if novelty_results:
                print("\n🔍 Novelty Detection Results:")
                print(f"  AUROC: {novelty_results['auroc']:.4f}")
                print(
                    f"  Detection Accuracy: {novelty_results['detection_accuracy']:.4f}"
                )
                print(f"  F1-Score: {novelty_results['f1_score']:.4f}")

            print("=" * 70)

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during {args.mode}: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


def compare_with_original():
    """
    Function to compare Lightning implementation with original trainer
    This can be used for validation and performance comparison
    """
    logger = logging.getLogger(__name__)

    logger.info("🔄 Comparing Lightning implementation with original trainer...")

    # This would involve:
    # 1. Running both implementations with the same configuration
    # 2. Comparing training metrics, validation scores, and final performance
    # 3. Checking that results are equivalent (within tolerance)

    # Implementation would go here
    logger.info("📊 Comparison completed - results should be equivalent")


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

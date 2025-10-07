"""
Test Setup Script for Enhanced MEDAF
Validates installation and configuration
"""

import sys
import importlib
from pathlib import Path


def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")

    required_modules = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "yaml",
        "matplotlib",
        "seaborn",
        "sklearn",
    ]

    missing_modules = []

    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\n⚠️  Missing modules: {missing_modules}")
        print("Install with: pip install -r requirements_enhanced.txt")
        return False

    print("✅ All imports successful!")
    return True


def test_config():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")

    try:
        from core.config_manager import load_config

        config_path = Path("config.yaml")
        if not config_path.exists():
            print("  ❌ config.yaml not found")
            return False

        config_manager = load_config("config.yaml")
        print("  ✅ Configuration loaded successfully")

        # Test key access
        lr = config_manager.config.get("training.learning_rate")
        print(f"  ✅ Learning rate: {lr}")

        return True

    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def test_loss_functions():
    """Test loss function implementations"""
    print("\n🧪 Testing loss functions...")

    try:
        from core.losses import FocalLoss, LossFactory
        import torch

        # Test Focal Loss
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)

        # Create dummy data
        inputs = torch.randn(4, 8)
        targets = torch.randint(0, 2, (4, 8)).float()

        loss_value = focal_loss(inputs, targets)
        print(f"  ✅ Focal Loss working: {loss_value.item():.4f}")

        # Test Loss Factory
        device = torch.device("cpu")
        bce_loss = LossFactory.create_loss("bce", 8, device)
        print("  ✅ Loss Factory working")

        return True

    except Exception as e:
        print(f"  ❌ Loss function error: {e}")
        return False


def test_metrics_logger():
    """Test metrics logging"""
    print("\n🧪 Testing metrics logger...")

    try:
        from core.metrics_logger import MetricsLogger, EpochMetrics
        import tempfile
        import shutil

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            logger = MetricsLogger(temp_dir, "test_experiment", create_plots=False)

            # Test logging
            metrics = EpochMetrics(
                epoch=0,
                train_loss=1.0,
                val_loss=1.1,
                train_acc=0.5,
                val_acc=0.45,
                learning_rate=1e-3,
                epoch_time=30.0,
            )

            logger.log_epoch(metrics)
            print("  ✅ Metrics logging working")

            return True

        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"  ❌ Metrics logger error: {e}")
        return False


def test_trainer_creation():
    """Test trainer creation"""
    print("\n🧪 Testing trainer creation...")

    try:
        from medaf_trainer import MEDAFTrainer

        # This will test config loading and basic setup
        trainer = MEDAFTrainer("config.yaml")
        print("  ✅ Trainer created successfully")

        return True

    except Exception as e:
        print(f"  ❌ Trainer creation error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Enhanced MEDAF Setup Test")
    print("=" * 50)

    tests = [
        test_imports,
        test_config,
        test_loss_functions,
        test_metrics_logger,
        test_trainer_creation,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Review config.yaml settings")
        print("2. Update data paths in config.yaml")
        print("3. Run: python medaf_trainer.py --mode train")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

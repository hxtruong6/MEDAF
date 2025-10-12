"""
PyTorch Lightning Trainer for MEDAF Multi-Label Classification
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from .lightning_module import MEDAFLightningModule
from .lightning_datamodule import MEDAFDataModule
from .lightning_callbacks import (
    MEDAFMetricsCallback,
    MEDAFThresholdOptimizationCallback,
    MEDAFNoveltyDetectionCallback,
    MEDAFROCCurveCallback,
    MEDAFModelCheckpointCallback,
)
from .config_manager import load_config


class MEDAFLightningTrainer:
    """
    PyTorch Lightning-based trainer for MEDAF Multi-Label Classification

    This trainer provides:
    - Simplified training loop with Lightning
    - Automatic mixed precision training
    - Multi-GPU support
    - Advanced logging and monitoring
    - Custom callbacks for MEDAF-specific functionality
    - Comprehensive evaluation and novelty detection
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config_manager = load_config(config_path)
        self.config = self.config_manager.config

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Setup device and memory optimization
        self._setup_device_and_memory()

        # Create necessary directories
        self.config_manager.create_directories()

        # Set random seed for reproducibility
        self._set_seed(self.config.get("seed", 42))

        # Initialize components
        self.lightning_module = None
        self.data_module = None
        self.trainer = None

        self.logger.info("MEDAF Lightning Trainer initialized successfully")

    def _setup_device_and_memory(self):
        """Setup device and apply memory optimizations"""
        device_config = self.config.get("hardware.device", "auto")

        if device_config == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = torch.device(device_config)

        self.logger.info(f"Using device: {self.device}")

        # Memory optimizations for CUDA
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Set memory management
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            # Set memory fraction
            memory_fraction = self.config.get("hardware.memory_fraction", 0.9)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

            self.logger.info(
                f"CUDA memory optimizations applied (fraction: {memory_fraction})"
            )

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        import numpy as np

        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Set Lightning seed
        pl.seed_everything(seed, workers=True)

        self.logger.info(f"Random seed set to: {seed}")

    def _create_lightning_module(self) -> MEDAFLightningModule:
        """Create the Lightning module"""
        model_args = self.config_manager.get_model_args()
        loss_config = self.config_manager.get_loss_config()
        optimizer_config = self.config_manager.get_optimizer_config()
        training_config = self.config_manager.get_training_args()
        class_names = self.config.get("class_names", [])
        num_classes = self.config.get("model.num_classes", 8)

        lightning_module = MEDAFLightningModule(
            model_args=model_args,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            class_names=class_names,
            num_classes=num_classes,
        )

        self.logger.info("Lightning module created successfully")
        return lightning_module

    def _create_data_module(self) -> MEDAFDataModule:
        """Create the Lightning data module"""
        data_module = MEDAFDataModule(
            train_csv=self.config.get("data.train_csv"),
            test_csv=self.config.get("data.test_csv"),
            image_root=self.config.get("data.image_root"),
            img_size=self.config.get("data.img_size", 224),
            batch_size=self.config.get("training.batch_size", 32),
            num_workers=self.config.get("training.num_workers", 1),
            val_ratio=self.config.get("training.val_ratio", 0.1),
            use_stratified_split=self.config.get("training.use_stratified_split", True),
            max_samples=self.config.get("data.max_samples"),
            pin_memory=self.config.get("hardware.pin_memory", True),
            train_list=self.config.get("data.train_list"),
            test_list=self.config.get("data.test_list"),
            use_full_dataset=True,  # Enable full dataset mode for 14 labels
        )

        self.logger.info("Data module created successfully")
        return data_module

    def _create_callbacks(self) -> List[pl.Callback]:
        """Create Lightning callbacks"""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.get("training.early_stopping.monitor", "val/loss"),
            mode=self.config.get("training.early_stopping.mode", "min"),
            patience=self.config.get("training.early_stopping.patience", 15),
            min_delta=float(self.config.get("training.early_stopping.min_delta", 1e-5)),
            verbose=True,
        )
        callbacks.append(early_stopping)

        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        # Model checkpointing
        checkpoint_dir = Path(self.config.get("checkpoints.dir"))
        checkpoint_callback = MEDAFModelCheckpointCallback(
            dirpath=str(checkpoint_dir),
            filename="medaf-lightning-{epoch:02d}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            save_weights_only=False,
        )
        callbacks.append(checkpoint_callback)

        # MEDAF-specific callbacks
        class_names = self.config.get("class_names", [])

        # Metrics callback
        metrics_callback = MEDAFMetricsCallback(
            class_names=class_names,
            log_every_n_epochs=5,
            save_plots=True,
            plots_dir=str(checkpoint_dir / "plots"),
        )
        callbacks.append(metrics_callback)

        # Threshold optimization callback
        if self.config.get("training.use_optimal_thresholds", False):
            threshold_callback = MEDAFThresholdOptimizationCallback(
                class_names=class_names,
                optimize_every_n_epochs=10,
                save_optimal_thresholds=True,
            )
            callbacks.append(threshold_callback)

        # ROC curve callback
        if self.config.get("training.create_roc_plots", True):
            roc_callback = MEDAFROCCurveCallback(
                class_names=class_names,
                create_every_n_epochs=10,
                save_plots=True,
                plots_dir=str(checkpoint_dir / "plots"),
            )
            callbacks.append(roc_callback)

        # Novelty detection callback (if enabled and not using full dataset)
        if (
            self.config.get("novelty_detection.enabled", False)
            and not self.data_module.use_full_dataset
        ):
            # Create unknown data loader for novelty detection
            unknown_loader = self.data_module.create_unknown_dataloader(
                novelty_type="all",
                max_samples=self.config.get("novelty_detection.max_unknown_samples"),
            )

            novelty_callback = MEDAFNoveltyDetectionCallback(
                unknown_dataloader=unknown_loader,
                class_names=class_names,
                gamma=self.config.get("novelty_detection.gamma", 1.0),
                temperature=self.config.get("novelty_detection.temperature", 1.0),
                fpr_target=self.config.get("novelty_detection.fpr_target", 0.05),
                evaluate_every_n_epochs=20,
            )
            callbacks.append(novelty_callback)
        elif self.data_module.use_full_dataset:
            self.logger.info(
                "Skipping novelty detection callback - using full dataset with all 14 labels"
            )

        self.logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks

    def _create_loggers(self) -> List[pl.loggers.Logger]:
        """Create Lightning loggers"""
        loggers = []

        # TensorBoard logger
        log_dir = self.config.get("logging.metrics_dir", "logs")
        tb_logger = TensorBoardLogger(
            save_dir=log_dir,
            name="medaf_lightning",
            version=f"run_{int(time.time())}",
        )
        loggers.append(tb_logger)

        # CSV logger for metrics
        csv_logger = CSVLogger(
            save_dir=log_dir,
            name="medaf_lightning",
            version=f"run_{int(time.time())}",
        )
        loggers.append(csv_logger)

        return loggers

    def _create_trainer(self) -> pl.Trainer:
        """Create the Lightning trainer"""
        # Training configuration
        max_epochs = self.config.get("training.num_epochs", 50)
        precision = 16 if self.config.get("hardware.mixed_precision", False) else 32

        # Device configuration
        if self.device.type == "cuda":
            devices = "auto"  # Use all available GPUs
            accelerator = "gpu"
        elif self.device.type == "mps":
            devices = 1
            accelerator = "mps"
        else:
            devices = 1
            accelerator = "cpu"

        # Create callbacks and loggers
        callbacks = self._create_callbacks()
        loggers = self._create_loggers()

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=devices,
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=True,  # For reproducibility
            benchmark=False,  # Disable for reproducibility
            log_every_n_steps=50,
            val_check_interval=1.0,  # Validate every epoch
        )

        self.logger.info("Lightning trainer created successfully")
        return trainer

    def train(self) -> Dict[str, Any]:
        """Main training method"""
        self.logger.info("Starting MEDAF Lightning training...")

        # Create components
        self.lightning_module = self._create_lightning_module()
        self.data_module = self._create_data_module()
        self.trainer = self._create_trainer()

        # Setup data module
        self.data_module.setup("fit")

        # Set class weights in lightning module if available
        class_weights = self.data_module.get_class_weights()
        if class_weights is not None:
            self.lightning_module.set_class_weights(class_weights)
            self.logger.info("Class weights set successfully")
        else:
            self.logger.info("No class weights calculated")

        # Start training
        training_start_time = time.time()

        self.trainer.fit(
            model=self.lightning_module,
            datamodule=self.data_module,
        )

        training_time = time.time() - training_start_time

        # Get best model path
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        best_model_score = self.trainer.checkpoint_callback.best_model_score

        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best model saved at: {best_model_path}")
        self.logger.info(f"Best model score: {best_model_score}")

        # Verify checkpoint exists
        if best_model_path and Path(best_model_path).exists():
            self.logger.info("✅ Checkpoint verification successful")
        else:
            self.logger.warning("⚠️ Best model checkpoint not found or empty path")

        return {
            "trainer": self.trainer,
            "lightning_module": self.lightning_module,
            "data_module": self.data_module,
            "best_model_path": best_model_path,
            "best_model_score": best_model_score,
            "training_time": training_time,
        }

    def test(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Test the trained model"""
        if checkpoint_path is None:
            # Use the best model from training
            if self.trainer is None:
                raise ValueError(
                    "No trainer available. Please train first or provide checkpoint_path."
                )
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Testing model from checkpoint: {checkpoint_path}")

        # Create components if not already created
        if self.lightning_module is None:
            self.lightning_module = self._create_lightning_module()
        if self.data_module is None:
            self.data_module = self._create_data_module()
        if self.trainer is None:
            self.trainer = self._create_trainer()

        # Setup data module for testing
        self.data_module.setup("test")

        # Load model from checkpoint
        try:
            self.lightning_module = MEDAFLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device,
                strict=False,  # Allow for missing keys
            )
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            # Fallback: create new module and load state dict manually
            self.lightning_module = self._create_lightning_module()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "state_dict" in checkpoint:
                self.lightning_module.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
            else:
                raise ValueError(
                    f"Invalid checkpoint format: {list(checkpoint.keys())}"
                )

        # Test the model
        test_results = self.trainer.test(
            model=self.lightning_module,
            datamodule=self.data_module,
            verbose=True,
        )

        self.logger.info("Testing completed successfully")

        return {
            "test_results": test_results,
            "checkpoint_path": checkpoint_path,
        }

    def evaluate_novelty_detection(
        self, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate novelty detection performance"""
        if checkpoint_path is None:
            if self.trainer is None:
                raise ValueError(
                    "No trainer available. Please train first or provide checkpoint_path."
                )
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(
            f"Evaluating novelty detection from checkpoint: {checkpoint_path}"
        )

        # Create components if not already created
        if self.lightning_module is None:
            self.lightning_module = self._create_lightning_module()
        if self.data_module is None:
            self.data_module = self._create_data_module()

        # Load model from checkpoint
        try:
            self.lightning_module = MEDAFLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self.device,
                strict=False,  # Allow for missing keys
            )
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            # Fallback: create new module and load state dict manually
            self.lightning_module = self._create_lightning_module()
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "state_dict" in checkpoint:
                self.lightning_module.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
            else:
                raise ValueError(
                    f"Invalid checkpoint format: {list(checkpoint.keys())}"
                )

        # Setup data module
        self.data_module.setup("fit")
        self.data_module.setup("test")

        # Create unknown data loader
        unknown_loader = self.data_module.create_unknown_dataloader(
            novelty_type="all",
            max_samples=self.config.get("novelty_detection.max_unknown_samples"),
        )

        # Import novelty detection functions
        from .multilabel_novelty_detection import (
            MultiLabelNoveltyDetector,
            evaluate_novelty_detection,
        )

        # Create and calibrate novelty detector
        detector = MultiLabelNoveltyDetector(
            gamma=self.config.get("novelty_detection.gamma", 1.0),
            temperature=self.config.get("novelty_detection.temperature", 1.0),
        )

        # Get validation loader for calibration
        val_loader = self.data_module.val_dataloader()

        # Calibrate threshold
        detector.calibrate_threshold(
            self.lightning_module.model,
            val_loader,
            self.device,
            fpr_target=self.config.get("novelty_detection.fpr_target", 0.05),
        )

        # Get test loader
        test_loader = self.data_module.test_dataloader()

        # Evaluate novelty detection
        novelty_results = evaluate_novelty_detection(
            self.lightning_module.model,
            test_loader,
            unknown_loader,
            self.device,
            detector,
        )

        # Print results
        self._print_novelty_detection_results(novelty_results)

        return novelty_results

    def _print_novelty_detection_results(self, results: Dict[str, Any]):
        """Print formatted novelty detection results"""
        print("\n" + "=" * 70)
        print("🔍 NOVELTY DETECTION EVALUATION RESULTS")
        print("=" * 70)

        print("\n📊 Overall Novelty Detection Performance:")
        print(f"   AUROC:              {results['auroc']:.4f}")
        print(
            f"   Detection Accuracy: {results['detection_accuracy']:.4f} ({results['detection_accuracy']*100:.2f}%)"
        )
        print(f"   Precision:          {results['precision']:.4f}")
        print(f"   Recall:             {results['recall']:.4f}")
        print(f"   F1-Score:           {results['f1_score']:.4f}")

        print(f"\n🎯 Detection Threshold: {results['threshold']:.4f}")
        print(f"   Known samples:   {results['num_known']}")
        print(f"   Unknown samples: {results['num_unknown']}")

        # Performance assessment
        auroc = results["auroc"]
        if auroc >= 0.9:
            assessment = "Excellent 🎉"
        elif auroc >= 0.8:
            assessment = "Good 👍"
        elif auroc >= 0.7:
            assessment = "Fair 🆗"
        else:
            assessment = "Needs Improvement ⚠️"

        print(f"\n💡 Performance Assessment: {assessment}")
        print("=" * 70)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary information"""
        if self.trainer is None:
            return {"error": "No training has been performed yet"}

        summary = {
            "best_model_path": self.trainer.checkpoint_callback.best_model_path,
            "best_model_score": self.trainer.checkpoint_callback.best_model_score,
            "total_epochs": self.trainer.current_epoch + 1,
            "device": str(self.device),
            "checkpoint_exists": (
                Path(self.trainer.checkpoint_callback.best_model_path).exists()
                if self.trainer.checkpoint_callback.best_model_path
                else False
            ),
        }

        if self.data_module is not None:
            dataset_info = self.data_module.get_dataset_info()
            summary.update(dataset_info)

        return summary

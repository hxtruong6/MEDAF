"""
MEDAF Multi-Label Classification Trainer
"""

import os
import gc
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# Local imports
from core.config_manager import load_config
from core.losses import LossFactory, calculate_class_weights_advanced
from core.metrics_logger import MetricsLogger, EpochMetrics, TrainingProgressTracker
from core.multilabel_net import MultiLabelMEDAF
from core.training_utils import (
    optimize_thresholds_per_class,
    evaluate_with_optimal_thresholds,
    print_enhanced_results,
    print_standard_results,
    print_auc_results,
    calculate_multilabel_accuracy,
    train_multilabel_enhanced,
    train_multilabel_enhanced_with_metrics,
    calculate_multilabel_attention_diversity,
    calculate_multilabel_auc,
    calculate_roc_curves,
)
from core.multilabel_novelty_detection import (
    MultiLabelNoveltyDetector,
    evaluate_novelty_detection,
)
from test_multilabel_medaf import ChestXrayKnownDataset, ChestXrayUnknownDataset


class MEDAFTrainer:
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

        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(
            log_dir=self.config.get("logging.metrics_dir"),
            experiment_name=f"medaf_{int(time.time())}",
            create_plots=self.config.get("logging.create_plots", True),
        )

        # Initialize progress tracker for early stopping
        self.progress_tracker = TrainingProgressTracker(
            patience=self.config.get("training.early_stopping.patience", 10),
            min_delta=float(self.config.get("training.early_stopping.min_delta", 1e-5)),
            monitor="val_loss",
            mode="min",
        )

        # Set random seed for reproducibility
        self._set_seed(self.config.get("seed", 42))

        self.logger.info("MEDAF Trainer initialized successfully")

    def _setup_device_and_memory(self):
        """Setup device and apply memory optimizations"""
        device_config = self.config.get("hardware.device", "auto")

        if device_config == "auto":
            # check mps
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

        # Clear memory
        gc.collect()

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.logger.info(f"Random seed set to: {seed}")

    def _extract_labels_for_stratification(self, dataset) -> np.ndarray:
        """
        Extract labels from dataset for stratification purposes.
        For multi-label data, we create a stratification strategy based on label combinations.
        """
        self.logger.info("Extracting labels for stratification...")
        labels_list = []

        # Extract labels from all samples
        for i in range(len(dataset)):
            _, labels = dataset[i]
            # Convert to binary array and then to string for stratification
            label_str = "".join(map(str, labels.int().tolist()))
            labels_list.append(label_str)

        # Log label distribution for debugging
        unique_labels, counts = np.unique(labels_list, return_counts=True)
        self.logger.info(f"Found {len(unique_labels)} unique label combinations")
        self.logger.info(
            f"Most common combinations: {dict(zip(unique_labels[:5], counts[:5]))}"
        )

        return np.array(labels_list)

    def _create_stratified_split(
        self, dataset, val_ratio: float, random_state: int = 42
    ) -> Tuple[data.Subset, data.Subset]:
        """
        Create stratified train-validation split for multi-label data.

        Args:
            dataset: The full dataset
            val_ratio: Ratio of validation data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_subset, val_subset)
        """
        self.logger.info("Creating stratified train-validation split...")

        try:
            # Extract labels for stratification
            labels_for_stratify = self._extract_labels_for_stratification(dataset)

            # Get indices
            indices = np.arange(len(dataset))

            # Use train_test_split with stratification
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_state,
                stratify=labels_for_stratify,
            )

            self.logger.info("✅ Stratified split successful")

        except ValueError as e:
            # If stratification fails (e.g., some classes have too few samples),
            # fall back to random split with a warning
            self.logger.warning(f"Stratified split failed: {e}")
            self.logger.warning("Falling back to random split...")

            val_size = max(1, int(len(dataset) * val_ratio))
            train_size = len(dataset) - val_size

            train_indices, val_indices = train_test_split(
                indices, test_size=val_ratio, random_state=random_state
            )

        # Create subsets
        train_subset = data.Subset(dataset, train_indices)
        val_subset = data.Subset(dataset, val_indices)

        self.logger.info(
            f"Split created: Train={len(train_subset)}, Val={len(val_subset)}"
        )

        return train_subset, val_subset

    def _create_data_loaders(
        self,
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """Create train, validation, and test data loaders"""
        self.logger.info("Creating data loaders...")

        # Load training dataset
        train_dataset = ChestXrayKnownDataset(
            csv_path=self.config.get("data.train_csv"),
            image_root=self.config.get("data.image_root"),
            img_size=self.config.get("data.img_size", 224),
            max_samples=self.config.get("data.max_samples"),
        )

        # Create validation split
        val_ratio = self.config.get("training.val_ratio", 0.1)
        use_stratified_split = self.config.get("training.use_stratified_split", False)

        if use_stratified_split:
            # Use stratified split to prevent dataset imbalance
            train_subset, val_subset = self._create_stratified_split(
                train_dataset,
                val_ratio=val_ratio,
                random_state=self.config.get("seed", 42),
            )
        else:
            # Use random split (original behavior)
            val_size = max(1, int(len(train_dataset) * val_ratio))
            train_size = len(train_dataset) - val_size

            train_subset, val_subset = data.random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config.get("seed", 42)),
            )

        # Create test dataset
        test_dataset = ChestXrayKnownDataset(
            csv_path=self.config.get("data.test_csv"),
            image_root=self.config.get("data.image_root"),
            img_size=self.config.get("data.img_size", 224),
            max_samples=self.config.get("data.max_samples"),
        )

        # Data loader parameters
        batch_size = self.config.get("training.batch_size", 32)
        num_workers = self.config.get("training.num_workers", 1)
        pin_memory = (
            self.config.get("hardware.pin_memory", True) and self.device.type == "cuda"
        )

        # Create data loaders
        train_loader = data.DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = data.DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.logger.info(
            f"Data loaders created: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader

    def _create_model(self) -> MultiLabelMEDAF:
        """Create and initialize the MEDAF model"""
        model_args = self.config_manager.get_model_args()
        model = MultiLabelMEDAF(model_args)
        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model created with {total_params:,} trainable parameters")

        return model

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer based on configuration"""
        optimizer_config = self.config_manager.get_optimizer_config()
        training_args = self.config_manager.get_training_args()

        optimizer_type = optimizer_config["type"].lower()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_args["learning_rate"],
                weight_decay=training_args["weight_decay"],
                betas=optimizer_config["betas"],
                eps=optimizer_config["eps"],
                amsgrad=optimizer_config["amsgrad"],
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_args["learning_rate"],
                betas=optimizer_config["betas"],
                eps=optimizer_config["eps"],
                amsgrad=optimizer_config["amsgrad"],
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=training_args["learning_rate"],
                weight_decay=training_args["weight_decay"],
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.logger.info(f"Optimizer created: {optimizer_type.upper()}")
        return optimizer

    def _create_loss_function(self, train_loader: data.DataLoader) -> nn.Module:
        """Create loss function with optional class weighting"""
        loss_config = self.config_manager.get_loss_config()
        class_names = self.config.get("class_names", [])
        num_classes = self.config.get("model.num_classes", 8)

        # Calculate class weights if enabled
        pos_weight = None
        if loss_config["class_weighting_enabled"]:
            self.logger.info("Calculating class weights...")
            pos_weight = calculate_class_weights_advanced(
                train_loader,
                num_classes,
                self.device,
                method=loss_config["class_weighting_method"],
            )
            self.logger.info(
                f"Class weights calculated using {loss_config['class_weighting_method']} method"
            )

        # show class weights by normalizing to 1 with corresponding class name
        if pos_weight is not None:
            for i, weight in enumerate(pos_weight):
                print(
                    f"Class {i} [{class_names[i]}]: {((weight/pos_weight.sum()) * 100):.2f}%"
                )

        # Create loss function
        loss_fn = LossFactory.create_loss(
            loss_type=loss_config["type"],
            num_classes=num_classes,
            device=self.device,
            pos_weight=pos_weight,
            focal_alpha=loss_config["focal_alpha"],
            focal_gamma=loss_config["focal_gamma"],
        )

        self.logger.info(f"Loss function created: {loss_config['type']}")
        return loss_fn

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch using enhanced multi-label training with AUC calculation"""
        model.train()

        # Get model arguments for enhanced training
        model_args = self.config_manager.get_model_args()

        # Create criterion dictionary expected by train_multilabel_enhanced
        criterion_dict = {"bce": criterion}

        # Use the enhanced training function with detailed metrics
        metrics = train_multilabel_enhanced_with_metrics(
            train_loader=train_loader,
            model=model,
            criterion=criterion_dict,
            optimizer=optimizer,
            args=model_args,
            device=self.device,
        )

        # Calculate AUC scores for training data (optional, can be expensive)
        calculate_train_auc = self.config.get("training.calculate_train_auc", False)
        train_auc_metrics = {}

        if calculate_train_auc:
            # Note: This is computationally expensive for large datasets
            # Consider calculating only every few epochs or on a subset
            self.logger.info("Calculating training AUC scores...")
            all_predictions = []
            all_targets = []

            model.eval()
            with torch.no_grad():
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    output_dict = model(inputs, targets)
                    logits = output_dict["logits"]

                    all_predictions.append(logits[3])
                    all_targets.append(targets)

            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            class_names = self.config.get("class_names", [])
            auc_results = calculate_multilabel_auc(
                all_predictions, all_targets, class_names
            )

            train_auc_metrics = {
                "macro_auc": auc_results["macro_auc"],
                "micro_auc": auc_results["micro_auc"],
                "weighted_auc": auc_results["weighted_auc"],
            }

            model.train()  # Set back to training mode

        # Return metrics in the expected format
        return {
            "loss": metrics["total_loss"],
            "accuracy": metrics["accuracy"],
            "expert_loss": metrics["expert_loss"],
            "gate_loss": metrics["gate_loss"],
            "diversity_loss": metrics["diversity_loss"],
            **train_auc_metrics,  # Include AUC metrics if calculated
        }

    def _validate_epoch(
        self, model: nn.Module, val_loader: data.DataLoader, criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate for one epoch with AUC calculation"""
        model.eval()

        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        # Store all predictions and targets for AUC calculation
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass
                output_dict = model(inputs, targets)
                logits = output_dict["logits"]

                # Calculate loss (using gate predictions)
                loss = criterion(logits[3], targets.float())

                # Accumulate metrics
                total_loss += loss.item()
                total_samples += inputs.size(0)

                # Calculate accuracy
                predictions = torch.sigmoid(logits[3]) > 0.5
                correct_predictions += (predictions == targets).all(dim=1).sum().item()

                # Store for AUC calculation
                all_predictions.append(logits[3])
                all_targets.append(targets)

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate AUC scores
        class_names = self.config.get("class_names", [])
        auc_results = calculate_multilabel_auc(
            all_predictions, all_targets, class_names
        )

        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "macro_auc": auc_results["macro_auc"],
            "micro_auc": auc_results["micro_auc"],
            "weighted_auc": auc_results["weighted_auc"],
        }

    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info("Starting MEDAF training...")

        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()

        # Create model
        model = self._create_model()

        # Create optimizer and loss function
        optimizer = self._create_optimizer(model)
        criterion = self._create_loss_function(train_loader)

        # Training parameters
        num_epochs = self.config.get("training.num_epochs", 50)
        save_every = self.config.get("checkpoints.save_every", 5)

        # Training loop
        best_val_loss = float("inf")
        training_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Train epoch
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate epoch
            val_metrics = self._validate_epoch(model, val_loader, criterion)

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Create epoch metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_acc=train_metrics["accuracy"],
                val_acc=val_metrics["accuracy"],
                learning_rate=optimizer.param_groups[0]["lr"],
                epoch_time=epoch_time,
                expert_loss=train_metrics["expert_loss"],
                gate_loss=train_metrics["gate_loss"],
                diversity_loss=train_metrics["diversity_loss"],
                # AUC metrics
                train_macro_auc=train_metrics.get("macro_auc"),
                val_macro_auc=val_metrics.get("macro_auc"),
                train_micro_auc=train_metrics.get("micro_auc"),
                val_micro_auc=val_metrics.get("micro_auc"),
                train_weighted_auc=train_metrics.get("weighted_auc"),
                val_weighted_auc=val_metrics.get("weighted_auc"),
            )

            # Log metrics
            self.metrics_logger.log_epoch(epoch_metrics)

            # Check for early stopping
            if self.progress_tracker.update(epoch_metrics):
                self.logger.info("Early stopping triggered")
                break

            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_metrics["loss"] < best_val_loss:
                self._save_checkpoint(model, optimizer, epoch, train_metrics["loss"])
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self.logger.info(f"New best validation loss: {best_val_loss:.6f}")

        # Training completed
        total_training_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_training_time:.2f} seconds")

        # Create final plots and summary
        self.metrics_logger.create_training_plots()
        self.metrics_logger.create_loss_breakdown_plot()
        self.metrics_logger.create_auc_plots()  # Create AUC plots
        summary = self.metrics_logger.create_summary_report()

        return {
            "model": model,
            "summary": summary,
            "best_val_loss": best_val_loss,
            "total_training_time": total_training_time,
        }

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
    ):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get("checkpoints.dir"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"medaf_epoch_{epoch}_{int(time.time())}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": dict(self.config._config),
            "model_args": self.config_manager.get_model_args(),
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _calibrate_novelty_detector(
        self, model: nn.Module, val_loader: data.DataLoader
    ) -> MultiLabelNoveltyDetector:
        """
        Calibrate novelty detector on validation data containing only known samples.

        Args:
            model: Trained MEDAF model
            val_loader: Validation data loader (known samples only)

        Returns:
            Calibrated MultiLabelNoveltyDetector instance
        """
        self.logger.info("Calibrating novelty detector...")

        # Get novelty detection parameters from config
        gamma = self.config.get("novelty_detection.gamma", 1.0)
        temperature = self.config.get("novelty_detection.temperature", 1.0)
        fpr_target = self.config.get("novelty_detection.fpr_target", 0.05)

        # Create detector
        detector = MultiLabelNoveltyDetector(gamma=gamma, temperature=temperature)

        # Calibrate threshold
        detector.calibrate_threshold(
            model, val_loader, self.device, fpr_target=fpr_target
        )

        self.logger.info(
            f"✅ Novelty detector calibrated (threshold={detector.threshold:.4f})"
        )

        return detector

    def _create_unknown_data_loader(
        self, novelty_type: str = "all", max_samples: Optional[int] = None
    ) -> data.DataLoader:
        """
        Create data loader for unknown/novel samples.

        Args:
            novelty_type: Type of novelty samples to load ("all", "independent", "mixed", "known_only")
            max_samples: Maximum number of samples (None = all)

        Returns:
            DataLoader for unknown samples
        """
        self.logger.info(
            f"Creating unknown data loader (novelty_type={novelty_type})..."
        )

        # Create unknown dataset
        unknown_dataset = ChestXrayUnknownDataset(
            csv_path=self.config.get("data.test_csv"),
            image_root=self.config.get("data.image_root"),
            img_size=self.config.get("data.img_size", 224),
            max_samples=max_samples,
            novelty_type=novelty_type,
        )

        # Create data loader
        batch_size = self.config.get("training.batch_size", 32)
        num_workers = self.config.get("training.num_workers", 1)
        pin_memory = (
            self.config.get("hardware.pin_memory", True) and self.device.type == "cuda"
        )

        unknown_loader = data.DataLoader(
            unknown_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.logger.info(f"Unknown data loader created: {len(unknown_dataset)} samples")

        return unknown_loader

    def evaluate_novelty_detection(
        self, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model's ability to detect unknown/novel samples.

        This method:
        1. Loads the trained model
        2. Calibrates the novelty detector on known validation data
        3. Evaluates detection performance on unknown samples
        4. Reports AUROC, precision, recall, and F1 for novelty detection

        Args:
            checkpoint_path: Path to model checkpoint (uses config default if None)

        Returns:
            Dictionary containing novelty detection evaluation results
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.get("checkpoints.evaluation_checkpoint")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint for novelty detection: {checkpoint_path}")

        # Load checkpoint and create model
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        model = self._create_model()

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise KeyError(
                f"Checkpoint does not contain model state. Available keys: {list(checkpoint.keys())}"
            )

        model.eval()

        # Create data loaders
        train_loader, val_loader, test_loader = self._create_data_loaders()

        # Calibrate novelty detector on validation data (known samples only)
        detector = self._calibrate_novelty_detector(model, val_loader)

        # Create unknown data loader
        max_unknown_samples = self.config.get("novelty_detection.max_unknown_samples")
        unknown_loader = self._create_unknown_data_loader(
            novelty_type="all", max_samples=max_unknown_samples
        )

        # Evaluate novelty detection performance
        self.logger.info("Evaluating novelty detection performance...")
        novelty_results = evaluate_novelty_detection(
            model, test_loader, unknown_loader, self.device, detector
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

    def evaluate_comprehensive(
        self, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation combining known label classification AND novelty detection.

        This is the complete evaluation pipeline that:
        1. Evaluates classification performance on known labels
        2. Evaluates novelty detection on unknown samples
        3. Provides detailed analysis of both capabilities

        Args:
            checkpoint_path: Path to model checkpoint (uses config default if None)

        Returns:
            Dictionary containing both classification and novelty detection results
        """
        if checkpoint_path is None:
            checkpoint_path = self.config.get("checkpoints.evaluation_checkpoint")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info("=" * 70)
        self.logger.info(
            "COMPREHENSIVE EVALUATION: Known Classification + Novelty Detection"
        )
        self.logger.info("=" * 70)

        # 1. Evaluate known label classification
        self.logger.info("\n[1/2] Evaluating known label classification...")
        classification_results = self.evaluate(checkpoint_path)

        # 2. Evaluate novelty detection
        enable_novelty = self.config.get("novelty_detection.enabled", True)
        novelty_results = None

        if enable_novelty:
            self.logger.info("\n[2/2] Evaluating novelty detection...")
            try:
                novelty_results = self.evaluate_novelty_detection(checkpoint_path)
            except Exception as e:
                self.logger.warning(f"Novelty detection evaluation failed: {e}")
                self.logger.warning("Continuing with classification results only...")
        else:
            self.logger.info("\n[2/2] Novelty detection disabled in configuration")

        # Compile comprehensive results
        comprehensive_results = {
            "classification": classification_results,
            "novelty_detection": novelty_results,
            "checkpoint": checkpoint_path,
        }

        # Print summary
        self._print_comprehensive_summary(comprehensive_results)

        return comprehensive_results

    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 70)

        # Classification summary
        if results["classification"]:
            cls_results = results["classification"]
            print("\n✅ Known Label Classification:")
            if "overall" in cls_results:
                print(
                    f"   Subset Accuracy: {cls_results['overall']['subset_accuracy']:.4f}"
                )
                print(
                    f"   Hamming Accuracy: {cls_results['overall']['hamming_accuracy']:.4f}"
                )
                print(f"   F1-Score: {cls_results['overall']['f1_score']:.4f}")
            if "auc_metrics" in cls_results and cls_results["auc_metrics"]:
                print(f"   Macro AUC: {cls_results['auc_metrics']['macro_auc']:.4f}")

        # Novelty detection summary
        if results["novelty_detection"]:
            nov_results = results["novelty_detection"]
            print("\n🔍 Novelty Detection:")
            print(f"   AUROC: {nov_results['auroc']:.4f}")
            print(f"   Detection Accuracy: {nov_results['detection_accuracy']:.4f}")
            print(f"   F1-Score: {nov_results['f1_score']:.4f}")

        print("\n" + "=" * 70)
        print("✅ Comprehensive evaluation completed successfully!")
        print("=" * 70)

    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate trained model on known labels (standard classification)"""
        if checkpoint_path is None:
            checkpoint_path = self.config.get("checkpoints.evaluation_checkpoint")

        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint for evaluation: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Create model and load state
        model = self._create_model()

        # Handle different checkpoint formats (backward compatibility)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise KeyError(
                f"Checkpoint does not contain model state. Available keys: {list(checkpoint.keys())}"
            )

        model.eval()

        # Create test loader
        _, _, test_loader = self._create_data_loaders()

        # Evaluate with optimal thresholds if enabled
        if self.config.get("training.use_optimal_thresholds", False):
            self.logger.info("Evaluating with optimal thresholds...")

            # Use part of test set for threshold optimization
            test_size = len(test_loader.dataset)
            val_size = min(1000, test_size // 5)
            eval_size = test_size - val_size

            val_dataset, eval_dataset = data.random_split(
                test_loader.dataset,
                [val_size, eval_size],
                generator=torch.Generator().manual_seed(self.config.get("seed", 42)),
            )

            val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
            eval_loader = data.DataLoader(eval_dataset, batch_size=32, shuffle=False)

            # Optimize thresholds
            class_names = self.config.get("class_names", [])
            optimal_thresholds, _ = optimize_thresholds_per_class(
                model, val_loader, self.device, len(class_names), class_names
            )

            # Evaluate with optimal thresholds
            results = evaluate_with_optimal_thresholds(
                model, eval_loader, self.device, optimal_thresholds, class_names
            )

            # Calculate AUC scores and ROC curves for final evaluation if enabled
            calculate_eval_auc = self.config.get("training.calculate_eval_auc", True)
            create_roc_plots = self.config.get("training.create_roc_plots", True)

            if calculate_eval_auc:
                self.logger.info("Calculating final AUC scores and ROC curves...")
                all_predictions = []
                all_targets = []

                with torch.no_grad():
                    for inputs, targets in eval_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        output_dict = model(inputs, targets)
                        logits = output_dict["logits"]

                        all_predictions.append(logits[3])
                        all_targets.append(targets)

                all_predictions = torch.cat(all_predictions, dim=0)
                all_targets = torch.cat(all_targets, dim=0)

                # Calculate AUC scores
                auc_results = calculate_multilabel_auc(
                    all_predictions, all_targets, class_names
                )
                results["auc_metrics"] = auc_results

                # Calculate ROC curves and create plots if enabled
                if create_roc_plots:
                    roc_data = calculate_roc_curves(
                        all_predictions, all_targets, class_names
                    )
                    self.metrics_logger.create_roc_curve_plot(roc_data)
            else:
                self.logger.info("AUC calculation disabled in configuration")

            print_enhanced_results(results)

        else:
            # Standard evaluation with AUC calculation
            self.logger.info("Standard evaluation (threshold=0.5) with AUC calculation")

            # Collect all predictions and targets
            all_predictions = []
            all_targets = []
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # Forward pass
                    output_dict = model(inputs, targets)
                    logits = output_dict["logits"]

                    # Calculate loss
                    criterion = self._create_loss_function(test_loader)
                    loss = criterion(logits[3], targets.float())
                    total_loss += loss.item()
                    num_batches += 1

                    # Store predictions and targets
                    all_predictions.append(logits[3])
                    all_targets.append(targets)

            # Concatenate all results
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Calculate standard metrics with threshold=0.5
            class_names = self.config.get("class_names", [])
            subset_acc, hamming_acc, precision, recall, f1 = (
                calculate_multilabel_accuracy(
                    all_predictions, all_targets, threshold=0.5
                )
            )

            # Calculate AUC scores and ROC curves if enabled
            calculate_eval_auc = self.config.get("training.calculate_eval_auc", True)
            create_roc_plots = self.config.get("training.create_roc_plots", True)

            if calculate_eval_auc:
                # Calculate AUC scores
                auc_results = calculate_multilabel_auc(
                    all_predictions, all_targets, class_names
                )

                # Calculate ROC curves and create plots if enabled
                if create_roc_plots:
                    roc_data = calculate_roc_curves(
                        all_predictions, all_targets, class_names
                    )
                    self.metrics_logger.create_roc_curve_plot(roc_data)
            else:
                self.logger.info("AUC calculation disabled in configuration")
                auc_results = None

            # Compile results
            results = {
                "overall": {
                    "subset_accuracy": subset_acc,
                    "hamming_accuracy": hamming_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "average_loss": total_loss / num_batches,
                },
                "auc_metrics": auc_results,
                "threshold_method": "fixed_0.5",
                "class_names": class_names,
            }

            # Print results
            print_standard_results(results)

        return results


def main():
    """Main function to run training or evaluation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="MEDAF Multi-Label Classification Trainer"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "eval_novelty", "eval_comprehensive"],
        default="train",
        help="Mode: train, eval (known labels), eval_novelty (unknown detection), or eval_comprehensive (both)",
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path for evaluation")

    args = parser.parse_args()

    # Create trainer
    trainer = MEDAFTrainer(args.config)

    if args.mode == "train":
        # Training mode
        results = trainer.train()
        print(f"✅ Training completed successfully")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Total training time: {results['total_training_time']:.2f} seconds")

    elif args.mode == "eval":
        # Evaluation mode - known labels only
        results = trainer.evaluate(args.checkpoint)
        print(f"✅ Evaluation completed successfully")

    elif args.mode == "eval_novelty":
        # Novelty detection evaluation mode
        results = trainer.evaluate_novelty_detection(args.checkpoint)
        print(f"✅ Novelty detection evaluation completed successfully")

    elif args.mode == "eval_comprehensive":
        # Comprehensive evaluation mode - both known and novelty
        results = trainer.evaluate_comprehensive(args.checkpoint)
        print(f"✅ Comprehensive evaluation completed successfully")


if __name__ == "__main__":
    main()

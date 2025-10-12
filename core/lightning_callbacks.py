"""
Custom PyTorch Lightning Callbacks for MEDAF Multi-Label Classification
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from .training_utils import (
    calculate_multilabel_auc,
    calculate_roc_curves,
    optimize_thresholds_per_class,
    evaluate_with_optimal_thresholds,
)
from .multilabel_novelty_detection import (
    MultiLabelNoveltyDetector,
    evaluate_novelty_detection,
)


class MEDAFMetricsCallback(Callback):
    """
    Custom callback for MEDAF-specific metrics and logging
    """

    def __init__(
        self,
        class_names: list,
        log_every_n_epochs: int = 5,
        save_plots: bool = True,
        plots_dir: str = "plots",
    ):
        super().__init__()
        self.class_names = class_names
        self.log_every_n_epochs = log_every_n_epochs
        self.save_plots = save_plots
        self.plots_dir = Path(plots_dir)

        if self.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Store metrics history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Log training metrics at the end of each epoch"""
        # Store metrics for plotting
        self.epochs.append(trainer.current_epoch)

        # Get metrics from trainer
        if trainer.logged_metrics:
            train_loss = trainer.logged_metrics.get("train/loss_epoch", 0)
            train_acc = trainer.logged_metrics.get("train/accuracy_epoch", 0)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Log validation metrics and create plots periodically"""
        # Store validation metrics
        if trainer.logged_metrics:
            val_loss = trainer.logged_metrics.get("val/loss", 0)
            val_acc = trainer.logged_metrics.get("val/accuracy", 0)

            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

        # Create plots every N epochs
        if (trainer.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._create_training_plots(trainer.current_epoch)

    def _create_training_plots(self, epoch: int):
        """Create training progress plots"""
        if not self.save_plots or len(self.epochs) < 2:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.epochs, self.train_losses, label="Train Loss", color="blue")
        ax1.plot(self.epochs, self.val_losses, label="Val Loss", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.epochs, self.train_accs, label="Train Accuracy", color="blue")
        ax2.plot(self.epochs, self.val_accs, label="Val Accuracy", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / f"training_progress_epoch_{epoch}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create final training summary plots"""
        if self.save_plots and len(self.epochs) > 0:
            self._create_training_plots("final")


class MEDAFThresholdOptimizationCallback(Callback):
    """
    Callback for optimizing classification thresholds on validation data
    """

    def __init__(
        self,
        class_names: list,
        optimize_every_n_epochs: int = 10,
        save_optimal_thresholds: bool = True,
    ):
        super().__init__()
        self.class_names = class_names
        self.optimize_every_n_epochs = optimize_every_n_epochs
        self.save_optimal_thresholds = save_optimal_thresholds
        self.optimal_thresholds = None

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Optimize thresholds periodically during training"""
        if (trainer.current_epoch + 1) % self.optimize_every_n_epochs == 0:
            self._optimize_thresholds(trainer, pl_module)

    def _optimize_thresholds(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Optimize classification thresholds on validation data"""
        print(f"\n🔧 Optimizing thresholds at epoch {trainer.current_epoch + 1}...")

        try:
            # Get validation dataloader
            val_dataloader = trainer.val_dataloaders[0]

            # Verify class names match model configuration
            expected_num_classes = pl_module.num_classes
            actual_num_classes = len(self.class_names)

            if actual_num_classes != expected_num_classes:
                print(
                    f"⚠️ Warning: Class names count ({actual_num_classes}) doesn't match model classes ({expected_num_classes})"
                )
                print(
                    f"   Using model's num_classes ({expected_num_classes}) for threshold optimization"
                )
                num_classes_to_use = expected_num_classes
                class_names_to_use = (
                    self.class_names[:expected_num_classes]
                    if actual_num_classes > expected_num_classes
                    else self.class_names
                    + [
                        f"Class_{i}"
                        for i in range(actual_num_classes, expected_num_classes)
                    ]
                )
            else:
                num_classes_to_use = expected_num_classes
                class_names_to_use = self.class_names

            # Optimize thresholds
            optimal_thresholds, threshold_metrics = optimize_thresholds_per_class(
                pl_module.model,
                val_dataloader,
                pl_module.device,
                num_classes_to_use,
                class_names_to_use,
            )

            self.optimal_thresholds = optimal_thresholds

            # Log threshold optimization results
            print("📊 Threshold Optimization Results:")
            for i, (class_name, threshold) in enumerate(
                zip(class_names_to_use, optimal_thresholds)
            ):
                print(f"  {class_name}: {threshold:.4f}")

            # Store thresholds in pl_module for later use
            pl_module.optimal_thresholds = optimal_thresholds

        except Exception as e:
            print(f"❌ Error during threshold optimization: {e}")
            print("   Skipping threshold optimization for this epoch")
            # Set default thresholds
            self.optimal_thresholds = [0.5] * pl_module.num_classes
            pl_module.optimal_thresholds = self.optimal_thresholds

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Final threshold optimization"""
        if self.optimal_thresholds is None:
            self._optimize_thresholds(trainer, pl_module)


class MEDAFNoveltyDetectionCallback(Callback):
    """
    Callback for evaluating novelty detection performance
    """

    def __init__(
        self,
        unknown_dataloader,
        class_names: list,
        gamma: float = 1.0,
        temperature: float = 1.0,
        fpr_target: float = 0.05,
        evaluate_every_n_epochs: int = 20,
    ):
        super().__init__()
        self.unknown_dataloader = unknown_dataloader
        self.class_names = class_names
        self.gamma = gamma
        self.temperature = temperature
        self.fpr_target = fpr_target
        self.evaluate_every_n_epochs = evaluate_every_n_epochs

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Evaluate novelty detection periodically"""
        if (trainer.current_epoch + 1) % self.evaluate_every_n_epochs == 0:
            self._evaluate_novelty_detection(trainer, pl_module)

    def _evaluate_novelty_detection(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Evaluate novelty detection performance"""
        print(
            f"\n🔍 Evaluating novelty detection at epoch {trainer.current_epoch + 1}..."
        )

        # Get validation dataloader (known samples)
        val_dataloader = trainer.val_dataloaders[0]

        # Create and calibrate novelty detector
        detector = MultiLabelNoveltyDetector(
            gamma=self.gamma, temperature=self.temperature
        )
        detector.calibrate_threshold(
            pl_module.model,
            val_dataloader,
            pl_module.device,
            fpr_target=self.fpr_target,
        )

        # Evaluate novelty detection
        novelty_results = evaluate_novelty_detection(
            pl_module.model,
            val_dataloader,
            self.unknown_dataloader,
            pl_module.device,
            detector,
        )

        # Log results
        print(f"📊 Novelty Detection Results:")
        print(f"  AUROC: {novelty_results['auroc']:.4f}")
        print(f"  Detection Accuracy: {novelty_results['detection_accuracy']:.4f}")
        print(f"  F1-Score: {novelty_results['f1_score']:.4f}")

        # Log to trainer
        trainer.logger.log_metrics(
            {
                "novelty/auroc": novelty_results["auroc"],
                "novelty/detection_accuracy": novelty_results["detection_accuracy"],
                "novelty/precision": novelty_results["precision"],
                "novelty/recall": novelty_results["recall"],
                "novelty/f1_score": novelty_results["f1_score"],
            },
            step=trainer.current_epoch,
        )


class MEDAFROCCurveCallback(Callback):
    """
    Callback for creating ROC curves and AUC analysis
    """

    def __init__(
        self,
        class_names: list,
        create_every_n_epochs: int = 10,
        save_plots: bool = True,
        plots_dir: str = "plots",
    ):
        super().__init__()
        self.class_names = class_names
        self.create_every_n_epochs = create_every_n_epochs
        self.save_plots = save_plots
        self.plots_dir = Path(plots_dir)

        if self.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        """Create ROC curves periodically"""
        if (trainer.current_epoch + 1) % self.create_every_n_epochs == 0:
            self._create_roc_curves(trainer, pl_module)

    def _create_roc_curves(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Create ROC curves for validation data"""
        print(f"\n📈 Creating ROC curves at epoch {trainer.current_epoch + 1}...")

        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders[0]

        # Collect predictions and targets
        all_predictions = []
        all_targets = []

        pl_module.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, targets = batch
                inputs = inputs.to(pl_module.device)
                targets = targets.to(pl_module.device)

                output_dict = pl_module.model(inputs, targets)
                logits = output_dict["logits"]

                # Use gate predictions
                predictions = torch.sigmoid(logits[3])

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        pl_module.train()

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate ROC curves
        roc_data = calculate_roc_curves(all_predictions, all_targets, self.class_names)

        # Create ROC curve plot
        if self.save_plots:
            self._plot_roc_curves(roc_data, trainer.current_epoch)

    def _plot_roc_curves(self, roc_data: Dict[str, Any], epoch: int):
        """Create and save ROC curve plots"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Per-class ROC curves
        for i, class_name in enumerate(self.class_names):
            if f"roc_curve_{i}" in roc_data:
                fpr, tpr, auc = roc_data[f"roc_curve_{i}"]
                ax1.plot(fpr, tpr, label=f"{class_name} (AUC={auc:.3f})", linewidth=2)

        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("Per-Class ROC Curves")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Macro-average ROC curve
        if "macro_roc_curve" in roc_data:
            fpr, tpr, auc = roc_data["macro_roc_curve"]
            ax2.plot(
                fpr,
                tpr,
                label=f"Macro-average (AUC={auc:.3f})",
                linewidth=2,
                color="red",
            )

        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Macro-Average ROC Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / f"roc_curves_epoch_{epoch}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"📊 ROC curves saved to {self.plots_dir / f'roc_curves_epoch_{epoch}.png'}"
        )


class MEDAFModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """
    Enhanced model checkpoint callback with MEDAF-specific features
    """

    def __init__(
        self,
        dirpath: str,
        filename: str = "medaf-{epoch:02d}-{val_loss:.2f}",
        monitor: str = "val/loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        save_weights_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            save_weights_only=save_weights_only,
            **kwargs,
        )

        # Create directory if it doesn't exist
        Path(dirpath).mkdir(parents=True, exist_ok=True)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        """Add MEDAF-specific information to checkpoint"""
        # Add model configuration
        checkpoint["model_args"] = pl_module.model_args
        checkpoint["loss_config"] = pl_module.loss_config
        checkpoint["optimizer_config"] = pl_module.optimizer_config
        checkpoint["training_config"] = pl_module.training_config
        checkpoint["class_names"] = pl_module.class_names

        # Add optimal thresholds if available
        if hasattr(pl_module, "optimal_thresholds"):
            checkpoint["optimal_thresholds"] = pl_module.optimal_thresholds

        # Add class weights if available
        if hasattr(pl_module, "pos_weight") and pl_module.pos_weight is not None:
            checkpoint["pos_weight"] = pl_module.pos_weight.cpu()

        super().on_save_checkpoint(trainer, pl_module, checkpoint)

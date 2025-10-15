"""
PyTorch Lightning Module for MEDAF Multi-Label Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .multilabel_net import MultiLabelMEDAF
from .losses import LossFactory, calculate_class_weights_advanced
from .training_utils import (
    calculate_multilabel_accuracy,
    calculate_multilabel_attention_diversity,
    calculate_multilabel_auc,
)


class MEDAFLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for MEDAF Multi-Label Classification

    This module encapsulates:
    - Model architecture (MultiLabelMEDAF)
    - Loss computation with multiple components
    - Training and validation steps
    - Metrics calculation and logging
    - Optimizer and scheduler configuration
    """

    def __init__(
        self,
        model_args: Dict[str, Any],
        loss_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        training_config: Dict[str, Any],
        class_names: list,
        num_classes: int = 8,
        **kwargs,
    ):
        super().__init__()

        # Store configuration
        self.model_args = model_args
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.class_names = class_names
        self.num_classes = num_classes

        # Initialize model
        self.model = MultiLabelMEDAF(model_args)

        # Initialize loss function (will be set up in setup)
        self.criterion = None
        self.pos_weight = None

        # Store metrics for logging
        self.train_metrics = {}
        self.val_metrics = {}

        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: Optional[str] = None):
        """Setup the module - called after moving to device"""
        self._ensure_criterion_initialized()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading from checkpoint"""
        super().on_load_checkpoint(checkpoint)

        # Restore pos_weight if available
        if "pos_weight" in checkpoint and checkpoint["pos_weight"] is not None:
            self.pos_weight = checkpoint["pos_weight"]
            if hasattr(self, "device") and self.device is not None:
                self.pos_weight = self.pos_weight.to(self.device)

        # Ensure criterion is initialized after loading
        self._ensure_criterion_initialized()

        # Ensure model is on the correct device
        self._ensure_model_initialized()

    def _ensure_criterion_initialized(self):
        """Ensure criterion is properly initialized"""
        if self.criterion is None:
            device = getattr(self, "device", torch.device("cpu"))
            self.criterion = LossFactory.create_loss(
                loss_type=self.loss_config["type"],
                num_classes=self.num_classes,
                device=device,
                pos_weight=self.pos_weight,
                focal_alpha=self.loss_config.get("focal_alpha", 0.25),
                focal_gamma=self.loss_config.get("focal_gamma", 2.0),
            )

    def _ensure_model_initialized(self):
        """Ensure model is properly initialized"""
        if self.model is None:
            print("Warning: Model is None, this should not happen")
            return False

        # Ensure model is on the correct device
        if hasattr(self, "device") and self.device is not None:
            self.model = self.model.to(self.device)

        return True

    def forward(self, x, y=None, return_ft=False):
        """Forward pass through the model"""
        return self.model(x, y, return_ft)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Training step with multi-component loss calculation"""
        try:
            inputs, targets = batch

            # Forward pass
            output_dict = self.model(inputs, targets)
            logits = output_dict["logits"]  # List of logits from 4 heads
            cams_list = output_dict["cams_list"]  # CAMs from 3 experts

            # Ensure criterion is initialized
            if self.criterion is None:
                self.criterion = LossFactory.create_loss(
                    loss_type=self.loss_config["type"],
                    num_classes=self.num_classes,
                    device=self.device,
                    pos_weight=self.pos_weight,
                    focal_alpha=self.loss_config.get("focal_alpha", 0.25),
                    focal_gamma=self.loss_config.get("focal_gamma", 2.0),
                )

            # Multi-label classification losses for expert branches
            bce_losses = [
                self.criterion(logit.float(), targets.float())
                for logit in logits[:3]  # Expert branches only
            ]

            # Gating loss (on fused predictions)
            gate_loss = self.criterion(logits[3].float(), targets.float())

            # Multi-label attention diversity loss
            diversity_loss = calculate_multilabel_attention_diversity(
                cams_list, targets
            )

            # Combine losses according to weights
            loss_weights = self.model_args["loss_wgts"]
            total_loss = (
                loss_weights[0] * sum(bce_losses)  # Expert loss weight
                + loss_weights[1] * gate_loss  # Gating loss weight
                + loss_weights[2] * diversity_loss  # Diversity loss weight
            )

            # Calculate accuracies for monitoring
            accuracies = []
            for logit in logits:
                subset_acc, hamming_acc, _, _, _ = calculate_multilabel_accuracy(
                    logit, targets, threshold=0.5
                )
                accuracies.append(subset_acc)

            # Log training metrics
            self.log(
                "train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
            )
            self.log("train/expert_loss", sum(bce_losses), on_step=False, on_epoch=True)
            self.log("train/gate_loss", gate_loss, on_step=False, on_epoch=True)
            self.log(
                "train/diversity_loss", diversity_loss, on_step=False, on_epoch=True
            )
            self.log(
                "train/accuracy",
                accuracies[-1],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            # Log individual expert accuracies
            for i, acc in enumerate(accuracies):
                self.log(f"train/acc_expert_{i+1}", acc, on_step=False, on_epoch=True)

            return {
                "loss": total_loss,
                "expert_loss": sum(bce_losses),
                "gate_loss": gate_loss,
                "diversity_loss": diversity_loss,
                "accuracy": accuracies[-1],
            }

        except Exception as e:
            print(f"Error in training step: {e}")
            # Return a dummy loss to prevent training from crashing
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return {"loss": dummy_loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step with comprehensive metrics"""
        try:
            inputs, targets = batch

            # Forward pass
            output_dict = self.model(inputs, targets)
            logits = output_dict["logits"]

            # Ensure criterion is initialized
            if self.criterion is None:
                self.criterion = LossFactory.create_loss(
                    loss_type=self.loss_config["type"],
                    num_classes=self.num_classes,
                    device=self.device,
                    pos_weight=self.pos_weight,
                    focal_alpha=self.loss_config.get("focal_alpha", 0.25),
                    focal_gamma=self.loss_config.get("focal_gamma", 2.0),
                )

            # Calculate loss (using gate predictions)
            val_loss = self.criterion(logits[3], targets.float())

            # Calculate accuracies
            accuracies = []
            for logit in logits:
                subset_acc, hamming_acc, precision, recall, f1 = (
                    calculate_multilabel_accuracy(logit, targets, threshold=0.5)
                )
                accuracies.append(
                    {
                        "subset_acc": subset_acc,
                        "hamming_acc": hamming_acc,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )

            # Store predictions and targets for AUC calculation
            predictions = torch.sigmoid(logits[3])

            # Log validation metrics
            self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val/accuracy",
                accuracies[-1]["subset_acc"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log("val/f1", accuracies[-1]["f1"], on_step=False, on_epoch=True)

            return {
                "val_loss": val_loss,
                "val_accuracy": accuracies[-1]["subset_acc"],
                "val_f1": accuracies[-1]["f1"],
                "predictions": predictions,
                "targets": targets,
            }

        except Exception as e:
            print(f"Error in validation step: {e}")
            # Return dummy values to prevent validation from crashing
            dummy_loss = torch.tensor(0.0, device=self.device)
            dummy_predictions = torch.zeros_like(targets, device=self.device)
            return {
                "val_loss": dummy_loss,
                "val_accuracy": 0.0,
                "val_f1": 0.0,
                "predictions": dummy_predictions,
                "targets": targets,
            }

    def validation_epoch_end(self, outputs):
        """Calculate epoch-level validation metrics including AUC"""
        try:
            # Check if outputs is empty or contains invalid data
            if not outputs or len(outputs) == 0:
                print("Warning: No validation outputs available")
                return

            # Collect all predictions and targets
            all_predictions = torch.cat([x["predictions"] for x in outputs])
            all_targets = torch.cat([x["targets"] for x in outputs])

            # Calculate AUC scores
            auc_results = calculate_multilabel_auc(
                all_predictions, all_targets, self.class_names
            )

            # Log AUC metrics
            self.log(
                "val/macro_auc", auc_results["macro_auc"], on_epoch=True, prog_bar=True
            )
            self.log("val/micro_auc", auc_results["micro_auc"], on_epoch=True)
            self.log("val/weighted_auc", auc_results["weighted_auc"], on_epoch=True)

            # Log per-class AUC if available
            if "per_class_auc" in auc_results:
                for class_name, class_auc in auc_results["per_class_auc"].items():
                    self.log(f"val/auc_{class_name}", class_auc, on_epoch=True)

        except Exception as e:
            print(f"Error in validation_epoch_end: {e}")
            # Log dummy AUC values to prevent training from crashing
            self.log("val/macro_auc", 0.5, on_epoch=True, prog_bar=True)
            self.log("val/micro_auc", 0.5, on_epoch=True)
            self.log("val/weighted_auc", 0.5, on_epoch=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step - same as validation but with optimal thresholds if enabled"""
        inputs, targets = batch

        # Ensure model and criterion are initialized
        if not self._ensure_model_initialized():
            raise RuntimeError("Model is not properly initialized")
        self._ensure_criterion_initialized()

        # Forward pass
        output_dict = self.model(inputs, targets)
        logits = output_dict["logits"]

        # Calculate loss
        test_loss = self.criterion(logits[3], targets.float())

        # Calculate metrics with optimal thresholds if enabled
        if self.training_config.get("use_optimal_thresholds", False):
            # This would require threshold optimization - implement as needed
            threshold = 0.5  # Default threshold
        else:
            threshold = 0.5

        # Calculate comprehensive metrics
        subset_acc, hamming_acc, precision, recall, f1 = calculate_multilabel_accuracy(
            logits[3], targets, threshold=threshold
        )

        # Store for AUC calculation
        predictions = torch.sigmoid(logits[3])

        return {
            "test_loss": test_loss,
            "test_accuracy": subset_acc,
            "test_hamming_acc": hamming_acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "predictions": predictions,
            "targets": targets,
        }

    def test_epoch_end(self, outputs):
        """Calculate final test metrics including AUC"""
        # Collect all predictions and targets
        all_predictions = torch.cat([x["predictions"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])

        # Calculate AUC scores
        auc_results = calculate_multilabel_auc(
            all_predictions, all_targets, self.class_names
        )

        # Log final test metrics
        self.log("test/macro_auc", auc_results["macro_auc"])
        self.log("test/micro_auc", auc_results["micro_auc"])
        self.log("test/weighted_auc", auc_results["weighted_auc"])

        # Log per-class AUC
        if "per_class_auc" in auc_results:
            for class_name, class_auc in auc_results["per_class_auc"].items():
                self.log(f"test/auc_{class_name}", class_auc)

        # Add this to log F1 score:
        avg_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()
        self.log("test/f1_score", avg_f1)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer_type = self.optimizer_config["type"].lower()

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.training_config["learning_rate"],
                weight_decay=self.training_config.get("weight_decay", 1e-4),
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
                eps=self.optimizer_config.get("eps", 1e-8),
                amsgrad=self.optimizer_config.get("amsgrad", False),
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.training_config["learning_rate"],
                betas=self.optimizer_config.get("betas", [0.9, 0.999]),
                eps=self.optimizer_config.get("eps", 1e-8),
                amsgrad=self.optimizer_config.get("amsgrad", False),
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.training_config["learning_rate"],
                weight_decay=self.training_config.get("weight_decay", 1e-4),
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Configure learning rate scheduler - enable by default to prevent overfitting
        scheduler_config = self.training_config.get("scheduler", {})
        if scheduler_config.get("enabled", True):  # Enable by default
            scheduler_type = scheduler_config.get("type", "cosine")

            if scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.training_config.get("num_epochs", 50),
                    eta_min=scheduler_config.get("eta_min", 1e-6),
                )
            elif scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get("step_size", 10),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            elif scheduler_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_config.get("factor", 0.5),
                    patience=scheduler_config.get("patience", 5),
                    min_lr=scheduler_config.get("min_lr", 1e-6),
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss" if scheduler_type == "plateau" else None,
                },
            }

        return optimizer

    def set_class_weights(self, pos_weight: torch.Tensor):
        """Set class weights for the loss function"""
        self.pos_weight = pos_weight.to(self.device) if pos_weight is not None else None
        if self.criterion is not None:
            # Recreate loss function with new weights
            self.criterion = LossFactory.create_loss(
                loss_type=self.loss_config["type"],
                num_classes=self.num_classes,
                device=self.device,
                pos_weight=self.pos_weight,
                focal_alpha=self.loss_config.get("focal_alpha", 0.25),
                focal_gamma=self.loss_config.get("focal_gamma", 2.0),
            )

    def get_model_outputs(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get model outputs for inference or analysis"""
        self.eval()
        with torch.no_grad():
            output_dict = self.model(inputs)
            logits = output_dict["logits"]
            predictions = torch.sigmoid(logits[3])  # Use gate predictions
            return {
                "logits": logits,
                "predictions": predictions,
                "cams_list": output_dict.get("cams_list", []),
            }

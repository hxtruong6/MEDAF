"""
Advanced Metrics Logging and Visualization for MEDAF Training
Senior-level implementation with CSV logging and visualization utilities
"""

import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@dataclass
class EpochMetrics:
    """Data class for storing epoch metrics"""

    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_acc: Optional[float] = None
    learning_rate: float = 0.0
    epoch_time: float = 0.0

    # Detailed losses
    expert_loss: Optional[float] = None
    gate_loss: Optional[float] = None
    diversity_loss: Optional[float] = None

    # Per-class metrics (stored as JSON strings for CSV compatibility)
    train_per_class_acc: Optional[str] = None
    val_per_class_acc: Optional[str] = None

    # AUC metrics
    train_macro_auc: Optional[float] = None
    val_macro_auc: Optional[float] = None
    train_micro_auc: Optional[float] = None
    val_micro_auc: Optional[float] = None
    train_weighted_auc: Optional[float] = None
    val_weighted_auc: Optional[float] = None

    # Additional metrics
    memory_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None


class MetricsLogger:
    """
    Advanced metrics logger with CSV export and visualization capabilities
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "medaf_experiment",
        create_plots: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.create_plots = create_plots

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.log_dir / "plots"
        if self.create_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV file
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.metrics_history: List[EpochMetrics] = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize CSV with headers
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Write headers based on EpochMetrics fields
                headers = list(EpochMetrics.__annotations__.keys())
                writer.writerow(headers)
            self.logger.info(f"Initialized metrics CSV: {self.csv_path}")

    def log_epoch(self, metrics: EpochMetrics):
        """Log metrics for a single epoch"""
        self.metrics_history.append(metrics)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # Convert dataclass to list of values
            values = [
                getattr(metrics, field) for field in EpochMetrics.__annotations__.keys()
            ]
            writer.writerow(values)

        val_loss_str = (
            f"{metrics.val_loss:.4f}" if metrics.val_loss is not None else "N/A"
        )
        train_acc_str = (
            f"{metrics.train_acc:.3f}" if metrics.train_acc is not None else "N/A"
        )

        self.logger.info(
            f"Epoch {metrics.epoch:3d} | "
            f"Train Loss: {metrics.train_loss:.4f} | "
            f"Val Loss: {val_loss_str} | "
            f"Train Acc: {train_acc_str} | "
            f"Time: {metrics.epoch_time:.1f}s"
        )

    def create_training_plots(self, save_format: str = "png"):
        """Create comprehensive training visualization plots"""
        if not self.metrics_history:
            self.logger.warning("No metrics to plot")
            return

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([asdict(m) for m in self.metrics_history])

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Training Metrics - {self.experiment_name}", fontsize=16, fontweight="bold"
        )

        # 1. Loss curves
        ax1 = axes[0, 0]
        ax1.plot(
            df["epoch"],
            df["train_loss"],
            label="Train Loss",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        if "val_loss" in df.columns and df["val_loss"].notna().any():
            ax1.plot(
                df["epoch"],
                df["val_loss"],
                label="Val Loss",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy curves
        ax2 = axes[0, 1]
        if "train_acc" in df.columns and df["train_acc"].notna().any():
            ax2.plot(
                df["epoch"],
                df["train_acc"],
                label="Train Acc",
                linewidth=2,
                marker="o",
                markersize=3,
            )
        if "val_acc" in df.columns and df["val_acc"].notna().any():
            ax2.plot(
                df["epoch"],
                df["val_acc"],
                label="Val Acc",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Learning rate schedule
        ax3 = axes[1, 0]
        if "learning_rate" in df.columns and df["learning_rate"].notna().any():
            ax3.plot(
                df["epoch"],
                df["learning_rate"],
                linewidth=2,
                marker="o",
                markersize=3,
                color="red",
            )
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Learning Rate")
            ax3.set_title("Learning Rate Schedule")
            ax3.set_yscale("log")
            ax3.grid(True, alpha=0.3)

        # 4. Epoch time
        ax4 = axes[1, 1]
        if "epoch_time" in df.columns and df["epoch_time"].notna().any():
            ax4.plot(
                df["epoch"],
                df["epoch_time"],
                linewidth=2,
                marker="o",
                markersize=3,
                color="green",
            )
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Time (seconds)")
            ax4.set_title("Training Time per Epoch")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = (
            self.plots_dir / f"{self.experiment_name}_training_curves.{save_format}"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Training plots saved: {plot_path}")

    def create_loss_breakdown_plot(self, save_format: str = "png"):
        """Create detailed loss breakdown plot"""
        if not self.metrics_history:
            return

        df = pd.DataFrame([asdict(m) for m in self.metrics_history])

        # Check if detailed loss components are available
        loss_components = ["expert_loss", "gate_loss", "diversity_loss"]
        available_components = [
            comp
            for comp in loss_components
            if comp in df.columns and df[comp].notna().any()
        ]

        if not available_components:
            self.logger.info("No detailed loss components to plot")
            return

        plt.figure(figsize=(12, 6))

        for component in available_components:
            plt.plot(
                df["epoch"],
                df[component],
                label=component.replace("_", " ").title(),
                linewidth=2,
                marker="o",
                markersize=3,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Components Breakdown")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plot_path = (
            self.plots_dir / f"{self.experiment_name}_loss_breakdown.{save_format}"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Loss breakdown plot saved: {plot_path}")

    def create_auc_plots(self, save_format: str = "png"):
        """Create AUC plots for training and validation"""
        if not self.metrics_history:
            return

        df = pd.DataFrame([asdict(m) for m in self.metrics_history])

        # Check if AUC metrics are available
        auc_metrics = [
            "train_macro_auc",
            "val_macro_auc",
            "train_micro_auc",
            "val_micro_auc",
        ]
        available_auc = [
            metric
            for metric in auc_metrics
            if metric in df.columns and df[metric].notna().any()
        ]

        if not available_auc:
            self.logger.info("No AUC metrics to plot")
            return

        # Create AUC plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("AUC Metrics Over Training", fontsize=16)

        # 1. Macro AUC
        ax1 = axes[0, 0]
        if "train_macro_auc" in df.columns and df["train_macro_auc"].notna().any():
            ax1.plot(
                df["epoch"],
                df["train_macro_auc"],
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3,
            )
        if "val_macro_auc" in df.columns and df["val_macro_auc"].notna().any():
            ax1.plot(
                df["epoch"],
                df["val_macro_auc"],
                label="Validation",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Macro AUC")
        ax1.set_title("Macro AUC")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. Micro AUC
        ax2 = axes[0, 1]
        if "train_micro_auc" in df.columns and df["train_micro_auc"].notna().any():
            ax2.plot(
                df["epoch"],
                df["train_micro_auc"],
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3,
            )
        if "val_micro_auc" in df.columns and df["val_micro_auc"].notna().any():
            ax2.plot(
                df["epoch"],
                df["val_micro_auc"],
                label="Validation",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Micro AUC")
        ax2.set_title("Micro AUC")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. Weighted AUC
        ax3 = axes[1, 0]
        if (
            "train_weighted_auc" in df.columns
            and df["train_weighted_auc"].notna().any()
        ):
            ax3.plot(
                df["epoch"],
                df["train_weighted_auc"],
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3,
            )
        if "val_weighted_auc" in df.columns and df["val_weighted_auc"].notna().any():
            ax3.plot(
                df["epoch"],
                df["val_weighted_auc"],
                label="Validation",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Weighted AUC")
        ax3.set_title("Weighted AUC")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # 4. AUC Comparison
        ax4 = axes[1, 1]
        if "val_macro_auc" in df.columns and df["val_macro_auc"].notna().any():
            ax4.plot(
                df["epoch"],
                df["val_macro_auc"],
                label="Macro AUC",
                linewidth=2,
                marker="o",
                markersize=3,
            )
        if "val_micro_auc" in df.columns and df["val_micro_auc"].notna().any():
            ax4.plot(
                df["epoch"],
                df["val_micro_auc"],
                label="Micro AUC",
                linewidth=2,
                marker="s",
                markersize=3,
            )
        if "val_weighted_auc" in df.columns and df["val_weighted_auc"].notna().any():
            ax4.plot(
                df["epoch"],
                df["val_weighted_auc"],
                label="Weighted AUC",
                linewidth=2,
                marker="^",
                markersize=3,
            )
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("AUC Score")
        ax4.set_title("Validation AUC Comparison")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / f"{self.experiment_name}_auc_metrics.{save_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"AUC plots saved: {plot_path}")

    def create_roc_curve_plot(self, roc_data: Dict[str, Any], save_format: str = "png"):
        """Create ROC curve plot for all classes"""
        if not roc_data:
            return

        plt.figure(figsize=(12, 8))

        # Plot ROC curves for each class
        for class_name, data in roc_data.items():
            fpr = data["fpr"]
            tpr = data["tpr"]
            auc_score = np.trapz(tpr, fpr)  # Approximate AUC from curve
            plt.plot(
                fpr, tpr, label=f"{class_name} (AUC = {auc_score:.3f})", linewidth=2
            )

        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", alpha=0.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Multi-Label Classification")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.tight_layout()

        # Save plot
        plot_path = self.plots_dir / f"{self.experiment_name}_roc_curves.{save_format}"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"ROC curves plot saved: {plot_path}")

    def create_summary_report(self) -> Dict[str, Any]:
        """Create a comprehensive training summary report"""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame([asdict(m) for m in self.metrics_history])

        # Calculate summary statistics
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.metrics_history),
            "total_training_time": df["epoch_time"].sum() if "epoch_time" in df else 0,
            "average_epoch_time": df["epoch_time"].mean() if "epoch_time" in df else 0,
            "final_train_loss": (
                df["train_loss"].iloc[-1] if "train_loss" in df else None
            ),
            "best_train_loss": df["train_loss"].min() if "train_loss" in df else None,
            "final_val_loss": (
                df["val_loss"].iloc[-1]
                if "val_loss" in df and df["val_loss"].notna().any()
                else None
            ),
            "best_val_loss": (
                df["val_loss"].min()
                if "val_loss" in df and df["val_loss"].notna().any()
                else None
            ),
            "final_train_acc": (
                df["train_acc"].iloc[-1]
                if "train_acc" in df and df["train_acc"].notna().any()
                else None
            ),
            "best_train_acc": (
                df["train_acc"].max()
                if "train_acc" in df and df["train_acc"].notna().any()
                else None
            ),
            "final_val_acc": (
                df["val_acc"].iloc[-1]
                if "val_acc" in df and df["val_acc"].notna().any()
                else None
            ),
            "best_val_acc": (
                df["val_acc"].max()
                if "val_acc" in df and df["val_acc"].notna().any()
                else None
            ),
        }

        # Save summary to JSON
        summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Training summary saved: {summary_path}")
        return summary

    def load_from_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load metrics from CSV file"""
        path = Path(csv_path) if csv_path else self.csv_path

        if not path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {path}")

        df = pd.read_csv(path)
        self.logger.info(f"Loaded {len(df)} epochs from {path}")
        return df

    def export_for_tensorboard(self) -> str:
        """Export metrics in TensorBoard-compatible format"""
        # This would integrate with TensorBoard if needed
        # For now, just return the CSV path
        return str(self.csv_path)


class TrainingProgressTracker:
    """
    Real-time training progress tracker with early stopping and best model tracking
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.patience = patience
        # Ensure min_delta is always a float to prevent type errors
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

        self.logger = logging.getLogger(__name__)

    def update(self, metrics: EpochMetrics) -> bool:
        """
        Update tracker with new metrics

        Returns:
            bool: True if training should stop (early stopping triggered)
        """
        current_score = getattr(metrics, self.monitor, None)

        if current_score is None:
            return False

        # Ensure current_score is a float to prevent type errors
        try:
            current_score = float(current_score)
        except (ValueError, TypeError) as e:
            self.logger.warning(
                f"Could not convert current_score to float: {current_score}, error: {e}"
            )
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = metrics.epoch
            self.wait = 0
            self.logger.info(
                f"New best {self.monitor}: {current_score:.6f} at epoch {metrics.epoch}"
            )
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = metrics.epoch
                self.logger.info(
                    f"Early stopping triggered at epoch {metrics.epoch}. "
                    f"Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}"
                )
                return True

        return False


# Example usage and testing
if __name__ == "__main__":
    # Test metrics logger
    logger = MetricsLogger("./logs/test_logs", "test_experiment")

    # Simulate training epochs
    for epoch in range(10):
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=1.0 - epoch * 0.1 + np.random.normal(0, 0.05),
            val_loss=1.2 - epoch * 0.08 + np.random.normal(0, 0.05),
            train_acc=0.5 + epoch * 0.04 + np.random.normal(0, 0.02),
            val_acc=0.45 + epoch * 0.035 + np.random.normal(0, 0.02),
            learning_rate=1e-3 * (0.9**epoch),
            epoch_time=30 + np.random.normal(0, 5),
        )
        logger.log_epoch(metrics)

    # Create plots and summary
    logger.create_training_plots()
    summary = logger.create_summary_report()

    print("✅ Metrics logging test completed")
    print(f"Summary: {summary}")

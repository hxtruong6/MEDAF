"""
Comprehensive Evaluation Module for MEDAF Lightning Implementation
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import logging

from .lightning_module import MEDAFLightningModule
from .lightning_datamodule import MEDAFDataModule
from .training_utils import (
    calculate_multilabel_accuracy,
    calculate_multilabel_auc,
    calculate_roc_curves,
    optimize_thresholds_per_class,
    evaluate_with_optimal_thresholds,
)
from .multilabel_novelty_detection import (
    MultiLabelNoveltyDetector,
    evaluate_novelty_detection,
)


class MEDAFLightningEvaluator:
    """
    Comprehensive evaluator for MEDAF Lightning implementation

    Provides:
    - Standard multi-label classification evaluation
    - Novelty detection evaluation
    - Threshold optimization
    - ROC curve analysis
    - Performance comparison with original implementation
    """

    def __init__(
        self,
        lightning_module: MEDAFLightningModule,
        data_module: MEDAFDataModule,
        device: torch.device,
        class_names: List[str],
        config: Dict[str, Any],
    ):
        self.lightning_module = lightning_module
        self.data_module = data_module
        self.device = device
        self.class_names = class_names
        self.config = config
        self.logger = logging.getLogger(__name__)

    def evaluate_classification(
        self,
        use_optimal_thresholds: bool = True,
        create_roc_plots: bool = True,
        save_results: bool = True,
        results_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        """
        Evaluate standard multi-label classification performance

        Args:
            use_optimal_thresholds: Whether to optimize thresholds per class
            create_roc_plots: Whether to create ROC curve plots
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("🧪 Starting classification evaluation...")

        # Setup data module for testing
        self.data_module.setup("test")
        test_loader = self.data_module.test_dataloader()

        if use_optimal_thresholds:
            # Use optimal thresholds
            self.logger.info("🔧 Optimizing classification thresholds...")

            # Split test data for threshold optimization
            test_size = len(test_loader.dataset)
            val_size = min(1000, test_size // 5)
            eval_size = test_size - val_size

            val_dataset, eval_dataset = torch.utils.data.random_split(
                test_loader.dataset,
                [val_size, eval_size],
                generator=torch.Generator().manual_seed(self.config.get("seed", 42)),
            )

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=32, shuffle=False
            )
            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=32, shuffle=False
            )

            # Optimize thresholds
            optimal_thresholds, threshold_metrics = optimize_thresholds_per_class(
                self.lightning_module.model,
                val_loader,
                self.device,
                len(self.class_names),
                self.class_names,
            )

            # Evaluate with optimal thresholds
            results = evaluate_with_optimal_thresholds(
                self.lightning_module.model,
                eval_loader,
                self.device,
                optimal_thresholds,
                self.class_names,
            )

            results["threshold_method"] = "optimal"
            results["optimal_thresholds"] = optimal_thresholds.tolist()

        else:
            # Use fixed threshold (0.5)
            self.logger.info("📊 Evaluating with fixed threshold (0.5)...")

            all_predictions = []
            all_targets = []
            total_loss = 0.0
            num_batches = 0

            self.lightning_module.eval()
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # Forward pass
                    output_dict = self.lightning_module.model(inputs, targets)
                    logits = output_dict["logits"]

                    # Calculate loss
                    loss = self.lightning_module.criterion(logits[3], targets.float())
                    total_loss += loss.item()
                    num_batches += 1

                    # Store predictions and targets
                    all_predictions.append(logits[3])
                    all_targets.append(targets)

            # Concatenate all results
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            # Calculate metrics with fixed threshold
            subset_acc, hamming_acc, precision, recall, f1 = (
                calculate_multilabel_accuracy(
                    all_predictions, all_targets, threshold=0.5
                )
            )

            results = {
                "overall": {
                    "subset_accuracy": subset_acc,
                    "hamming_accuracy": hamming_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "average_loss": total_loss / num_batches,
                },
                "threshold_method": "fixed_0.5",
                "class_names": self.class_names,
            }

        # Calculate AUC scores
        if create_roc_plots:
            self.logger.info("📈 Calculating AUC scores and ROC curves...")

            # Get predictions for AUC calculation
            if use_optimal_thresholds:
                # Use evaluation loader predictions
                all_predictions = []
                all_targets = []

                with torch.no_grad():
                    for inputs, targets in eval_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)

                        output_dict = self.lightning_module.model(inputs, targets)
                        logits = output_dict["logits"]

                        all_predictions.append(logits[3])
                        all_targets.append(targets)

                all_predictions = torch.cat(all_predictions, dim=0)
                all_targets = torch.cat(all_targets, dim=0)

            # Calculate AUC scores
            auc_results = calculate_multilabel_auc(
                all_predictions, all_targets, self.class_names
            )
            results["auc_metrics"] = auc_results

            # Calculate ROC curves
            roc_data = calculate_roc_curves(
                all_predictions, all_targets, self.class_names
            )
            results["roc_data"] = roc_data

            # Create ROC plots
            if save_results:
                self._create_roc_plots(roc_data, results_dir)

        # Save results
        if save_results:
            self._save_evaluation_results(results, results_dir, "classification")

        self.logger.info("✅ Classification evaluation completed")
        return results

    def evaluate_novelty_detection(
        self,
        max_unknown_samples: Optional[int] = None,
        save_results: bool = True,
        results_dir: str = "evaluation_results",
    ) -> Dict[str, Any]:
        """
        Evaluate novelty detection performance

        Args:
            max_unknown_samples: Maximum number of unknown samples to use
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            Dictionary containing novelty detection results
        """
        self.logger.info("🔍 Starting novelty detection evaluation...")

        # Setup data modules
        self.data_module.setup("fit")
        self.data_module.setup("test")

        # Get validation loader for calibration (known samples)
        val_loader = self.data_module.val_dataloader()

        # Get test loader (known samples)
        test_loader = self.data_module.test_dataloader()

        # Create unknown data loader
        unknown_loader = self.data_module.create_unknown_dataloader(
            novelty_type="all",
            max_samples=max_unknown_samples,
        )

        # Get novelty detection parameters
        gamma = self.config.get("novelty_detection.gamma", 1.0)
        temperature = self.config.get("novelty_detection.temperature", 1.0)
        fpr_target = self.config.get("novelty_detection.fpr_target", 0.05)

        # Create and calibrate novelty detector
        detector = MultiLabelNoveltyDetector(gamma=gamma, temperature=temperature)
        detector.calibrate_threshold(
            self.lightning_module.model, val_loader, self.device, fpr_target=fpr_target
        )

        # Evaluate novelty detection
        novelty_results = evaluate_novelty_detection(
            self.lightning_module.model,
            test_loader,
            unknown_loader,
            self.device,
            detector,
        )

        # Save results
        if save_results:
            self._save_evaluation_results(
                novelty_results, results_dir, "novelty_detection"
            )

        self.logger.info("✅ Novelty detection evaluation completed")
        return novelty_results

    def evaluate_comprehensive(
        self, save_results: bool = True, results_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation including both classification and novelty detection

        Args:
            save_results: Whether to save results to file
            results_dir: Directory to save results

        Returns:
            Dictionary containing comprehensive evaluation results
        """
        self.logger.info("📊 Starting comprehensive evaluation...")

        # Create results directory
        if save_results:
            Path(results_dir).mkdir(parents=True, exist_ok=True)

        # 1. Classification evaluation
        self.logger.info("[1/2] Evaluating classification performance...")
        classification_results = self.evaluate_classification(
            use_optimal_thresholds=self.config.get(
                "training.use_optimal_thresholds", True
            ),
            create_roc_plots=self.config.get("training.create_roc_plots", True),
            save_results=save_results,
            results_dir=results_dir,
        )

        # 2. Novelty detection evaluation
        novelty_results = None
        if self.config.get("novelty_detection.enabled", True):
            self.logger.info("[2/2] Evaluating novelty detection performance...")
            try:
                novelty_results = self.evaluate_novelty_detection(
                    max_unknown_samples=self.config.get(
                        "novelty_detection.max_unknown_samples"
                    ),
                    save_results=save_results,
                    results_dir=results_dir,
                )
            except Exception as e:
                self.logger.warning(f"Novelty detection evaluation failed: {e}")
                self.logger.warning("Continuing with classification results only...")
        else:
            self.logger.info("[2/2] Novelty detection disabled in configuration")

        # Compile comprehensive results
        comprehensive_results = {
            "classification": classification_results,
            "novelty_detection": novelty_results,
            "evaluation_timestamp": torch.utils.data.get_worker_info(),
            "config": self.config,
        }

        # Save comprehensive results
        if save_results:
            self._save_evaluation_results(
                comprehensive_results, results_dir, "comprehensive"
            )

        # Print summary
        self._print_comprehensive_summary(comprehensive_results)

        self.logger.info("✅ Comprehensive evaluation completed")
        return comprehensive_results

    def _create_roc_plots(self, roc_data: Dict[str, Any], results_dir: str):
        """Create and save ROC curve plots"""
        import matplotlib.pyplot as plt

        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

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
        plt.savefig(plots_dir / "roc_curves_final.png", dpi=150, bbox_inches="tight")
        plt.close()

        self.logger.info(f"📊 ROC curves saved to {plots_dir / 'roc_curves_final.png'}")

    def _save_evaluation_results(
        self, results: Dict[str, Any], results_dir: str, evaluation_type: str
    ):
        """Save evaluation results to file"""
        import json
        import time

        results_path = (
            Path(results_dir) / f"{evaluation_type}_results_{int(time.time())}.json"
        )

        # Convert any tensors to lists for JSON serialization
        def convert_tensors(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            else:
                return obj

        serializable_results = convert_tensors(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"💾 Results saved to {results_path}")

    def _print_comprehensive_summary(self, results: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        print("\n" + "=" * 70)
        print("📋 COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 70)

        # Classification summary
        if results["classification"]:
            cls_results = results["classification"]
            print("\n✅ Classification Results:")
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
                print(f"   Micro AUC: {cls_results['auc_metrics']['micro_auc']:.4f}")

        # Novelty detection summary
        if results["novelty_detection"]:
            nov_results = results["novelty_detection"]
            print("\n🔍 Novelty Detection Results:")
            print(f"   AUROC: {nov_results['auroc']:.4f}")
            print(f"   Detection Accuracy: {nov_results['detection_accuracy']:.4f}")
            print(f"   F1-Score: {nov_results['f1_score']:.4f}")

        print("\n" + "=" * 70)
        print("✅ Comprehensive evaluation completed successfully!")
        print("=" * 70)

    def compare_with_original(
        self, original_results_path: str, tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Compare Lightning implementation results with original trainer results

        Args:
            original_results_path: Path to original trainer results
            tolerance: Tolerance for numerical comparison

        Returns:
            Dictionary containing comparison results
        """
        self.logger.info("🔄 Comparing with original implementation...")

        # Load original results
        import json

        with open(original_results_path, "r") as f:
            original_results = json.load(f)

        # Run Lightning evaluation
        lightning_results = self.evaluate_comprehensive(save_results=False)

        # Compare results
        comparison = {
            "classification_comparison": {},
            "novelty_detection_comparison": {},
            "overall_match": True,
        }

        # Compare classification results
        if (
            "classification" in original_results
            and "classification" in lightning_results
        ):
            orig_cls = original_results["classification"]
            light_cls = lightning_results["classification"]

            # Compare overall metrics
            if "overall" in orig_cls and "overall" in light_cls:
                for metric in ["subset_accuracy", "hamming_accuracy", "f1_score"]:
                    if metric in orig_cls["overall"] and metric in light_cls["overall"]:
                        orig_val = orig_cls["overall"][metric]
                        light_val = light_cls["overall"][metric]
                        diff = abs(orig_val - light_val)

                        comparison["classification_comparison"][metric] = {
                            "original": orig_val,
                            "lightning": light_val,
                            "difference": diff,
                            "within_tolerance": diff <= tolerance,
                        }

                        if diff > tolerance:
                            comparison["overall_match"] = False

        # Compare novelty detection results
        if (
            "novelty_detection" in original_results
            and "novelty_detection" in lightning_results
        ):
            orig_nov = original_results["novelty_detection"]
            light_nov = lightning_results["novelty_detection"]

            for metric in ["auroc", "detection_accuracy", "f1_score"]:
                if metric in orig_nov and metric in light_nov:
                    orig_val = orig_nov[metric]
                    light_val = light_nov[metric]
                    diff = abs(orig_val - light_val)

                    comparison["novelty_detection_comparison"][metric] = {
                        "original": orig_val,
                        "lightning": light_val,
                        "difference": diff,
                        "within_tolerance": diff <= tolerance,
                    }

                    if diff > tolerance:
                        comparison["overall_match"] = False

        # Print comparison results
        self._print_comparison_results(comparison, tolerance)

        return comparison

    def _print_comparison_results(self, comparison: Dict[str, Any], tolerance: float):
        """Print comparison results"""
        print("\n" + "=" * 70)
        print("🔄 LIGHTNING vs ORIGINAL COMPARISON")
        print("=" * 70)

        # Classification comparison
        if comparison["classification_comparison"]:
            print("\n📊 Classification Results Comparison:")
            for metric, data in comparison["classification_comparison"].items():
                status = "✅" if data["within_tolerance"] else "❌"
                print(f"   {status} {metric}:")
                print(f"     Original:  {data['original']:.4f}")
                print(f"     Lightning: {data['lightning']:.4f}")
                print(
                    f"     Difference: {data['difference']:.4f} (tolerance: {tolerance})"
                )

        # Novelty detection comparison
        if comparison["novelty_detection_comparison"]:
            print("\n🔍 Novelty Detection Results Comparison:")
            for metric, data in comparison["novelty_detection_comparison"].items():
                status = "✅" if data["within_tolerance"] else "❌"
                print(f"   {status} {metric}:")
                print(f"     Original:  {data['original']:.4f}")
                print(f"     Lightning: {data['lightning']:.4f}")
                print(
                    f"     Difference: {data['difference']:.4f} (tolerance: {tolerance})"
                )

        # Overall result
        overall_status = "✅ MATCH" if comparison["overall_match"] else "❌ MISMATCH"
        print(f"\n🎯 Overall Result: {overall_status}")
        print("=" * 70)

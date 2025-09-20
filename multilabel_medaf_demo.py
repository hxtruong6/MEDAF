#!/usr/bin/env python3
"""
Multi-Label MEDAF Demo Script
Demonstrates both Phase 1 and Phase 2 implementations with comparative analysis
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

# Import Phase 1 (basic multi-label)
from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_train import train_multilabel

# Import Phase 2 (per-class gating)
from core.multilabel_net_v2 import MultiLabelMEDAFv2
from core.multilabel_train_v2 import train_multilabel_v2, ComparativeTrainingFramework

# Import test utilities
from test_multilabel_medaf import SyntheticMultiLabelDataset


class MultiLabelMEDAFDemo:
    """
    Comprehensive demo for Multi-Label MEDAF
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def create_dataset(self):
        """Create multi-label dataset for demonstration"""
        print("Creating Multi-Label Dataset...")

        dataset = SyntheticMultiLabelDataset(
            num_samples=self.config["num_samples"],
            img_size=self.config["img_size"],
            num_classes=self.config["num_classes"],
            avg_labels_per_sample=self.config["avg_labels_per_sample"],
            random_state=42,
        )

        # Split into train/test
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Create data loaders
        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
        )

        self.test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        print(f"Dataset created: {train_size} train, {test_size} test samples")

    def demo_phase1(self):
        """Demonstrate Phase 1: Basic Multi-Label MEDAF"""
        print("\n" + "=" * 60)
        print("PHASE 1 DEMO: Basic Multi-Label MEDAF")
        print("=" * 60)

        # Configuration for Phase 1
        args = {
            "img_size": self.config["img_size"],
            "backbone": "resnet18",
            "num_classes": self.config["num_classes"],
            "gate_temp": 100,
            "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
            "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
            "loss_wgts": [0.7, 1.0, 0.01],
        }

        # Create Phase 1 model
        model = MultiLabelMEDAF(args)
        model.to(self.device)

        print(
            f"Phase 1 Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Training setup
        criterion = {"bce": nn.BCEWithLogitsLoss()}
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["learning_rate"]
        )

        # Training
        phase1_metrics = []
        for epoch in range(self.config["num_epochs"]):
            metrics = train_multilabel(
                self.train_loader, model, criterion, optimizer, args, self.device
            )
            phase1_metrics.append(metrics)

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss={metrics:.4f}")

        self.results["phase1"] = {
            "model": model,
            "final_loss": phase1_metrics[-1],
            "metrics_history": phase1_metrics,
        }

        print(f"Phase 1 Final Loss: {phase1_metrics[-1]:.4f}")

    def demo_phase2_comparative(self):
        """Demonstrate Phase 2: Comparative Analysis"""
        print("\n" + "=" * 60)
        print("PHASE 2 DEMO: Per-Class Gating Comparative Analysis")
        print("=" * 60)

        # Base configuration
        base_args = {
            "img_size": self.config["img_size"],
            "backbone": "resnet18",
            "num_classes": self.config["num_classes"],
            "gate_temp": 100,
            "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
            "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
            "loss_wgts": [0.7, 1.0, 0.01],
            "enhanced_diversity": True,
            "diversity_type": "cosine",
        }

        # Configurations to compare
        configurations = {
            "global_gating": {
                "name": "Global Gating",
                "use_per_class_gating": False,
                "use_label_correlation": False,
            },
            "per_class_gating": {
                "name": "Per-Class Gating",
                "use_per_class_gating": True,
                "use_label_correlation": False,
            },
            "enhanced_per_class": {
                "name": "Enhanced Per-Class",
                "use_per_class_gating": True,
                "use_label_correlation": True,
                "gating_regularization": 0.01,
            },
        }

        # Create comparative framework
        framework = ComparativeTrainingFramework(base_args)
        criterion = {"bce": nn.BCEWithLogitsLoss()}

        phase2_results = {}

        for config_key, config_opts in configurations.items():
            print(f"\n--- Training {config_opts['name']} ---")

            # Merge configuration
            args = base_args.copy()
            args.update({k: v for k, v in config_opts.items() if k != "name"})

            # Create model
            model = MultiLabelMEDAFv2(args)
            model.to(self.device)

            # Print model info
            summary = model.get_gating_summary()
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Model: {summary['gating_type']} gating")
            print(f"Parameters: {param_count:,}")
            print(f"Label correlation: {summary['use_label_correlation']}")

            # Training
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.config["learning_rate"]
            )

            metrics_history = []
            best_acc = 0

            for epoch in range(self.config["num_epochs"]):
                framework.current_epoch = epoch

                metrics = train_multilabel_v2(
                    self.train_loader,
                    model,
                    criterion,
                    optimizer,
                    args,
                    self.device,
                    framework,
                )

                metrics_history.append(metrics)

                if metrics["subset_acc"] > best_acc:
                    best_acc = metrics["subset_acc"]

                if epoch % 2 == 0:
                    print(
                        f"Epoch {epoch}: Loss={metrics['total_loss']:.4f}, Acc={metrics['subset_acc']:.2f}%"
                    )

            phase2_results[config_key] = {
                "model": model,
                "config": config_opts,
                "best_accuracy": best_acc,
                "final_metrics": metrics_history[-1],
                "metrics_history": metrics_history,
            }

        # Print comparative analysis
        framework.print_comparison()

        self.results["phase2"] = phase2_results

    def analyze_attention_patterns(self):
        """Analyze attention patterns between global and per-class gating"""
        print("\n" + "=" * 60)
        print("ATTENTION PATTERN ANALYSIS")
        print("=" * 60)

        if "phase2" not in self.results:
            print("Phase 2 results not available for analysis")
            return

        # Get sample batch
        sample_batch = next(iter(self.test_loader))
        inputs, targets = sample_batch[0][:4].to(self.device), sample_batch[1][:4].to(
            self.device
        )

        print(f"Analyzing batch with shape: {inputs.shape}")
        print(f"Target labels:\n{targets}")

        # Analyze global vs per-class gating
        global_model = self.results["phase2"]["global_gating"]["model"]
        per_class_model = self.results["phase2"]["per_class_gating"]["model"]

        global_model.eval()
        per_class_model.eval()

        with torch.no_grad():
            # Global gating analysis
            global_outputs = global_model(
                inputs, targets, return_attention_weights=True
            )
            global_gate_pred = global_outputs["gate_pred"]

            print(f"\nGlobal Gating Weights (averaged across samples):")
            print(f"Expert preferences: {global_gate_pred.mean(dim=0)}")

            # Per-class gating analysis
            pc_outputs = per_class_model(inputs, targets, return_attention_weights=True)
            if "per_class_weights" in pc_outputs:
                pc_weights = pc_outputs["per_class_weights"]

                print(f"\nPer-Class Gating Weights:")
                print(f"Shape: {pc_weights.shape}")

                # Average expert preferences per class
                avg_class_prefs = pc_weights.mean(dim=0)
                print(f"Average expert preferences per class:")
                for class_idx in range(avg_class_prefs.shape[0]):
                    expert_prefs = avg_class_prefs[class_idx]
                    dominant_expert = expert_prefs.argmax().item()
                    max_pref = expert_prefs.max().item()
                    print(
                        f"  Class {class_idx}: Expert {dominant_expert} ({max_pref:.3f}) - {expert_prefs}"
                    )

                # Measure specialization
                expert_entropy = -(
                    avg_class_prefs * torch.log(avg_class_prefs + 1e-8)
                ).sum(dim=-1)
                avg_entropy = expert_entropy.mean().item()

                print(f"\nSpecialization Analysis:")
                print(
                    f"Average gating entropy: {avg_entropy:.3f} (lower = more specialized)"
                )
                print(f"Class entropies: {expert_entropy}")

                # Expert usage distribution
                expert_usage = avg_class_prefs.mean(dim=0)
                print(f"Overall expert usage: {expert_usage}")

    def plot_training_curves(self):
        """Plot training curves for comparison"""
        print("\n" + "=" * 60)
        print("PLOTTING TRAINING CURVES")
        print("=" * 60)

        if "phase2" not in self.results:
            print("Phase 2 results not available for plotting")
            return

        plt.figure(figsize=(15, 5))

        # Loss curves
        plt.subplot(1, 3, 1)
        for config_key, results in self.results["phase2"].items():
            metrics_history = results["metrics_history"]
            losses = [m["total_loss"] for m in metrics_history]
            plt.plot(losses, label=results["config"]["name"])
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Accuracy curves
        plt.subplot(1, 3, 2)
        for config_key, results in self.results["phase2"].items():
            metrics_history = results["metrics_history"]
            accuracies = [m["subset_acc"] for m in metrics_history]
            plt.plot(accuracies, label=results["config"]["name"])
        plt.title("Subset Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)

        # Diversity loss curves
        plt.subplot(1, 3, 3)
        for config_key, results in self.results["phase2"].items():
            metrics_history = results["metrics_history"]
            diversity_losses = [m["diversity_loss"] for m in metrics_history]
            plt.plot(diversity_losses, label=results["config"]["name"])
        plt.title("Diversity Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Diversity Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("multilabel_medaf_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Training curves saved as 'multilabel_medaf_comparison.png'")

    def print_final_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("MULTI-LABEL MEDAF DEMO SUMMARY")
        print("=" * 70)

        # Phase 1 summary
        if "phase1" in self.results:
            print("\n📊 Phase 1 (Basic Multi-Label MEDAF):")
            print(f"   Final Loss: {self.results['phase1']['final_loss']:.4f}")

        # Phase 2 summary
        if "phase2" in self.results:
            print("\n🎯 Phase 2 (Per-Class Gating Comparative Analysis):")

            for config_key, results in self.results["phase2"].items():
                config_name = results["config"]["name"]
                best_acc = results["best_accuracy"]
                final_loss = results["final_metrics"]["total_loss"]

                print(
                    f"   {config_name:20}: Acc={best_acc:6.2f}%, Loss={final_loss:.4f}"
                )

            # Find best configuration
            best_config = max(
                self.results["phase2"].items(), key=lambda x: x[1]["best_accuracy"]
            )

            print(f"\n🏆 Best Configuration: {best_config[1]['config']['name']}")
            print(f"   Best Accuracy: {best_config[1]['best_accuracy']:.2f}%")

            # Calculate improvements
            if (
                "global_gating" in self.results["phase2"]
                and "per_class_gating" in self.results["phase2"]
            ):
                global_acc = self.results["phase2"]["global_gating"]["best_accuracy"]
                pc_acc = self.results["phase2"]["per_class_gating"]["best_accuracy"]
                improvement = pc_acc - global_acc

                print(f"\n📈 Per-Class Gating Improvement: {improvement:+.2f}%")

                if improvement > 0:
                    print("   ✅ Per-class gating shows performance benefits!")
                else:
                    print(
                        "   ℹ️  Results may vary with longer training and real datasets"
                    )

        print(f"\n🔧 Configuration Used:")
        print(
            f"   Dataset: {self.config['num_samples']} samples, {self.config['num_classes']} classes"
        )
        print(
            f"   Training: {self.config['num_epochs']} epochs, batch size {self.config['batch_size']}"
        )
        print(f"   Device: {self.device}")

        print(f"\n📝 Key Insights:")
        print(f"   • Multi-label MEDAF successfully handles multiple labels per sample")
        print(f"   • Per-class gating enables class-specific expert specialization")
        print(
            f"   • Attention diversity encourages experts to focus on different regions"
        )
        print(f"   • Configurable architecture allows easy experimentation")

        print("\n🚀 Next Steps:")
        print("   1. Experiment with real multi-label datasets (PASCAL VOC, MS-COCO)")
        print("   2. Conduct comprehensive ablation studies")
        print("   3. Implement advanced research extensions")
        print("   4. Scale to larger models and datasets")

    def run_demo(self):
        """Run complete demonstration"""
        print("🎬 Multi-Label MEDAF Complete Demonstration")
        print("Phase 1: Basic Multi-Label + Phase 2: Per-Class Gating")

        # Create dataset
        self.create_dataset()

        # Demo Phase 1
        self.demo_phase1()

        # Demo Phase 2 with comparative analysis
        self.demo_phase2_comparative()

        # Analyze attention patterns
        self.analyze_attention_patterns()

        # Plot results
        try:
            self.plot_training_curves()
        except Exception as e:
            print(f"Plotting failed: {e} (matplotlib may not be available)")

        # Final summary
        self.print_final_summary()


def main():
    """Main demo function"""

    # Demo configuration
    config = {
        "num_samples": 200,
        "img_size": 32,
        "num_classes": 8,
        "avg_labels_per_sample": 3,
        "batch_size": 16,
        "num_epochs": 8,  # Short for demo
        "learning_rate": 0.001,
    }

    print("Multi-Label MEDAF Comprehensive Demo")
    print("====================================")
    print("This demo showcases both phases of Multi-Label MEDAF:")
    print("• Phase 1: Basic multi-label classification with global gating")
    print("• Phase 2: Per-class gating with comparative analysis")
    print(f"\nConfiguration: {config}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demo
    demo = MultiLabelMEDAFDemo(config)
    demo.run_demo()


if __name__ == "__main__":
    main()

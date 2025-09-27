#!/usr/bin/env python3
"""Multi-Label MEDAF Phase 1 training demo on NIH ChestX-ray14."""

import ast
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
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


KNOWN_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

DEFAULT_IMAGE_ROOT = Path("datasets/data/NIH/images-224")
DEFAULT_KNOWN_CSV = Path("datasets/data/NIH/chestxray_train_known.csv")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/medaf_phase1")


class ChestXrayKnownDataset(data.Dataset):
    """Dataset that reads the known-label ChestX-ray14 split for Phase 1 training."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        img_size: int = 224,
        max_samples: Optional[int] = None,
        transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.img_size = img_size
        self.class_names = KNOWN_LABELS
        self.num_classes = len(self.class_names)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"ChestX-ray CSV not found: {self.csv_path}")
        if not self.image_root.exists():
            raise FileNotFoundError(
                f"ChestX-ray image directory not found: {self.image_root}"
            )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        df = pd.read_csv(self.csv_path)
        if "known_labels" not in df.columns:
            raise ValueError(
                "Expected 'known_labels' column in CSV. Run utils/create_chestxray_splits.py first."
            )

        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.records = df.to_dict("records")
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}

    @staticmethod
    def _parse_label_list(raw_value):
        if isinstance(raw_value, list):
            return raw_value
        if pd.isna(raw_value):
            return []
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if not raw_value:
                return []
            try:
                parsed = ast.literal_eval(raw_value)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                if isinstance(parsed, str):
                    return [parsed]
            except (ValueError, SyntaxError):
                pass
            return [item.strip() for item in raw_value.split("|") if item.strip()]
        return []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.image_root / record["Image Index"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in self._parse_label_list(record.get("known_labels", [])):
            if label in self.label_to_idx:
                labels[self.label_to_idx[label]] = 1.0

        return image, labels


class MultiLabelMEDAFDemo:
    """
    Comprehensive demo for Multi-Label MEDAF
    """

    def __init__(self, config):
        self.config = config
        self.config.setdefault("data_source", "chestxray")
        self.config.setdefault("img_size", 224)
        self.config.setdefault("num_classes", len(KNOWN_LABELS))
        self.config.setdefault("val_ratio", 0.1)
        self.config.setdefault("num_workers", 0)
        self.config.setdefault("checkpoint_dir", str(DEFAULT_CHECKPOINT_DIR))
        self.config.setdefault("phase1_checkpoint", "medaf_phase1_chestxray.pt")
        self.config.setdefault("num_samples", 1000)
        self.config.setdefault("avg_labels_per_sample", 3)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset_name = None
        self.class_names = KNOWN_LABELS

    def create_dataset(self):
        """Create train/validation loaders for Phase 1."""

        data_source = self.config.get("data_source", "chestxray").lower()

        if data_source == "chestxray":
            csv_path = Path(self.config.get("known_csv", DEFAULT_KNOWN_CSV))
            image_root = Path(self.config.get("image_root", DEFAULT_IMAGE_ROOT))
            max_samples = self.config.get("max_samples")
            if isinstance(max_samples, str):
                max_samples = int(max_samples)
            print(f"Loading ChestX-ray14 known-label split from {csv_path}")
            dataset = ChestXrayKnownDataset(
                csv_path=csv_path,
                image_root=image_root,
                img_size=self.config.get("img_size", 224),
                max_samples=max_samples,
            )
            self.dataset_name = "ChestX-ray14 (known labels)"
            self.class_names = dataset.class_names
            self.config["num_classes"] = dataset.num_classes
            self.config["img_size"] = dataset.img_size
        else:
            print(
                "ChestX-ray data not requested or unavailable. Falling back to synthetic dataset."
            )
            dataset = SyntheticMultiLabelDataset(
                num_samples=self.config.get("num_samples", 1000),
                img_size=self.config.get("img_size", 32),
                num_classes=self.config.get("num_classes", 8),
                avg_labels_per_sample=self.config.get("avg_labels_per_sample", 3),
                random_state=42,
            )
            self.dataset_name = "Synthetic"
            self.class_names = [f"class_{i}" for i in range(dataset.num_classes)]

        val_ratio = float(self.config.get("val_ratio", 0.1))
        val_size = max(1, int(len(dataset) * val_ratio))
        train_size = len(dataset) - val_size

        train_dataset, val_dataset = data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        batch_size = self.config.get("batch_size", 16)
        num_workers = self.config.get("num_workers", 4)
        pin_memory = torch.cuda.is_available()

        self.train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.test_loader = self.val_loader

        print(
            f"Dataset prepared: {train_size} train / {val_size} val samples ({self.dataset_name})"
        )

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

        final_loss = phase1_metrics[-1] if phase1_metrics else float("nan")
        checkpoint_path = self.save_model(model, args, phase1_metrics)

        self.results["phase1"] = {
            "model": model,
            "final_loss": final_loss,
            "metrics_history": phase1_metrics,
            "checkpoint": str(checkpoint_path),
        }

        if phase1_metrics:
            print(f"Phase 1 Final Loss: {final_loss:.4f}")
        else:
            print("Phase 1 completed with zero epochs (no training performed)")
        print(f"Phase 1 checkpoint saved to: {checkpoint_path}")
        print(
            "Use load_phase1_checkpoint(CheckpointPath) to reload this model for evaluation."
        )

    def save_model(self, model, args, loss_history):
        ckpt_dir = Path(self.config["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / self.config["phase1_checkpoint"]

        payload = {
            "state_dict": model.state_dict(),
            "args": args,
            "class_names": self.class_names,
            "dataset": self.dataset_name,
        }
        torch.save(payload, checkpoint_path)

        metadata = {
            "dataset": self.dataset_name,
            "class_names": self.class_names,
            "num_epochs": self.config["num_epochs"],
            "batch_size": self.config.get("batch_size"),
            "learning_rate": self.config.get("learning_rate"),
            "loss_history": [float(loss) for loss in loss_history],
            "device": str(self.device),
            "checkpoint": str(checkpoint_path),
            "config": {
                k: v
                for k, v in self.config.items()
                if isinstance(v, (int, float, str, bool))
            },
        }
        metadata_path = checkpoint_path.with_suffix(".json")
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        return checkpoint_path

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

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            print(f"Matplotlib not available: {exc}")
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
            if "checkpoint" in self.results["phase1"]:
                print(f"   Checkpoint: {self.results['phase1']['checkpoint']}")

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

        train_count = len(self.train_loader.dataset) if self.train_loader else 0
        val_count = len(self.val_loader.dataset) if self.val_loader else 0

        print(f"\n🔧 Configuration Used:")
        print(
            f"   Dataset: {self.dataset_name} | train {train_count}, val {val_count}, {self.config['num_classes']} classes"
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
        if self.config.get("run_phase2"):
            print("Phase 1: Basic Multi-Label + Phase 2: Per-Class Gating")
        else:
            print("Phase 1: Basic Multi-Label Training")

        # Create dataset
        self.create_dataset()

        # Demo Phase 1
        self.demo_phase1()

        # Demo Phase 2 with comparative analysis
        if self.config.get("run_phase2"):
            self.demo_phase2_comparative()

        # Analyze attention patterns
        if self.config.get("run_phase2"):
            self.analyze_attention_patterns()

        # Plot results
        if self.config.get("run_phase2"):
            try:
                self.plot_training_curves()
            except Exception as e:
                print(f"Plotting failed: {e} (matplotlib may not be available)")

        # Final summary
        self.print_final_summary()


def load_phase1_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device, None] = None,
):
    """Load a saved Phase 1 MEDAF checkpoint."""

    device_obj = torch.device(device) if device else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    args = checkpoint.get("args")
    if args is None:
        raise KeyError("Checkpoint is missing 'args'.")

    model = MultiLabelMEDAF(args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device_obj)
    model.eval()

    return model, checkpoint


def main():
    """Main demo function"""

    # Demo configuration
    config = {
        "data_source": "chestxray",
        "known_csv": str(DEFAULT_KNOWN_CSV),
        "image_root": str(DEFAULT_IMAGE_ROOT),
        "batch_size": 32,
        "num_epochs": 5,
        "learning_rate": 1e-4,
        "val_ratio": 0.1,
        "num_workers": 2,
        # "max_samples": None,  # Set to an int for quicker experiments
        "max_samples": 1000,
        "phase1_checkpoint": "medaf_phase1_chestxray.pt",
        "checkpoint_dir": str(DEFAULT_CHECKPOINT_DIR),
        "run_phase2": False,
    }

    print("Multi-Label MEDAF Comprehensive Demo")
    print("====================================")
    if config.get("run_phase2"):
        print("This demo showcases both phases of Multi-Label MEDAF:")
        print("• Phase 1: Basic multi-label classification with global gating")
        print("• Phase 2: Per-class gating with comparative analysis")
    else:
        print(
            "This run focuses on Phase 1: training MEDAF on ChestX-ray14 known labels."
        )
    print(f"\nConfiguration: {config}")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demo
    demo = MultiLabelMEDAFDemo(config)
    demo.run_demo()


if __name__ == "__main__":
    main()

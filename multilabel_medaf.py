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

from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_train import train_multilabel


# ===== MEMORY OPTIMIZATION =====

import gc
import os

from test_multilabel_medaf import ChestXrayKnownDataset

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
gc.collect()

# Set memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Reduce memory usage
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)

print("✅ Memory optimizations applied")


import os

CURRENT_DIR = "/home/s2320437/WORK/aidan-medaf/"
os.chdir(CURRENT_DIR)
print(f"Current working directory: {os.getcwd()}")


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

DEFAULT_IMAGE_ROOT = Path(f"{CURRENT_DIR}/datasets/data/chestxray/NIH/images-224")
DEFAULT_KNOWN_CSV = Path(
    f"{CURRENT_DIR}/datasets/data/chestxray/NIH/chestxray_train_known.csv"
)

DEFAULT_CHECKPOINT_DIR = Path(f"{CURRENT_DIR}/checkpoints/medaf_phase1")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}
train_loader = None
val_loader = None
test_loader = None
dataset_name = None
class_names = KNOWN_LABELS


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# Demo configuration
config = {
    "data_source": "chestxray",
    "known_csv": str(DEFAULT_KNOWN_CSV),
    "image_root": str(DEFAULT_IMAGE_ROOT),
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "val_ratio": 0.1,
    "num_workers": 1,
    "max_samples": None,  # Set to an int for quicker experiments
    # "max_samples": 1000,
    "phase1_checkpoint": "medaf_phase1_chestxray.pt",
    "checkpoint_dir": str(DEFAULT_CHECKPOINT_DIR),
    "run_phase2": False,
}


def save_model(model, args, loss_history):
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / config["phase1_checkpoint"]

    payload = {
        "state_dict": model.state_dict(),
        "args": args,
        "class_names": class_names,
        "dataset": dataset_name,
    }
    torch.save(payload, checkpoint_path)

    metadata = {
        "dataset": dataset_name,
        "class_names": class_names,
        "num_epochs": config["num_epochs"],
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "loss_history": [float(loss) for loss in loss_history],
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "config": {
            k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))
        },
    }
    metadata_path = checkpoint_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return checkpoint_path


def print_evaluation_results(results):
    """Print comprehensive evaluation results"""
    print("\n" + "=" * 60)
    print("🎯 MEDAF MODEL EVALUATION RESULTS")
    print("=" * 60)

    # Overall metrics
    overall = results["overall"]
    print(f"\n📊 Overall Performance:")
    print(
        f"   Subset Accuracy:  {overall['subset_accuracy']:.4f} ({overall['subset_accuracy']*100:.2f}%)"
    )
    print(
        f"   Hamming Accuracy: {overall['hamming_accuracy']:.4f} ({overall['hamming_accuracy']*100:.2f}%)"
    )
    print(f"   Precision:        {overall['precision']:.4f}")
    print(f"   Recall:           {overall['recall']:.4f}")
    print(f"   F1-Score:         {overall['f1_score']:.4f}")
    print(f"   Average Loss:     {overall['average_loss']:.4f}")

    # Per-expert comparison
    print(f"\n🔬 Per-Expert Performance:")
    for expert_name, expert_metrics in results["per_expert"].items():
        print(
            f"   {expert_name:>8}: Subset={expert_metrics['subset_accuracy']:.4f}, "
            f"Hamming={expert_metrics['hamming_accuracy']:.4f}"
        )

    # Per-class metrics
    print(f"\n🏷️  Per-Class Performance:")
    class_names = results["class_names"]
    precision = results["per_class"]["precision"]
    recall = results["per_class"]["recall"]
    f1_score = results["per_class"]["f1_score"]

    print(f"   {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("   " + "-" * 50)

    for i, class_name in enumerate(class_names):
        print(
            f"   {class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_score[i]:<10.4f}"
        )

    # Best performing classes
    best_f1_idx = max(range(len(f1_score)), key=lambda i: f1_score[i])
    worst_f1_idx = min(range(len(f1_score)), key=lambda i: f1_score[i])

    print(
        f"\n🏆 Best Class:  {class_names[best_f1_idx]} (F1={f1_score[best_f1_idx]:.4f})"
    )
    print(
        f"📉 Worst Class: {class_names[worst_f1_idx]} (F1={f1_score[worst_f1_idx]:.4f})"
    )


def evaluate_medaf_final(model, data_loader, device, class_names, threshold=0.1):
    """Final clean evaluation function - removes all duplicates"""
    model.eval()

    print(f"🔍 FINAL EVALUATION (Threshold: {threshold})")
    print("=" * 50)

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]  # Use gating network

            # Convert to predictions
            probs = torch.sigmoid(gate_logits)
            pred_binary = (probs > threshold).float()

            # Store data
            all_predictions.append(pred_binary.cpu())
            all_targets.append(targets.cpu())

            # Compute loss
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(gate_logits, targets.float())
            total_loss += loss.item()
            num_batches += 1

    # Concatenate all data
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    subset_acc = (all_predictions == all_targets).all(dim=1).float().mean().item()
    hamming_acc = (all_predictions == all_targets).float().mean().item()

    # Per-class metrics
    tp = (all_predictions * all_targets).sum(dim=0)
    fp = (all_predictions * (1 - all_targets)).sum(dim=0)
    fn = ((1 - all_predictions) * all_targets).sum(dim=0)

    precision = torch.zeros_like(tp, dtype=torch.float32)
    recall = torch.zeros_like(tp, dtype=torch.float32)
    f1 = torch.zeros_like(tp, dtype=torch.float32)

    for i in range(len(tp)):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    # Compile results
    results = {
        "overall": {
            "subset_accuracy": subset_acc,
            "hamming_accuracy": hamming_acc,
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1_score": f1.mean().item(),
            "average_loss": total_loss / num_batches,
            "threshold_used": threshold,
        },
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1.tolist(),
        },
        "class_names": class_names,
    }

    return results


def print_final_results(results):
    """Print final clean results"""
    print("\n" + "=" * 60)
    print("🎯 MEDAF FINAL EVALUATION RESULTS")
    print("=" * 60)

    overall = results["overall"]
    print(f"\n📊 Overall Performance:")
    print(
        f"   Subset Accuracy:  {overall['subset_accuracy']:.4f} ({overall['subset_accuracy']*100:.2f}%)"
    )
    print(
        f"   Hamming Accuracy: {overall['hamming_accuracy']:.4f} ({overall['hamming_accuracy']*100:.2f}%)"
    )
    print(f"   Precision:        {overall['precision']:.4f}")
    print(f"   Recall:           {overall['recall']:.4f}")
    print(f"   F1-Score:         {overall['f1_score']:.4f}")
    print(f"   Average Loss:     {overall['average_loss']:.4f}")
    print(f"   Threshold Used:   {overall['threshold_used']}")

    # Per-class performance
    print(f"\n🏷️  Per-Class Performance:")
    class_names = results["class_names"]
    precision = results["per_class"]["precision"]
    recall = results["per_class"]["recall"]
    f1_score = results["per_class"]["f1_score"]

    print(f"   {'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("   " + "-" * 50)

    for i, class_name in enumerate(class_names):
        print(
            f"   {class_name:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_score[i]:<10.4f}"
        )

    # Best and worst classes
    best_f1_idx = max(range(len(f1_score)), key=lambda i: f1_score[i])
    worst_f1_idx = min(range(len(f1_score)), key=lambda i: f1_score[i])

    print(
        f"\n🏆 Best Class:  {class_names[best_f1_idx]} (F1={f1_score[best_f1_idx]:.4f})"
    )
    print(
        f"📉 Worst Class: {class_names[worst_f1_idx]} (F1={f1_score[worst_f1_idx]:.4f})"
    )

    # Model assessment
    print(f"\n💡 Model Assessment:")
    if overall["f1_score"] < 0.1:
        print("   ⚠️  Model is under-trained (F1 < 0.1)")
        print("   📈 Recommendation: Train with more data and epochs")
    elif overall["f1_score"] < 0.3:
        print("   🔶 Model shows some learning but needs improvement")
    else:
        print("   ✅ Model shows good performance")


def evaluation():
    # ===== RUN FINAL EVALUATION =====

    print("\n" + "=" * 60)
    print("🔍 FINAL EVALUATION - CLEAN VERSION")
    print("=" * 60)

    # Load and evaluate the trained model
    checkpoint_path = results["phase1"]["checkpoint"]
    print(f"📁 Loading checkpoint: {checkpoint_path}")

    THRESHOLD = 0.5

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint.get("args", {})

        if not args:
            raise ValueError("Checkpoint missing 'args'")

        # Create and load model
        model = MultiLabelMEDAF(args)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Evaluate with optimal threshold
        final_results = evaluate_medaf_final(
            model, val_loader, device, class_names, threshold=THRESHOLD
        )

        # Print results
        print_final_results(final_results)

        # Save results
        eval_save_path = Path(checkpoint_path).parent / "final_evaluation_results.json"
        with open(eval_save_path, "w") as f:
            json_results = {}
            for key, value in final_results.items():
                if isinstance(value, dict):
                    json_results[key] = {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in value.items()
                    }
                else:
                    json_results[key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )

            json.dump(json_results, f, indent=2)

        print(f"\n💾 Final results saved to: {eval_save_path}")

        # Store results
        results["final_evaluation"] = final_results
        results["loaded_model"] = model

        print(f"\n✅ FINAL EVALUATION COMPLETE")
        print(f"   Model successfully evaluated on validation set")
        print(f"   Ready for full dataset training")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    data_source = config.get("data_source", "chestxray").lower()

    if data_source == "chestxray":
        csv_path = Path(config.get("known_csv", DEFAULT_KNOWN_CSV))
        image_root = Path(config.get("image_root", DEFAULT_IMAGE_ROOT))
        max_samples = config.get("max_samples")
        if isinstance(max_samples, str):
            max_samples = int(max_samples)
        print(f"Loading ChestX-ray14 known-label split from {csv_path}")
        dataset = ChestXrayKnownDataset(
            csv_path=csv_path,
            image_root=image_root,
            img_size=config.get("img_size", 224),
            max_samples=max_samples,
        )
        dataset_name = "ChestX-ray14 (known labels)"
        config["num_classes"] = dataset.num_classes
        config["img_size"] = dataset.img_size

    val_ratio = float(config.get("val_ratio", 0.1))
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 4)
    pin_memory = torch.cuda.is_available()

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = val_loader

    print(
        f"Dataset prepared: {train_size} train / {val_size} val samples ({dataset_name})"
    )

    """Demonstrate Phase 1: Basic Multi-Label MEDAF"""

    print("\n" + "=" * 60)
    print("PHASE 1: Basic Multi-Label MEDAF")
    print("=" * 60)

    # Configuration for Phase 1
    args = {
        "img_size": config["img_size"],
        "backbone": "resnet18",
        "num_classes": config["num_classes"],
        "gate_temp": 100,
        "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
        "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
        # [expert_weight, gate_weight, diversity_weight] = 0.7 * (b1, b2, b3) + 1.0 * (gate) + 0.01 * (divAttn)
        "loss_wgts": [0.7, 1.0, 0.01],
    }

    # Create Phase 1 model
    model = MultiLabelMEDAF(args)
    model.to(device)

    print(
        f"Phase 1 Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Training setup
    criterion = {"bce": nn.BCEWithLogitsLoss()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training
    phase1_metrics = []
    for epoch in range(config["num_epochs"]):
        metrics = train_multilabel(
            train_loader, model, criterion, optimizer, args, device
        )
        phase1_metrics.append(metrics)

        if epoch % 2 == 0:
            print(f"\n==== Epoch {epoch}: Loss={metrics:.4f} ====")

    final_loss = phase1_metrics[-1] if phase1_metrics else float("nan")
    checkpoint_path = save_model(model, args, phase1_metrics)

    results["phase1"] = {
        "model": model,
        "final_loss": final_loss,
        "metrics_history": phase1_metrics,
        "checkpoint": str(checkpoint_path),
    }
    print("-" * 40)
    print("\n\n")
    if phase1_metrics:
        print(f"\n==== Phase 1 Final Loss: {final_loss:.4f} ====")
    else:
        print("Phase 1 completed with zero epochs (no training performed)")
    print(f"\n==== Phase 1 checkpoint saved to: {checkpoint_path} ====")


if __name__ == "__main__":
    main()

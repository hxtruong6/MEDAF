import ast
import json
import time
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
from core.training_utils import (
    calculate_class_weights,
    optimize_thresholds_per_class,
    evaluate_with_optimal_thresholds,
    train_multilabel_enhanced,
    print_enhanced_results,
)


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


def setup_resume_training(checkpoint_path):
    """
    Convenient function to set up resuming training from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file to resume from

    Example:
        # To resume training from your latest checkpoint:
        setup_resume_training(
            "/home/s2320437/WORK/aidan-medaf/checkpoints/medaf_phase1/medaf_phase1_chestxray_epoch_19_1759539211.pt"
        )
    """
    global config
    config["resume_from_checkpoint"] = checkpoint_path
    print(f"🔄 Resume training configured:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Total epochs to train: {config['num_epochs']}")
    print(f"   Run the script to start training!")


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
DEFAULT_TRAIN_CSV = Path(
    f"{CURRENT_DIR}/datasets/data/chestxray/NIH/chestxray_strategy1_train.csv"
)
DEFAULT_TEST_CSV = Path(
    f"{CURRENT_DIR}/datasets/data/chestxray/NIH/chestxray_strategy1_test.csv"
)

DEFAULT_CHECKPOINT_DIR = Path(f"{CURRENT_DIR}/checkpoints/medaf_phase1")

EVALUATION_CHECKPOINT = Path(
    f"{CURRENT_DIR}/checkpoints/medaf_phase1/medaf_phase1_chestxray_epoch_29_1759624078.pt"
)


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
    "train_csv": str(DEFAULT_TRAIN_CSV),
    "test_csv": str(DEFAULT_TEST_CSV),
    "image_root": str(DEFAULT_IMAGE_ROOT),
    "batch_size": 32,
    "num_epochs": 50,  # Increased for better convergence
    "learning_rate": 5e-5,  # Lower LR for medical data stability
    "val_ratio": 0.1,
    "num_workers": 1,
    "max_samples": None,  # Set to an int for quicker experiments
    # "max_samples": 100,
    "phase1_checkpoint": "medaf_phase1_chestxray_enhanced.pt",
    "checkpoint_dir": str(DEFAULT_CHECKPOINT_DIR),
    "run_phase2": False,
    # Enhanced training configuration
    "use_class_weights": True,  # Enable class weighting for imbalanced data
    "class_weight_method": "inverse_freq",  # 'inverse_freq', 'effective_num', or 'focal'
    "use_optimal_thresholds": True,  # Enable per-class threshold optimization
    "enhanced_training": True,  # Use enhanced training loop
    # Resume training configuration
    "resume_from_checkpoint": None,  # Path to checkpoint to resume from
    # "resume_from_checkpoint": "checkpoints/medaf_phase1/medaf_phase1_chestxray_epoch_19_1759539211.pt",
}


# Configuration for Phase 1
args = {
    "img_size": config.get("img_size", 224),
    "backbone": "resnet18",
    "num_classes": config.get("num_classes", 8),
    "gate_temp": 100,
    "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
    "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
    # [expert_weight, gate_weight, diversity_weight] = 0.7 * (b1, b2, b3) + 1.0 * (gate) + 0.01 * (divAttn)
    "loss_wgts": [0.7, 1.0, 0.01],
}

def save_model(model, args, loss_history, suffix, optimizer=None, epoch=None):
    """Enhanced save function that includes optimizer state and epoch for resuming training"""
    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Split the filename and extension to insert suffix before extension
    base_filename = config["phase1_checkpoint"]
    if base_filename.endswith(".pt"):
        name_without_ext = base_filename[:-3]  # Remove .pt
        checkpoint_path = ckpt_dir / f"{name_without_ext}{suffix}.pt"
    else:
        checkpoint_path = ckpt_dir / (base_filename + suffix)

    payload = {
        "state_dict": model.state_dict(),
        "args": args,
        "class_names": class_names,
        "dataset": dataset_name,
        "loss_history": [float(loss) for loss in loss_history],
        "epoch": epoch,
    }

    # Add optimizer state if provided (for resuming training)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

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
        "epoch": epoch,
        "has_optimizer_state": optimizer is not None,
        "config": {
            k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))
        },
    }
    metadata_path = checkpoint_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return checkpoint_path


def load_checkpoint_for_resume(checkpoint_path, model, optimizer=None):
    """Load checkpoint and return model, optimizer, start_epoch, and loss_history for resuming training"""
    print(f"📁 Loading checkpoint for resume: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["state_dict"])
    print(f"✅ Model state loaded successfully")

    # Load optimizer state if available and optimizer provided
    start_epoch = 0
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"✅ Optimizer state loaded successfully")
    elif optimizer is not None:
        print(f"⚠️  No optimizer state in checkpoint - starting with fresh optimizer")

    # Get starting epoch
    if "epoch" in checkpoint and checkpoint["epoch"] is not None:
        start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
        print(f"📊 Resuming from epoch {start_epoch}")
    else:
        print(f"⚠️  No epoch info in checkpoint - starting from epoch 0")

    # Get loss history
    loss_history = checkpoint.get("loss_history", [])
    print(f"📈 Loaded {len(loss_history)} previous loss values")

    # Verify model args match
    checkpoint_args = checkpoint.get("args", {})
    print(f"🔧 Checkpoint args: {checkpoint_args}")

    return model, optimizer, start_epoch, loss_history, checkpoint_args


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
    """Final clean evaluation function - optimized for performance"""
    model.eval()

    print(f"🔍 FINAL EVALUATION (Threshold: {threshold}) | Device: {device}")
    print(f"📊 Total batches to process: {len(data_loader)}")
    print("=" * 50)

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    # Create criterion once outside the loop for efficiency
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # Show progress every 10% of batches
            if batch_idx % max(1, len(data_loader) // 10) == 0:
                progress = (batch_idx / len(data_loader)) * 100
                print(
                    f"   Progress: {progress:.1f}% ({batch_idx}/{len(data_loader)} batches)"
                )

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]  # Use gating network

            # Convert to predictions
            probs = torch.sigmoid(gate_logits)
            pred_binary = (probs > threshold).float()

            # Store data (keep on GPU for now to reduce transfers)
            all_predictions.append(pred_binary)
            all_targets.append(targets)

            # Compute loss
            loss = criterion(gate_logits, targets.float())
            total_loss += loss.item()
            num_batches += 1

    print("   📊 Concatenating predictions and computing metrics...")

    # Concatenate all data (keep on GPU for faster computation)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(
        f"   📈 Processing {all_predictions.shape[0]} samples with {all_predictions.shape[1]} classes"
    )

    # Compute metrics on GPU for speed
    subset_acc = (all_predictions == all_targets).all(dim=1).float().mean().item()
    hamming_acc = (all_predictions == all_targets).float().mean().item()

    # Per-class metrics (vectorized computation on GPU)
    tp = (all_predictions * all_targets).sum(dim=0)
    fp = (all_predictions * (1 - all_targets)).sum(dim=0)
    fn = ((1 - all_predictions) * all_targets).sum(dim=0)

    # Vectorized precision/recall/f1 computation (avoid loops)
    precision = torch.zeros_like(tp, dtype=torch.float32, device=device)
    recall = torch.zeros_like(tp, dtype=torch.float32, device=device)
    f1 = torch.zeros_like(tp, dtype=torch.float32, device=device)

    # Use torch.where for vectorized conditional operations
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2 * (precision * recall) / (precision + recall),
        torch.zeros_like(tp),
    )

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


def evaluation(checkpoint_path=EVALUATION_CHECKPOINT):
    # ===== RUN FINAL EVALUATION =====

    print("\n" + "=" * 60)
    print("🔍 FINAL EVALUATION -")
    print("=" * 60)
    print(f"   Checkpoint: {checkpoint_path}")

    # Load and evaluate the trained model
    print(f"📁 Loading checkpoint: {checkpoint_path}")

    THRESHOLD = 0.5

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        args = checkpoint.get("args", {})

        args = {
            "img_size": config.get("img_size", 224),
            "backbone": "resnet18",
            "num_classes": config.get("num_classes", 8),
            "gate_temp": 100,
            "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
            "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
            # [expert_weight, gate_weight, diversity_weight] = 0.7 * (b1, b2, b3) + 1.0 * (gate) + 0.01 * (divAttn)
            "loss_wgts": [0.7, 1.0, 0.01],
        }

        if not args:
            raise ValueError("Checkpoint missing 'args'")

        # Create and load model
        model = MultiLabelMEDAF(args)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create test loader from the actual test dataset
        global test_loader
        print(f"Loading ChestX-ray14 test dataset from {config.get('test_csv')}")
        test_dataset = ChestXrayKnownDataset(
            csv_path=config.get("test_csv"),
            image_root=config.get("image_root"),
            img_size=config.get("img_size", 224),
            # max_samples=None,  # Use full test set
            max_samples=config.get("max_samples", None),
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=config.get("batch_size", 16),
            shuffle=False,
            num_workers=config.get("num_workers", 1),
            pin_memory=torch.cuda.is_available(),
        )

        print(f"   Test dataset size: {len(test_dataset)}")

        # Enhanced evaluation with optimal thresholds
        if config.get("use_optimal_thresholds", False):
            print(f"\n🎯 EVALUATION WITH OPTIMAL THRESHOLDS")
            print("=" * 60)

            # First, optimize thresholds using validation split from test data
            # Create a small validation split for threshold optimization
            test_size = len(test_dataset)
            val_size = min(
                1000, test_size // 5
            )  # Use 20% or max 1000 samples for threshold optimization
            eval_size = test_size - val_size

            val_dataset_for_thresh, eval_dataset = data.random_split(
                test_dataset,
                [val_size, eval_size],
                generator=torch.Generator().manual_seed(42),
            )

            val_loader_for_thresh = data.DataLoader(
                val_dataset_for_thresh,
                batch_size=config.get("batch_size", 16),
                shuffle=False,
                num_workers=config.get("num_workers", 1),
                pin_memory=torch.cuda.is_available(),
            )

            eval_loader = data.DataLoader(
                eval_dataset,
                batch_size=config.get("batch_size", 16),
                shuffle=False,
                num_workers=config.get("num_workers", 1),
                pin_memory=torch.cuda.is_available(),
            )

            # Optimize thresholds
            optimal_thresholds, threshold_metrics = optimize_thresholds_per_class(
                model, val_loader_for_thresh, device, len(class_names), class_names
            )

            # Evaluate with optimal thresholds
            final_results = evaluate_with_optimal_thresholds(
                model, eval_loader, device, optimal_thresholds, class_names
            )

            # Print enhanced results
            print_enhanced_results(final_results)

        else:
            # Standard evaluation with fixed threshold
            final_results = evaluate_medaf_final(
                model, test_loader, device, class_names, threshold=THRESHOLD
            )

            # Print results
            print_final_results(final_results)

        # Save results
        current_time = int(time.time())
        eval_save_path = (
            Path(checkpoint_path).parent
            / f"final_evaluation_results_{THRESHOLD}_{current_time}.json"
        )
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
        train_csv_path = Path(config.get("train_csv", DEFAULT_TRAIN_CSV))
        test_csv_path = Path(config.get("test_csv", DEFAULT_TEST_CSV))
        image_root = Path(config.get("image_root", DEFAULT_IMAGE_ROOT))
        max_samples = config.get("max_samples")
        if isinstance(max_samples, str):
            max_samples = int(max_samples)

        print(f"Loading ChestX-ray14 train dataset from {train_csv_path}")
        train_full_dataset = ChestXrayKnownDataset(
            csv_path=train_csv_path,
            image_root=image_root,
            img_size=config.get("img_size", 224),
            max_samples=max_samples,
        )

        dataset_name = "ChestX-ray14 (strategy1 split)"
        config["num_classes"] = train_full_dataset.num_classes
        config["img_size"] = train_full_dataset.img_size

    # Create validation split from training data only
    val_ratio = float(config.get("val_ratio", 0.1))
    val_size = max(1, int(len(train_full_dataset) * val_ratio))
    train_size = len(train_full_dataset) - val_size
    print(f"   Train size: {train_size} | Val size: {val_size}")

    train_dataset, val_dataset = data.random_split(
        train_full_dataset,
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

    print(
        f"Dataset prepared: {train_size} train / {val_size} val samples ({dataset_name})"
    )

    """Demonstrate Phase 1: Basic Multi-Label MEDAF"""

    print("\n" + "=" * 60)
    print("PHASE 1: Basic Multi-Label MEDAF")
    print("=" * 60)

    # Create Phase 1 model
    model = MultiLabelMEDAF(args)
    model.to(device)

    print(
        f"Phase 1 Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Enhanced Training setup with class weights
    print(f"\n🔧 Setting up enhanced training...")
    print(f"   Class weighting: {config.get('use_class_weights', False)}")
    print(f"   Enhanced training: {config.get('enhanced_training', False)}")

    # Calculate class weights if enabled
    if config.get("use_class_weights", False):
        print(
            f"   Calculating class weights using method: {config.get('class_weight_method', 'inverse_freq')}"
        )
        pos_weights = calculate_class_weights(
            train_loader,
            config["num_classes"],
            device,
            method=config.get("class_weight_method", "inverse_freq"),
        )
        criterion = {"bce": nn.BCEWithLogitsLoss(pos_weight=pos_weights)}
        print(f"   ✅ Class-weighted BCE loss initialized")
    else:
        criterion = {"bce": nn.BCEWithLogitsLoss()}
        print(f"   📝 Standard BCE loss (no class weighting)")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    print(f"   📊 Optimizer: Adam with LR={config['learning_rate']}")

    # Check if we should resume from checkpoint
    start_epoch = 0
    phase1_metrics = []

    if config.get("resume_from_checkpoint"):
        try:
            print(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
            print("=" * 50)

            model, optimizer, start_epoch, phase1_metrics, checkpoint_args = (
                load_checkpoint_for_resume(
                    config["resume_from_checkpoint"], model, optimizer
                )
            )

            # Verify args compatibility
            for key in ["num_classes", "img_size", "backbone"]:
                if key in checkpoint_args and checkpoint_args[key] != args.get(key):
                    print(
                        f"⚠️  Warning: {key} mismatch - checkpoint: {checkpoint_args[key]}, current: {args.get(key)}"
                    )

            # Use configured num_epochs as total epochs
            total_epochs = config["num_epochs"]
            print(
                f"📊 Training plan: Resume from epoch {start_epoch} → train until epoch {total_epochs-1}"
            )
            print(f"📈 Previous training loss history: {len(phase1_metrics)} epochs")

            # Check if we're already at or past the target epochs
            if start_epoch >= total_epochs:
                print(
                    f"⚠️  Warning: Already trained {start_epoch} epochs, target is {total_epochs}"
                )
                print(
                    f"   No additional training needed. Increase num_epochs if you want more training."
                )

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"🔄 Starting fresh training instead...")
            start_epoch = 0
            phase1_metrics = []
            total_epochs = config["num_epochs"]
    else:
        total_epochs = config["num_epochs"]
        print(f"🆕 Starting fresh training for {total_epochs} epochs")

    # Training loop with enhanced training
    print(
        f"\n🚀 Starting enhanced training from epoch {start_epoch} to {total_epochs-1}"
    )

    # Choose training function based on configuration
    if config.get("enhanced_training", False):
        train_function = train_multilabel_enhanced
        print(f"   Using enhanced training with class weighting")
    else:
        train_function = train_multilabel
        print(f"   Using standard training")

    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{total_epochs}")
        print(f"{'='*60}")

        metrics = train_function(
            train_loader, model, criterion, optimizer, args, device
        )
        phase1_metrics.append(metrics)

        if epoch % 2 == 0:
            print(f"\n==== Epoch {epoch}: Loss={metrics:.4f} ====")

    final_loss = phase1_metrics[-1] if phase1_metrics else float("nan")

    suffix = f"_epoch_{total_epochs-1}_{int(time.time())}"
    checkpoint_path = save_model(
        model, args, phase1_metrics, suffix, optimizer, total_epochs - 1
    )

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
        print(f"==== Total epochs trained: {len(phase1_metrics)} ====")
        if config.get("resume_from_checkpoint"):
            print(
                f"==== Resumed from epoch {start_epoch}, trained until epoch {total_epochs - 1} ===="
            )
    else:
        print("Phase 1 completed with zero epochs (no training performed)")
    print(f"\n==== Phase 1 checkpoint saved to: {checkpoint_path} ====")


if __name__ == "__main__":
    # === Training Mode ===
    # Check if user wants to resume training
    # if config.get("resume_from_checkpoint"):
    #     print("🔄 RESUMING TRAINING MODE")
    #     main()
    # else:
    #     print("🆕 FRESH TRAINING MODE")
    #     print("💡 To resume from checkpoint, see resume_training_example()")
    #     main()

    # === Evaluation Mode ===
    evaluation()

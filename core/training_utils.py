"""
Shared Training Utilities for MEDAF
Consolidates common functions to prevent code duplication
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from misc.util import AverageMeter, update_meter


# ============================================================================
# CLASS WEIGHT CALCULATION
# ============================================================================


def calculate_class_weights(
    train_loader, num_classes: int, device: torch.device, method: str = "inverse_freq"
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced multi-label data

    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to put weights on
        method: 'inverse_freq', 'effective_num', or 'focal'

    Returns:
        pos_weights: Tensor of positive class weights for BCEWithLogitsLoss
    """
    print("🔍 Calculating class weights for imbalanced data...")

    class_counts = torch.zeros(num_classes, device=device)
    total_samples = 0

    # Count positive samples for each class
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets = targets.to(device)
        class_counts += targets.sum(dim=0)
        total_samples += targets.shape[0]

        if batch_idx % 100 == 0:
            print(f"   Processed {batch_idx}/{len(train_loader)} batches")

    # Calculate negative counts
    neg_counts = total_samples - class_counts

    if method == "inverse_freq":
        # Standard inverse frequency weighting
        pos_weights = neg_counts / (class_counts + 1e-8)
    elif method == "effective_num":
        # Effective number of samples (handles class imbalance better)
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        pos_weights = (1.0 - beta) / (effective_num + 1e-8)
        pos_weights = pos_weights / pos_weights.min()  # Normalize
    elif method == "focal":
        # Focal loss inspired weighting
        pos_weights = torch.pow(neg_counts / (class_counts + 1e-8), 0.25)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # Clamp weights to reasonable range
    pos_weights = torch.clamp(pos_weights, min=0.1, max=50.0)

    # Print statistics
    _print_class_statistics(class_counts, neg_counts, pos_weights, total_samples)

    return pos_weights


def _print_class_statistics(
    class_counts: torch.Tensor,
    neg_counts: torch.Tensor,
    pos_weights: torch.Tensor,
    total_samples: int,
):
    """Print class distribution statistics"""
    print(f"\n📊 Class Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Class distribution:")

    class_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
    ]

    for i, name in enumerate(class_names):
        if i < len(class_counts):
            pos_count = int(class_counts[i].item())
            neg_count = int(neg_counts[i].item())
            pos_ratio = pos_count / total_samples * 100
            weight = pos_weights[i].item()
            print(
                f"   {name:15}: {pos_count:6d} pos ({pos_ratio:5.2f}%) | Weight: {weight:6.2f}"
            )


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================


def optimize_thresholds_per_class(
    model: nn.Module,
    val_loader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[float], Dict[str, Dict[str, float]]]:
    """
    Find optimal threshold for each class using validation data

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device
        num_classes: Number of classes
        class_names: List of class names for display

    Returns:
        optimal_thresholds: List of optimal thresholds per class
        threshold_metrics: Dictionary with metrics for each threshold
    """
    print("🎯 Optimizing per-class thresholds...")

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    model.eval()
    all_probs = []
    all_targets = []

    # Collect predictions and targets
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Get model predictions (use gate logits - the final fused predictions)
            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]  # Use gating network output
            probs = torch.sigmoid(gate_logits)

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

            if batch_idx % 50 == 0:
                print(f"   Processed {batch_idx}/{len(val_loader)} validation batches")

    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0)  # [N, num_classes]
    all_targets = torch.cat(all_targets, dim=0)  # [N, num_classes]

    print(f"   Total validation samples: {all_probs.shape[0]}")

    # Find optimal threshold for each class
    optimal_thresholds = []
    threshold_metrics = {}

    print(f"\n🔍 Per-class threshold optimization:")
    print(
        f"   {'Class':<15} {'Optimal':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Samples':<8}"
    )
    print("   " + "-" * 70)

    for class_idx in range(num_classes):
        class_probs = all_probs[:, class_idx]
        class_targets = all_targets[:, class_idx]

        # Skip if no positive samples
        pos_samples = class_targets.sum().item()
        if pos_samples == 0:
            print(
                f"   {class_names[class_idx]:<15} {'0.50':<8} {'0.0000':<8} {'0.0000':<10} {'0.0000':<8} {pos_samples:<8}"
            )
            optimal_thresholds.append(0.5)
            continue

        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}

        # Test different thresholds
        thresholds = np.arange(0.05, 0.95, 0.05)
        for threshold in thresholds:
            pred = (class_probs > threshold).float()

            tp = (pred * class_targets).sum().item()
            fp = (pred * (1 - class_targets)).sum().item()
            fn = ((1 - pred) * class_targets).sum().item()

            if tp + fp > 0 and tp + fn > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }

        optimal_thresholds.append(best_threshold)
        threshold_metrics[class_names[class_idx]] = best_metrics

        # Display results
        metrics = best_metrics
        print(
            f"   {class_names[class_idx]:<15} {best_threshold:<8.2f} {metrics.get('f1', 0):<8.4f} "
            f"{metrics.get('precision', 0):<10.4f} {metrics.get('recall', 0):<8.4f} {pos_samples:<8.0f}"
        )

    print(f"\n✅ Threshold optimization complete!")
    print(f"   Average optimal threshold: {np.mean(optimal_thresholds):.3f}")
    print(
        f"   Threshold range: {min(optimal_thresholds):.3f} - {max(optimal_thresholds):.3f}"
    )

    return optimal_thresholds, threshold_metrics


# ============================================================================
# EVALUATION WITH OPTIMAL THRESHOLDS
# ============================================================================


def evaluate_with_optimal_thresholds(
    model: nn.Module,
    test_loader,
    device: torch.device,
    optimal_thresholds: List[float],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model using optimal per-class thresholds

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        optimal_thresholds: List of optimal thresholds per class
        class_names: List of class names

    Returns:
        results: Dictionary with comprehensive evaluation results
    """
    print("📊 Evaluating with optimal thresholds...")

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(optimal_thresholds))]

    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]  # Use gate logits
            probs = torch.sigmoid(gate_logits)

            # Calculate loss
            loss = criterion(gate_logits, targets.float())
            total_loss += loss.item()
            num_batches += 1

            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Apply optimal thresholds per class
    pred_binary = torch.zeros_like(all_probs)
    for class_idx, threshold in enumerate(optimal_thresholds):
        pred_binary[:, class_idx] = (all_probs[:, class_idx] > threshold).float()

    # Calculate overall metrics
    subset_acc = (pred_binary == all_targets).all(dim=1).float().mean().item()
    hamming_acc = (pred_binary == all_targets).float().mean().item()

    # Per-class metrics
    per_class_metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []

    for class_idx, class_name in enumerate(class_names):
        pred_class = pred_binary[:, class_idx]
        target_class = all_targets[:, class_idx]

        tp = (pred_class * target_class).sum().item()
        fp = (pred_class * (1 - target_class)).sum().item()
        fn = ((1 - pred_class) * target_class).sum().item()
        tn = ((1 - pred_class) * (1 - target_class)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class_metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "threshold": optimal_thresholds[class_idx],
        }

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Compile results
    results = {
        "overall": {
            "subset_accuracy": subset_acc,
            "hamming_accuracy": hamming_acc,
            "precision": np.mean(precision_list),
            "recall": np.mean(recall_list),
            "f1_score": np.mean(f1_list),
            "average_loss": total_loss / num_batches,
        },
        "per_class": per_class_metrics,
        "class_names": class_names,
        "optimal_thresholds": optimal_thresholds,
    }

    return results


# ============================================================================
# MULTI-LABEL ACCURACY CALCULATION
# ============================================================================


def calculate_multilabel_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: Union[float, List[float], torch.Tensor] = 0.5,
) -> Tuple[float, float, float, float, float]:
    """
    Compute multi-label accuracy metrics

    Args:
        predictions: Model predictions [B, num_classes]
        targets: Multi-hot ground truth [B, num_classes]
        threshold: Threshold for binary predictions (can be single value or per-class)

    Returns:
        subset_acc: Exact match accuracy (all labels correct)
        hamming_acc: Label-wise accuracy
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    with torch.no_grad():
        # Convert logits to probabilities
        probs = torch.sigmoid(predictions)

        # Handle per-class thresholds
        if isinstance(threshold, (list, tuple, torch.Tensor)):
            # Per-class thresholds
            pred_binary = torch.zeros_like(probs)
            for i, t in enumerate(threshold):
                pred_binary[:, i] = (probs[:, i] > t).float()
        else:
            # Single threshold for all classes
            pred_binary = (probs > threshold).float()

        # Subset accuracy (exact match)
        subset_acc = (pred_binary == targets).all(dim=1).float().mean()

        # Hamming accuracy (label-wise accuracy)
        hamming_acc = (pred_binary == targets).float().mean()

        # Precision, Recall, F1
        tp = (pred_binary * targets).sum(dim=0)
        fp = (pred_binary * (1 - targets)).sum(dim=0)
        fn = ((1 - pred_binary) * targets).sum(dim=0)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Average across classes
        precision = precision.mean()
        recall = recall.mean()
        f1 = f1.mean()

    return subset_acc, hamming_acc, precision, recall, f1


# ============================================================================
# ATTENTION DIVERSITY LOSS
# ============================================================================

def calculate_multilabel_attention_diversity(
    cams_list: List[torch.Tensor], targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Multi-label attention diversity loss

    Encourages different experts to focus on different spatial regions
    for all positive classes in multi-label setting.

    Args:
        cams_list: List of CAMs from 3 experts [B, num_classes, H, W]
        targets: Multi-hot labels [B, num_classes]
        eps: Small value for numerical stability

    Returns:
        diversity_loss: Scalar tensor representing attention diversity loss
    """
    if targets is None or targets.sum() == 0:
        return torch.tensor(0.0, device=cams_list[0].device)

    cos = nn.CosineSimilarity(dim=1, eps=eps)
    diversity_loss = 0.0
    total_pairs = 0
    batch_size = targets.size(0)

    for batch_idx in range(batch_size):
        # Get positive class indices for this sample
        positive_classes = torch.where(targets[batch_idx] == 1)[0]

        if len(positive_classes) == 0:
            continue

        # Process each positive class
        for class_idx in positive_classes:
            # Extract CAMs for this class from all experts
            expert_cams = torch.stack(
                [
                    cams_list[0][batch_idx, class_idx],  # Expert 1: [H, W]
                    cams_list[1][batch_idx, class_idx],  # Expert 2: [H, W]
                    cams_list[2][batch_idx, class_idx],  # Expert 3: [H, W]
                ]
            )  # [3, H, W]

            # Flatten spatial dimensions and normalize
            expert_cams = expert_cams.view(3, -1)  # [3, H*W]
            expert_cams = F.normalize(expert_cams, p=2, dim=-1)

            # Remove mean activation to focus on relative attention patterns
            mean = expert_cams.mean(dim=-1, keepdim=True)  # [3, 1]
            expert_cams = F.relu(expert_cams - mean)

            # Compute pairwise cosine similarity (encourage orthogonality)
            for i in range(3):
                for j in range(i + 1, 3):
                    similarity = cos(
                        expert_cams[i : i + 1], expert_cams[j : j + 1]
                    ).mean()
                    diversity_loss += similarity
                    total_pairs += 1

    # Average over all pairs
    if total_pairs > 0:
        return diversity_loss / total_pairs
    else:
        return torch.tensor(0.0, device=cams_list[0].device)


# ============================================================================
# RESULTS PRINTING
# ============================================================================


def print_enhanced_results(results: Dict[str, Any]):
    """Print comprehensive results with optimal thresholds"""
    print("\n" + "=" * 80)
    print("🎯 ENHANCED MEDAF EVALUATION RESULTS (Optimal Thresholds)")
    print("=" * 80)

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

    # Per-class performance with thresholds
    print(f"\n🏷️  Per-Class Performance (Optimal Thresholds):")
    print(
        f"   {'Class':<15} {'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
    )
    print("   " + "-" * 65)

    per_class = results["per_class"]
    for class_name, metrics in per_class.items():
        print(
            f"   {class_name:<15} {metrics['threshold']:<10.3f} {metrics['precision']:<10.4f} "
            f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}"
        )

    # Best and worst classes
    f1_scores = [metrics["f1"] for metrics in per_class.values()]
    class_names = list(per_class.keys())

    if f1_scores:
        best_idx = f1_scores.index(max(f1_scores))
        worst_idx = f1_scores.index(min(f1_scores))

        print(
            f"\n🏆 Best Class:  {class_names[best_idx]} (F1={f1_scores[best_idx]:.4f})"
        )
        print(
            f"📉 Worst Class: {class_names[worst_idx]} (F1={f1_scores[worst_idx]:.4f})"
        )

    # Improvement assessment
    print(f"\n💡 Model Assessment:")
    if overall["f1_score"] < 0.2:
        print("   ⚠️  Model still needs improvement (F1 < 0.2)")
        print(
            "   📈 Consider: longer training, better data augmentation, loss weight tuning"
        )
    elif overall["f1_score"] < 0.4:
        print("   🔶 Model shows good improvement with optimal thresholds")
        print("   📈 Consider: expert configuration tuning, advanced loss functions")
    else:
        print("   ✅ Model shows strong performance with optimal thresholds!")
        print("   🎯 Consider: fine-tuning for production deployment")


# ============================================================================
# TRAINING CONFIGURATION DISPLAY
# ============================================================================


def print_training_config(config: Dict[str, Any]):
    """Print training configuration in a formatted way"""
    print("🚀 ENHANCED MEDAF TRAINING")
    print("=" * 60)
    print("Key improvements:")
    print("✅ Class-weighted BCE loss for imbalanced data")
    print("✅ Per-class threshold optimization")
    print("✅ Enhanced evaluation with medical metrics")
    print("✅ Improved training configuration")
    print("=" * 60)

    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"   Batch Size: {config.get('batch_size', 'N/A')}")
    print(f"   Class Weighting: {config.get('use_class_weights', False)}")
    print(f"   Optimal Thresholds: {config.get('use_optimal_thresholds', False)}")
    print(f"   Enhanced Training: {config.get('enhanced_training', False)}")


def confirm_training_start() -> bool:
    """Ask user for training confirmation"""
    response = input(f"\n🤔 Start enhanced training? (y/n): ").lower().strip()
    if response != "y":
        print("Training cancelled.")
        return False
    return True


# ============================================================================
# STANDARD TRAINING FUNCTIONS
# ============================================================================


def train_multilabel_standard(
    train_loader, model, criterion, optimizer, args, device=None
):
    """
    Standard training loop for multi-label MEDAF

    Args:
        train_loader: DataLoader with multi-label data
        model: MultiLabelMEDAF model
        criterion: Dictionary containing loss functions
        optimizer: Optimizer
        args: Training arguments
        device: Device to run on

    Returns:
        Average training loss
    """
    model.train()

    loss_keys = args["loss_keys"]  # ["b1", "b2", "b3", "gate", "divAttn", "total"]
    acc_keys = args["acc_keys"]  # ["acc1", "acc2", "acc3", "accGate"]

    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)  # Multi-hot labels [B, num_classes]

        # Forward pass
        output_dict = model(inputs, targets)
        logits = output_dict["logits"]  # List of logits from 4 heads
        cams_list = output_dict["cams_list"]  # CAMs from 3 experts

        # Multi-label classification losses for expert branches
        bce_losses = [
            criterion["bce"](logit.float(), targets.float())
            for logit in logits[:3]  # Expert branches only
        ]

        # Gating loss (on fused predictions)
        gate_loss = criterion["bce"](logits[3].float(), targets.float())

        # Multi-label attention diversity loss
        diversity_loss = calculate_multilabel_attention_diversity(cams_list, targets)

        # Combine losses according to weights
        loss_values = bce_losses + [gate_loss, diversity_loss]
        total_loss = (
            args["loss_wgts"][0] * sum(bce_losses)  # Expert loss weight
            + args["loss_wgts"][1] * gate_loss  # Gating loss weight
            + args["loss_wgts"][2] * diversity_loss  # Diversity loss weight
        )
        loss_values.append(total_loss)

        # Compute multi-label accuracies
        acc_values = []
        for logit in logits:
            subset_acc, hamming_acc, _, _, _ = calculate_multilabel_accuracy(
                logit, targets, threshold=0.5
            )
            acc_values.append(subset_acc * 100)  # Convert to percentage

        # Update meters
        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        train_accs = {acc_keys[k]: acc_values[k] for k in range(len(acc_keys))}

        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print progress
        if i % 50 == 0:  # Print every 50 batches
            tmp_str = f"\nBatch [{i}/{len(train_loader)}] "
            tmp_str += "< Training Loss >\n"
            for k, v in loss_meter.items():
                tmp_str += f"{k}:{v.value:.4f} "
            tmp_str += "\n< Training Accuracy >\n"
            for k, v in acc_meter.items():
                tmp_str += f"{k}:{v.value:.1f} "
            print(tmp_str)

    time_elapsed = time.time() - time_start
    print(f"\nEpoch completed in {time_elapsed:.1f}s")

    return loss_meter[loss_keys[-1]].value


def train_multilabel_enhanced_with_metrics(
    train_loader, model, criterion, optimizer, args, device=None
):
    """
    Enhanced training loop for multi-label MEDAF with detailed metrics return

    This is a wrapper around train_multilabel_enhanced that returns detailed metrics
    for better integration with training frameworks.

    Args:
        train_loader: DataLoader with multi-label data
        model: MultiLabelMEDAF model
        criterion: Dictionary containing loss functions (should include weighted BCE)
        optimizer: Optimizer
        args: Training arguments
        device: Device to run on

    Returns:
        Dictionary with detailed training metrics
    """
    model.train()

    loss_keys = args["loss_keys"]  # ["b1", "b2", "b3", "gate", "divAttn", "total"]
    acc_keys = args["acc_keys"]  # ["acc1", "acc2", "acc3", "accGate"]

    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)  # Multi-hot labels [B, num_classes]

        # Forward pass
        output_dict = model(inputs, targets)
        logits = output_dict["logits"]  # List of logits from 4 heads
        cams_list = output_dict["cams_list"]  # CAMs from 3 experts

        # Multi-label classification losses for expert branches (with class weighting)
        bce_losses = [
            criterion["bce"](logit.float(), targets.float())
            for logit in logits[:3]  # Expert branches only
        ]

        # Gating loss (on fused predictions) - with class weighting
        gate_loss = criterion["bce"](logits[3].float(), targets.float())

        # Multi-label attention diversity loss
        diversity_loss = calculate_multilabel_attention_diversity(cams_list, targets)

        # Combine losses according to weights
        loss_values = bce_losses + [gate_loss, diversity_loss]
        total_loss = (
            args["loss_wgts"][0] * sum(bce_losses)  # Expert loss weight
            + args["loss_wgts"][1] * gate_loss  # Gating loss weight
            + args["loss_wgts"][2] * diversity_loss  # Diversity loss weight
        )
        loss_values.append(total_loss)

        # Compute multi-label accuracies (still using 0.5 threshold for training monitoring)
        acc_values = []
        for logit in logits:
            subset_acc, hamming_acc, _, _, _ = calculate_multilabel_accuracy(
                logit, targets, threshold=0.5
            )
            acc_values.append(subset_acc * 100)  # Convert to percentage

        # Update meters
        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        train_accs = {acc_keys[k]: acc_values[k] for k in range(len(acc_keys))}

        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print progress with class weight info
        if i % 50 == 0:  # Print every 50 batches
            tmp_str = f"\nBatch [{i}/{len(train_loader)}] "
            tmp_str += "< Training Loss (Class-Weighted) >\n"
            for k, v in loss_meter.items():
                tmp_str += f"{k}:{v.value:.4f} "
            tmp_str += "\n< Training Accuracy >\n"
            for k, v in acc_meter.items():
                tmp_str += f"{k}:{v.value:.1f} "
            print(tmp_str)

    time_elapsed = time.time() - time_start
    print(f"\nEpoch completed in {time_elapsed:.1f}s")

    # Final epoch summary
    tmp_str = "< Final Training Loss (Class-Weighted) >\n"
    for k, v in loss_meter.items():
        tmp_str += f"{k}:{v.value:.4f} "
    tmp_str += "\n< Final Training Accuracy >\n"
    for k, v in acc_meter.items():
        tmp_str += f"{k}:{v.value:.1f} "
    print(tmp_str)

    # Return detailed metrics for better integration
    return {
        "total_loss": loss_meter[loss_keys[-1]].value,
        "expert_loss": (
            loss_meter[loss_keys[0]].value
            + loss_meter[loss_keys[1]].value
            + loss_meter[loss_keys[2]].value
        )
        / 3,
        "gate_loss": loss_meter[loss_keys[3]].value,
        "diversity_loss": loss_meter[loss_keys[4]].value,
        "accuracy": acc_meter[acc_keys[-1]].value,  # Gate accuracy
    }


def train_multilabel_enhanced(
    train_loader, model, criterion, optimizer, args, device=None
):
    """
    Enhanced training loop for multi-label MEDAF with class weighting

    Args:
        train_loader: DataLoader with multi-label data
        model: MultiLabelMEDAF model
        criterion: Dictionary containing loss functions (should include weighted BCE)
        optimizer: Optimizer
        args: Training arguments
        device: Device to run on

    Returns:
        Average training loss
    """
    model.train()

    loss_keys = args["loss_keys"]  # ["b1", "b2", "b3", "gate", "divAttn", "total"]
    acc_keys = args["acc_keys"]  # ["acc1", "acc2", "acc3", "accGate"]

    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()

    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)  # Multi-hot labels [B, num_classes]

        # Forward pass
        output_dict = model(inputs, targets)
        logits = output_dict["logits"]  # List of logits from 4 heads
        cams_list = output_dict["cams_list"]  # CAMs from 3 experts

        # Multi-label classification losses for expert branches (with class weighting)
        bce_losses = [
            criterion["bce"](logit.float(), targets.float())
            for logit in logits[:3]  # Expert branches only
        ]

        # Gating loss (on fused predictions) - with class weighting
        gate_loss = criterion["bce"](logits[3].float(), targets.float())

        # Multi-label attention diversity loss
        diversity_loss = calculate_multilabel_attention_diversity(cams_list, targets)

        # Combine losses according to weights
        loss_values = bce_losses + [gate_loss, diversity_loss]
        total_loss = (
            args["loss_wgts"][0] * sum(bce_losses)  # Expert loss weight
            + args["loss_wgts"][1] * gate_loss  # Gating loss weight
            + args["loss_wgts"][2] * diversity_loss  # Diversity loss weight
        )
        loss_values.append(total_loss)

        # Compute multi-label accuracies (still using 0.5 threshold for training monitoring)
        acc_values = []
        for logit in logits:
            subset_acc, hamming_acc, _, _, _ = calculate_multilabel_accuracy(
                logit, targets, threshold=0.5
            )
            acc_values.append(subset_acc * 100)  # Convert to percentage

        # Update meters
        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        train_accs = {acc_keys[k]: acc_values[k] for k in range(len(acc_keys))}

        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print progress with class weight info
        if i % 50 == 0:  # Print every 50 batches
            tmp_str = f"\nBatch [{i}/{len(train_loader)}] "
            tmp_str += "< Training Loss (Class-Weighted) >\n"
            for k, v in loss_meter.items():
                tmp_str += f"{k}:{v.value:.4f} "
            tmp_str += "\n< Training Accuracy >\n"
            for k, v in acc_meter.items():
                tmp_str += f"{k}:{v.value:.1f} "
            print(tmp_str)

    time_elapsed = time.time() - time_start
    print(f"\nEpoch completed in {time_elapsed:.1f}s")

    # Final epoch summary
    tmp_str = "< Final Training Loss (Class-Weighted) >\n"
    for k, v in loss_meter.items():
        tmp_str += f"{k}:{v.value:.4f} "
    tmp_str += "\n< Final Training Accuracy >\n"
    for k, v in acc_meter.items():
        tmp_str += f"{k}:{v.value:.1f} "
    print(tmp_str)

    return loss_meter[loss_keys[-1]].value

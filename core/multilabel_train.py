import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.util import *


def multiLabelAttnDiv(cams_list, targets, eps=1e-6):
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


def multiLabelAccuracy(predictions, targets, threshold=0.5):
    """
    Compute multi-label accuracy metrics

    Args:
        predictions: Model predictions [B, num_classes]
        targets: Multi-hot ground truth [B, num_classes]
        threshold: Threshold for binary predictions

    Returns:
        subset_acc: Exact match accuracy (all labels correct)
        hamming_acc: Label-wise accuracy
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    with torch.no_grad():
        # Convert logits to probabilities and then to binary predictions
        probs = torch.sigmoid(predictions)
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


def train_multilabel(train_loader, model, criterion, optimizer, args, device=None):
    """
    Training loop for multi-label MEDAF

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
        diversity_loss = multiLabelAttnDiv(cams_list, targets)

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
            subset_acc, hamming_acc, _, _, _ = multiLabelAccuracy(logit, targets)
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
            tmp_str = f"Batch [{i}/{len(train_loader)}] "
            tmp_str += "< Training Loss >\n"
            for k, v in loss_meter.items():
                tmp_str += f"{k}:{v.value:.4f} "
            tmp_str += "\n< Training Accuracy >\n"
            for k, v in acc_meter.items():
                tmp_str += f"{k}:{v.value:.1f} "
            print(tmp_str)

    time_elapsed = time.time() - time_start
    print(f"\nEpoch completed in {time_elapsed:.1f}s")

    # Final epoch summary
    tmp_str = "< Final Training Loss >\n"
    for k, v in loss_meter.items():
        tmp_str += f"{k}:{v.value:.4f} "
    tmp_str += "\n< Final Training Accuracy >\n"
    for k, v in acc_meter.items():
        tmp_str += f"{k}:{v.value:.1f} "
    print(tmp_str)

    return loss_meter[loss_keys[-1]].value

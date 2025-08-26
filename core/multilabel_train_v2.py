import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from misc.util import *
from .multilabel_train import multiLabelAttnDiv, multiLabelAccuracy


def enhancedMultiLabelAttnDiv(cams_list, targets, gate_weights=None, 
                             diversity_type="cosine", eps=1e-6):
    """
    Enhanced multi-label attention diversity loss with per-class gating awareness
    
    Args:
        cams_list: List of CAMs from 3 experts [B, num_classes, H, W]
        targets: Multi-hot labels [B, num_classes]
        gate_weights: Per-class gating weights [B, num_classes, 3] (optional)
        diversity_type: Type of diversity measure ('cosine', 'l2', 'kl')
        eps: Small value for numerical stability
        
    Returns:
        diversity_loss: Enhanced diversity loss considering gating weights
    """
    if targets is None or targets.sum() == 0:
        return torch.tensor(0.0, device=cams_list[0].device)
    
    diversity_loss = 0.0
    total_pairs = 0
    batch_size = targets.size(0)
    
    for batch_idx in range(batch_size):
        positive_classes = torch.where(targets[batch_idx] == 1)[0]
        
        if len(positive_classes) == 0:
            continue
            
        for class_idx in positive_classes:
            # Extract CAMs for this class
            expert_cams = torch.stack([
                cams_list[0][batch_idx, class_idx],
                cams_list[1][batch_idx, class_idx], 
                cams_list[2][batch_idx, class_idx]
            ])  # [3, H, W]
            
            # Flatten and normalize
            expert_cams = expert_cams.view(3, -1)
            expert_cams = F.normalize(expert_cams, p=2, dim=-1)
            
            # Remove mean activation
            mean = expert_cams.mean(dim=-1, keepdim=True)
            expert_cams = F.relu(expert_cams - mean)
            
            # Weight by gating importance if available
            if gate_weights is not None:
                class_gate_weights = gate_weights[batch_idx, class_idx]  # [3]
                # Scale CAMs by their gating weights (more important experts should be more diverse)
                expert_cams = expert_cams * class_gate_weights.unsqueeze(-1)
            
            # Compute diversity based on type
            if diversity_type == "cosine":
                cos = nn.CosineSimilarity(dim=1, eps=eps)
                for i in range(3):
                    for j in range(i + 1, 3):
                        similarity = cos(expert_cams[i:i+1], expert_cams[j:j+1]).mean()
                        diversity_loss += similarity
                        total_pairs += 1
                        
            elif diversity_type == "l2":
                for i in range(3):
                    for j in range(i + 1, 3):
                        l2_dist = torch.norm(expert_cams[i] - expert_cams[j], p=2)
                        diversity_loss -= l2_dist  # Negative because we want to maximize distance
                        total_pairs += 1
                        
            elif diversity_type == "kl":
                # KL divergence between attention distributions
                expert_probs = F.softmax(expert_cams, dim=-1)
                for i in range(3):
                    for j in range(i + 1, 3):
                        kl_div = F.kl_div(
                            expert_probs[i:i+1].log(), 
                            expert_probs[j:j+1], 
                            reduction='batchmean'
                        )
                        diversity_loss -= kl_div  # Negative to encourage diversity
                        total_pairs += 1
    
    return diversity_loss / max(total_pairs, 1)


class ComparativeTrainingFramework:
    """
    Framework for comparing global vs per-class gating performance
    """
    
    def __init__(self, args):
        self.args = args
        self.results = defaultdict(list)
        self.current_epoch = 0
        
    def log_metrics(self, model_type, epoch, metrics):
        """Log metrics for comparison"""
        metrics_with_meta = {
            'epoch': epoch,
            'model_type': model_type,
            **metrics
        }
        self.results[model_type].append(metrics_with_meta)
    
    def get_comparison_summary(self):
        """Get summary comparing both approaches"""
        summary = {}
        
        for model_type, results in self.results.items():
            if not results:
                continue
                
            latest = results[-1]
            summary[model_type] = {
                'final_epoch': latest['epoch'],
                'final_subset_acc': latest.get('subset_acc', 0),
                'final_hamming_acc': latest.get('hamming_acc', 0),
                'final_f1': latest.get('f1', 0),
                'final_loss': latest.get('total_loss', 0),
                'avg_diversity_loss': np.mean([r.get('diversity_loss', 0) for r in results[-10:]])
            }
        
        return summary
    
    def print_comparison(self):
        """Print detailed comparison"""
        summary = self.get_comparison_summary()
        
        print("\n" + "="*60)
        print("COMPARATIVE ANALYSIS: Global vs Per-Class Gating")
        print("="*60)
        
        if 'global' in summary and 'per_class' in summary:
            global_results = summary['global']
            pc_results = summary['per_class']
            
            print(f"{'Metric':<20} {'Global':<15} {'Per-Class':<15} {'Improvement':<15}")
            print("-" * 65)
            
            metrics = ['final_subset_acc', 'final_hamming_acc', 'final_f1']
            for metric in metrics:
                global_val = global_results.get(metric, 0)
                pc_val = pc_results.get(metric, 0)
                improvement = ((pc_val - global_val) / max(global_val, 1e-8)) * 100
                
                print(f"{metric.replace('final_', ''):<20} {global_val:<15.4f} {pc_val:<15.4f} {improvement:+.2f}%")
            
            # Loss comparison (lower is better)
            global_loss = global_results.get('final_loss', float('inf'))
            pc_loss = pc_results.get('final_loss', float('inf'))
            loss_improvement = ((global_loss - pc_loss) / max(global_loss, 1e-8)) * 100
            print(f"{'loss_reduction':<20} {global_loss:<15.4f} {pc_loss:<15.4f} {loss_improvement:+.2f}%")
            
        print("="*60)


def train_multilabel_v2(train_loader, model, criterion, optimizer, args, 
                       device=None, comparative_framework=None):
    """
    Enhanced training loop for MultiLabelMEDAFv2 with comparative analysis
    
    Args:
        train_loader: DataLoader with multi-label data
        model: MultiLabelMEDAFv2 model
        criterion: Dictionary containing loss functions
        optimizer: Optimizer
        args: Training arguments
        device: Device to run on
        comparative_framework: Optional framework for comparing approaches
        
    Returns:
        Training metrics dictionary
    """
    model.train()

    loss_keys = args["loss_keys"]
    acc_keys = args["acc_keys"]
    
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter = {p: AverageMeter() for p in acc_keys}
    
    # Additional metrics for enhanced analysis
    diversity_meter = AverageMeter()
    gating_entropy_meter = AverageMeter()
    
    time_start = time.time()
    
    # Get model configuration
    gating_summary = model.get_gating_summary()
    gating_type = gating_summary["gating_type"]
    
    print(f"\nTraining with {gating_type} gating...")

    for i, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)

        # Forward pass with enhanced outputs
        output_dict = model(inputs, targets, return_attention_weights=True)
        logits = output_dict["logits"]
        cams_list = output_dict["cams_list"]
        gate_pred = output_dict["gate_pred"]
        
        # Get per-class weights if available
        per_class_weights = output_dict.get("per_class_weights", None)
        
        # Classification losses for expert branches
        bce_losses = [
            criterion["bce"](logit.float(), targets.float()) 
            for logit in logits[:3]
        ]
        
        # Gating loss
        gate_loss = criterion["bce"](logits[3].float(), targets.float())
        
        # Enhanced diversity loss
        if args.get("enhanced_diversity", False) and per_class_weights is not None:
            diversity_loss = enhancedMultiLabelAttnDiv(
                cams_list, targets, per_class_weights,
                diversity_type=args.get("diversity_type", "cosine")
            )
        else:
            # Standard diversity loss
            diversity_loss = multiLabelAttnDiv(cams_list, targets)
        
        # Per-class gating regularization (encourage specialization)
        gating_reg_loss = 0.0
        if per_class_weights is not None and args.get("gating_regularization", 0) > 0:
            # Encourage different classes to use different expert combinations
            # Compute entropy of gating weights per class
            gate_entropy = -(per_class_weights * torch.log(per_class_weights + 1e-8)).sum(dim=-1)
            gating_reg_loss = -gate_entropy.mean()  # Negative to encourage high entropy (diversity)
        
        # Total loss
        loss_values = bce_losses + [gate_loss, diversity_loss]
        total_loss = (
            args["loss_wgts"][0] * sum(bce_losses) +
            args["loss_wgts"][1] * gate_loss +
            args["loss_wgts"][2] * diversity_loss +
            args.get("gating_regularization", 0) * gating_reg_loss
        )
        loss_values.append(total_loss)

        # Compute accuracies
        acc_values = []
        for logit in logits:
            subset_acc, hamming_acc, precision, recall, f1 = multiLabelAccuracy(logit, targets)
            acc_values.append(subset_acc * 100)
        
        # Update meters
        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        train_accs = {acc_keys[k]: acc_values[k] for k in range(len(acc_keys))}
        
        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))
        
        # Update additional meters
        diversity_meter.update(diversity_loss.item(), inputs.size(0))
        if per_class_weights is not None:
            gate_entropy = -(per_class_weights * torch.log(per_class_weights + 1e-8)).sum(dim=-1).mean()
            gating_entropy_meter.update(gate_entropy.item(), inputs.size(0))

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Progress logging
        if i % 50 == 0:
            tmp_str = f"Batch [{i}/{len(train_loader)}] ({gating_type}) "
            tmp_str += f"Loss: {total_loss.item():.4f}, "
            tmp_str += f"Div: {diversity_loss.item():.6f}, "
            if per_class_weights is not None:
                tmp_str += f"GateEnt: {gating_entropy_meter.value:.3f}, "
            tmp_str += f"SubsetAcc: {acc_values[3]:.2f}%"
            print(tmp_str)

    time_elapsed = time.time() - time_start
    
    # Compile final metrics
    final_metrics = {
        'total_loss': loss_meter[loss_keys[-1]].value,
        'diversity_loss': diversity_meter.value,
        'subset_acc': acc_meter[acc_keys[-1]].value,
        'hamming_acc': acc_meter[acc_keys[-1]].value,  # Using gate accuracy as proxy
        'gating_type': gating_type,
        'training_time': time_elapsed
    }
    
    if per_class_weights is not None:
        final_metrics['gating_entropy'] = gating_entropy_meter.value
    
    # Log to comparative framework
    if comparative_framework is not None:
        comparative_framework.log_metrics(gating_type, comparative_framework.current_epoch, final_metrics)
    
    # Print summary
    print(f"\nEpoch Summary ({gating_type} gating):")
    print(f"  Total Loss: {final_metrics['total_loss']:.4f}")
    print(f"  Diversity Loss: {final_metrics['diversity_loss']:.6f}")
    print(f"  Subset Accuracy: {final_metrics['subset_acc']:.2f}%")
    if 'gating_entropy' in final_metrics:
        print(f"  Gating Entropy: {final_metrics['gating_entropy']:.3f}")
    print(f"  Training Time: {time_elapsed:.1f}s")
    
    return final_metrics


def evaluate_multilabel_v2(model, test_loader, criterion, args, device=None):
    """
    Enhanced evaluation for MultiLabelMEDAFv2
    
    Returns detailed metrics including per-class gating analysis
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_gate_weights = []
    
    eval_loss = 0.0
    eval_diversity = 0.0
    num_batches = 0
    
    gating_type = model.get_gating_summary()["gating_type"]
    
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data[0].to(device), data[1].to(device)
            
            # Forward pass
            outputs = model(inputs, targets, return_attention_weights=True)
            logits = outputs["logits"]
            cams_list = outputs["cams_list"]
            
            # Use fused logits for evaluation
            fused_logits = logits[3]
            
            # Compute losses
            bce_loss = criterion["bce"](fused_logits, targets)
            diversity_loss = multiLabelAttnDiv(cams_list, targets)
            
            eval_loss += bce_loss.item()
            eval_diversity += diversity_loss.item()
            num_batches += 1
            
            # Store predictions and targets
            all_predictions.append(torch.sigmoid(fused_logits))
            all_targets.append(targets)
            
            # Store gating weights if per-class
            if "per_class_weights" in outputs:
                all_gate_weights.append(outputs["per_class_weights"])
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute comprehensive metrics
    subset_acc, hamming_acc, precision, recall, f1 = multiLabelAccuracy(
        all_predictions, all_targets, threshold=0.5
    )
    
    eval_metrics = {
        'eval_loss': eval_loss / num_batches,
        'eval_diversity': eval_diversity / num_batches,
        'subset_accuracy': subset_acc.item(),
        'hamming_accuracy': hamming_acc.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1.item(),
        'gating_type': gating_type
    }
    
    # Per-class gating analysis
    if all_gate_weights:
        all_gate_weights = torch.cat(all_gate_weights, dim=0)  # [N, num_classes, 3]
        
        # Analyze expert specialization
        expert_preferences = all_gate_weights.mean(dim=0)  # [num_classes, 3]
        expert_entropy = -(expert_preferences * torch.log(expert_preferences + 1e-8)).sum(dim=-1)
        
        eval_metrics.update({
            'avg_gating_entropy': expert_entropy.mean().item(),
            'expert_specialization': expert_preferences.std(dim=0).mean().item(),
            'max_expert_preference': expert_preferences.max().item(),
            'min_expert_preference': expert_preferences.min().item()
        })
    
    return eval_metrics

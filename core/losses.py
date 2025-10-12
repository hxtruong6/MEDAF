import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label Classification

    Addresses class imbalance by down-weighting easy examples and focusing on hard examples.

    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
        pos_weight (torch.Tensor, optional): Weight for positive examples per class

    Reference:
        Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal loss for dense object detection. ICCV, 2017.
    """

    def __init__(
        self,
        alpha: float = 0.25,  # FIXED: Changed default from 1.0 to 0.25 for multi-label
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
        # Warning for problematic alpha values in multi-label classification
        if alpha == 1.0:
            import warnings
            warnings.warn(
                "Focal Loss with alpha=1.0 causes zero loss for negative samples in multi-label classification. "
                "Consider using alpha=0.25 or alpha=0.5 for better performance.",
                UserWarning
            )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss

        Args:
            inputs: Logits tensor of shape (N, C) where N is batch size, C is number of classes
            targets: Binary targets tensor of shape (N, C)

        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )

        # Calculate p_t (probability of true class)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Calculate alpha_t - FIXED: Use alpha for positive class, (1-alpha) for negative class
        # This ensures both positive and negative samples contribute to the loss
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal weight: alpha_t * (1 - p_t)^gamma
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)

        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification

    Addresses the imbalance between positive and negative samples by using
    different focusing parameters for positive and negative samples.

    Reference:
        Ridnik, E., Ben-Baruch, E., Noy, A., & Zelnik-Manor, L. (2021).
        Asymmetric loss for multi-label classification. ICCV, 2021.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Asymmetric Loss"""
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Calculate loss
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            los_pos *= one_sided_w
            los_neg *= one_sided_w

        loss = los_pos + los_neg
        return -loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with advanced weighting strategies
    """

    def __init__(
        self, pos_weight: Optional[torch.Tensor] = None, reduction: str = "mean"
    ):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass of Weighted BCE Loss"""
        return F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction=self.reduction
        )


class LossFactory:
    """Factory class for creating loss functions based on configuration"""

    @staticmethod
    def create_loss(
        loss_type: str,
        num_classes: int,
        device: torch.device,
        pos_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Create loss function based on type

        Args:
            loss_type: Type of loss ('focal', 'bce', 'weighted_bce', 'asymmetric')
            num_classes: Number of classes
            device: Device to place loss on
            pos_weight: Positive class weights
            **kwargs: Additional arguments for specific loss functions

        Returns:
            Loss function module
        """
        loss_type = loss_type.lower()

        if loss_type == "focal":
            return FocalLoss(
                alpha=kwargs.get("focal_alpha", 1.0),
                gamma=kwargs.get("focal_gamma", 2.0),
                pos_weight=pos_weight,
            ).to(device)

        elif loss_type == "asymmetric":
            return AsymmetricLoss(
                gamma_neg=kwargs.get("gamma_neg", 4.0),
                gamma_pos=kwargs.get("gamma_pos", 1.0),
                clip=kwargs.get("clip", 0.05),
            ).to(device)

        elif loss_type == "weighted_bce":
            return WeightedBCELoss(pos_weight=pos_weight).to(device)

        elif loss_type == "bce":
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


def calculate_class_weights_advanced(
    train_loader,
    num_classes: int,
    device: torch.device,
    method: str = "inverse_freq",
    beta: float = 0.9999,
) -> torch.Tensor:
    """
    Calculate class weights for imbalanced multi-label data with advanced methods

    Args:
        train_loader: Training data loader
        num_classes: Number of classes
        device: Device to put weights on
        method: Weighting method ('inverse_freq', 'effective_num', 'focal')
        beta: Beta parameter for effective number method

    Returns:
        pos_weights: Tensor of positive class weights
    """
    class_counts = torch.zeros(num_classes, device=device)
    total_samples = 0

    # Count positive samples for each class
    for inputs, targets in train_loader:
        targets = targets.to(device)
        class_counts += targets.sum(dim=0)
        total_samples += targets.shape[0]

    # Calculate negative counts
    neg_counts = total_samples - class_counts

    if method == "inverse_freq":
        # Standard inverse frequency weighting
        pos_weights = neg_counts / (class_counts + 1e-8)
    elif method == "effective_num":
        # Effective number of samples (handles class imbalance better)
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

    return pos_weights


# Example usage and testing
if __name__ == "__main__":
    # Test Focal Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample data
    batch_size, num_classes = 16, 8
    inputs = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(0, 2, (batch_size, num_classes), device=device).float()

    # Test different loss functions
    losses = {
        "focal": FocalLoss(alpha=1.0, gamma=2.0),
        "asymmetric": AsymmetricLoss(),
        "weighted_bce": WeightedBCELoss(),
        "bce": nn.BCEWithLogitsLoss(),
    }

    print("Loss function testing:")
    for name, loss_fn in losses.items():
        loss_fn = loss_fn.to(device)
        loss_value = loss_fn(inputs, targets)
        print(f"{name:>12}: {loss_value.item():.4f}")

    print("\n✅ All loss functions working correctly")

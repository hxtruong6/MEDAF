"""
Multi-Label Novelty Detection for MEDAF

This module implements the hybrid scoring mechanism for detecting unknown/novel samples
in multi-label classification settings, as described in the MEDAF paper and guide.

Key Components:
1. Hybrid Score Computation: Combines logit-based and feature-based scores
2. Threshold Calibration: Uses validation data to set rejection thresholds
3. Multi-Label Adaptation: Handles different types of novelty (independent, mixed, combinatorial)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, roc_curve


class MultiLabelNoveltyDetector:
    """
    Novelty detection for multi-label MEDAF models.

    This class implements the hybrid scoring mechanism described in the MEDAF paper:
    - Logit-based score: Measures confidence in known label predictions
    - Feature-based score: Uses CAM diversity to detect distributional shifts
    - Hybrid score: Combines both for robust novelty detection

    The detector handles three types of multi-label novelty:
    1. Independent Novelty: Only unknown labels present
    2. Mixed Novelty: Unknown + known labels (most challenging)
    3. Combinatorial Novelty: Novel combinations of known labels
    """

    def __init__(self, gamma: float = 1.0, temperature: float = 1.0):
        """
        Initialize the novelty detector.

        Args:
            gamma: Weight for feature-based score in hybrid computation (default: 1.0)
            temperature: Temperature for logit-based energy computation (default: 1.0)
        """
        self.gamma = gamma
        self.temperature = temperature
        self.threshold = None  # Will be calibrated on validation data
        self.is_calibrated = False

    def compute_logit_score(
        self, logits: torch.Tensor, predicted_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logit-based novelty score using Joint Energy approach.

        This implements the adapted logit-based score for multi-label:
        S_lg(x) = -log(sum_k exp(l_g,k / T))

        Low energy (high score) indicates known samples with tight predictions.
        High energy (low score) indicates unknown samples with dispersed predictions.

        Args:
            logits: Fused logits from gating network [B, num_classes]
            predicted_labels: Binary predictions [B, num_classes] (where sigmoid > 0.5)

        Returns:
            Logit-based scores [B] - higher values indicate more "known-like" samples
        """
        # Apply temperature scaling to logits
        scaled_logits = logits / self.temperature

        # Compute joint energy: -log(sum_k exp(l_k))
        # This captures the "tightness" of predictions
        # Low energy = tight predictions (known samples)
        # High energy = dispersed predictions (unknown samples)
        joint_energy = -torch.logsumexp(scaled_logits, dim=1)

        # Convert to score (higher = more known-like)
        # We negate energy so that low energy becomes high score
        logit_scores = -joint_energy

        return logit_scores

    def compute_feature_score(
        self, cams_list: List[torch.Tensor], predicted_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature-based novelty score using CAM diversity.

        This implements the adapted feature-based score for multi-label:
        S_ft(x) = (1/|Y_hat|) * sum_{y in Y_hat} ||(1/N) * sum_i M_{i,y}||_2

        High CAM norms indicate compact, confident activations (known samples).
        Low CAM norms indicate dispersed, uncertain activations (unknown samples).

        Args:
            cams_list: List of CAM tensors from all experts [B, num_classes, H, W]
            predicted_labels: Binary predictions [B, num_classes] (where sigmoid > 0.5)

        Returns:
            Feature-based scores [B] - higher values indicate more "known-like" samples
        """
        batch_size, num_classes = predicted_labels.shape
        num_experts = len(cams_list)

        # Get predicted positive labels for each sample
        predicted_positive = predicted_labels.bool()  # [B, num_classes]

        feature_scores = []

        for b in range(batch_size):
            # Get positive labels for this sample
            positive_labels = torch.where(predicted_positive[b])[0]

            if len(positive_labels) == 0:
                # No positive predictions - likely unknown sample
                feature_scores.append(0.0)
                continue

            # Compute CAM norms for each positive label
            label_norms = []
            for label_idx in positive_labels:
                # Average CAMs across experts for this label
                avg_cam = torch.stack(
                    [cams_list[i][b, label_idx] for i in range(num_experts)]
                ).mean(dim=0)

                # Compute L2 norm of averaged CAM
                cam_norm = torch.norm(avg_cam, p=2)
                label_norms.append(cam_norm.item())

            # Average CAM norms across positive labels
            avg_cam_norm = np.mean(label_norms) if label_norms else 0.0
            feature_scores.append(avg_cam_norm)

        return torch.tensor(feature_scores, device=cams_list[0].device)

    def compute_hybrid_score(
        self,
        logits: torch.Tensor,
        cams_list: List[torch.Tensor],
        predicted_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid novelty score combining logit and feature information.

        This implements the core MEDAF novelty detection mechanism:
        S(x) = S_lg(x) + γ * S_ft(x)

        The hybrid score combines:
        - Logit-based score: Semantic confidence in known label predictions
        - Feature-based score: Representation compactness via CAM diversity

        Args:
            logits: Fused logits from gating network [B, num_classes]
            cams_list: List of CAM tensors from all experts [B, num_classes, H, W]
            predicted_labels: Binary predictions [B, num_classes] (where sigmoid > 0.5)

        Returns:
            Hybrid novelty scores [B] - higher values indicate more "known-like" samples
        """
        # Compute individual scores
        logit_scores = self.compute_logit_score(logits, predicted_labels)
        feature_scores = self.compute_feature_score(cams_list, predicted_labels)

        # Combine into hybrid score
        hybrid_scores = logit_scores + self.gamma * feature_scores

        return hybrid_scores

    def calibrate_threshold(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        fpr_target: float = 0.05,
    ) -> float:
        """
        Calibrate rejection threshold on validation data containing only known samples.

        This implements the threshold calibration process described in the guide:
        1. Compute hybrid scores for all validation samples (known only)
        2. Set threshold at target FPR (e.g., 5th percentile for 5% FPR)
        3. This ensures most known samples are accepted while unknowns are rejected

        Args:
            model: Trained MEDAF model
            val_loader: Validation data loader (known samples only)
            device: Device for computation
            fpr_target: Target false positive rate (default: 0.05 for 5% FPR)

        Returns:
            Calibrated threshold value
        """
        model.eval()
        all_scores = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                output_dict = model(inputs, targets)
                gate_logits = output_dict["logits"][-1]  # Use gating network
                cams_list = output_dict["cams_list"]  # Get CAMs from all experts

                # Convert to predictions
                probs = torch.sigmoid(gate_logits)
                predicted_labels = (probs > 0.5).float()

                # Compute hybrid scores
                scores = self.compute_hybrid_score(
                    gate_logits, cams_list, predicted_labels
                )
                all_scores.extend(scores.cpu().numpy())

        # Set threshold at target FPR percentile
        threshold = np.percentile(all_scores, fpr_target * 100)

        self.threshold = threshold
        self.is_calibrated = True

        print(
            f"Threshold calibrated: {threshold:.4f} (FPR target: {fpr_target*100:.1f}%)"
        )
        return threshold

    def detect_novelty(
        self,
        logits: torch.Tensor,
        cams_list: List[torch.Tensor],
        predicted_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect novel/unknown samples using hybrid scoring.

        This is the main novelty detection function that:
        1. Computes hybrid scores for input samples
        2. Compares scores to calibrated threshold
        3. Returns novelty predictions and confidence scores

        Args:
            logits: Fused logits from gating network [B, num_classes]
            cams_list: List of CAM tensors from all experts [B, num_classes, H, W]
            predicted_labels: Binary predictions [B, num_classes] (where sigmoid > 0.5)

        Returns:
            is_novel: Boolean tensor [B] - True for novel samples, False for known
            confidence_scores: Hybrid scores [B] - higher values indicate more known-like
        """
        if not self.is_calibrated:
            raise ValueError(
                "Detector not calibrated. Call calibrate_threshold() first."
            )

        # Compute hybrid scores
        confidence_scores = self.compute_hybrid_score(
            logits, cams_list, predicted_labels
        )

        # Apply threshold for novelty detection
        is_novel = confidence_scores < self.threshold

        return is_novel, confidence_scores

    def classify_novelty_type(
        self, predicted_labels: torch.Tensor, is_novel: torch.Tensor
    ) -> List[str]:
        """
        Classify the type of novelty detected in each sample.

        This implements the three types of multi-label novelty:
        1. Independent Novelty: Only unknown labels (no known labels predicted)
        2. Mixed Novelty: Unknown + known labels (some known labels predicted)
        3. Combinatorial Novelty: Novel combinations of known labels (advanced)

        Args:
            predicted_labels: Binary predictions [B, num_classes]
            is_novel: Boolean tensor [B] indicating novel samples

        Returns:
            List of novelty type strings for each sample
        """
        novelty_types = []

        for i, novel in enumerate(is_novel):
            if not novel:
                novelty_types.append("Known")
                continue

            # Count predicted positive labels
            num_positive = predicted_labels[i].sum().item()

            if num_positive == 0:
                # No labels predicted - likely independent novelty
                novelty_types.append("Independent Novelty")
            else:
                # Some labels predicted - mixed novelty
                novelty_types.append("Mixed Novelty")

        return novelty_types


def evaluate_novelty_detection(
    model: nn.Module,
    known_loader: torch.utils.data.DataLoader,
    unknown_loader: torch.utils.data.DataLoader,
    device: torch.device,
    detector: MultiLabelNoveltyDetector,
) -> Dict:
    """
    Evaluate novelty detection performance using AUROC and other metrics.

    This function implements the evaluation protocol described in the guide:
    1. Compute hybrid scores for known and unknown samples
    2. Calculate AUROC for novelty detection
    3. Report detection accuracy and other metrics

    Args:
        model: Trained MEDAF model
        known_loader: Data loader for known samples
        unknown_loader: Data loader for unknown samples
        device: Device for computation
        detector: Calibrated novelty detector

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    # Collect scores and labels
    all_scores = []
    all_labels = []  # 0 for known, 1 for unknown

    with torch.no_grad():
        # Process known samples
        for inputs, targets in known_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]
            cams_list = output_dict["cams_list"]

            probs = torch.sigmoid(gate_logits)
            predicted_labels = (probs > 0.5).float()

            scores = detector.compute_hybrid_score(
                gate_logits, cams_list, predicted_labels
            )
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend([0] * len(scores))  # 0 for known

        # Process unknown samples
        for inputs, targets in unknown_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output_dict = model(inputs, targets)
            gate_logits = output_dict["logits"][-1]
            cams_list = output_dict["cams_list"]

            probs = torch.sigmoid(gate_logits)
            predicted_labels = (probs > 0.5).float()

            scores = detector.compute_hybrid_score(
                gate_logits, cams_list, predicted_labels
            )
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend([1] * len(scores))  # 1 for unknown

    # Compute AUROC
    auroc = roc_auc_score(all_labels, all_scores)

    # Compute detection accuracy using calibrated threshold
    is_novel_pred = np.array(all_scores, dtype=np.float32) < detector.threshold
    detection_accuracy = np.mean(
        is_novel_pred == np.array(all_labels, dtype=np.float32)
    )

    # Compute precision and recall
    tp = np.sum((is_novel_pred == 1) & (np.array(all_labels, dtype=np.float32) == 1))
    fp = np.sum((is_novel_pred == 1) & (np.array(all_labels, dtype=np.float32) == 0))
    fn = np.sum((is_novel_pred == 0) & (np.array(all_labels, dtype=np.float32) == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "auroc": auroc,
        "detection_accuracy": detection_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold": detector.threshold,
        "num_known": len([l for l in all_labels if l == 0]),
        "num_unknown": len([l for l in all_labels if l == 1]),
    }

    return results


# Example usage function
def demonstrate_novelty_detection():
    """
    Demonstrate how to use the novelty detection system.

    This function shows the complete workflow:
    1. Create and calibrate the detector
    2. Detect novelty in test samples
    3. Evaluate performance
    """
    print("🔍 Multi-Label Novelty Detection Demo")
    print("=" * 50)

    # Example parameters
    gamma = 1.0  # Weight for feature-based score
    temperature = 1.0  # Temperature for logit-based energy

    # Create detector
    detector = MultiLabelNoveltyDetector(gamma=gamma, temperature=temperature)

    print(f"✅ Detector created with γ={gamma}, T={temperature}")
    print("📝 Next steps:")
    print("   1. Load your trained MEDAF model")
    print("   2. Prepare validation data (known samples only)")
    print("   3. Call detector.calibrate_threshold(model, val_loader, device)")
    print("   4. Use detector.detect_novelty() for test samples")
    print("   5. Evaluate with evaluate_novelty_detection()")

    return detector


if __name__ == "__main__":
    # Run demonstration
    detector = demonstrate_novelty_detection()

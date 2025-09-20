import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import build_backbone, conv1x1, Classifier


class MultiLabelMEDAF(nn.Module):
    """
    Multi-Label version of MEDAF (Multi-Expert Diverse Attention Fusion)

    Key changes from original MEDAF:
    1. Support for multi-hot label targets
    2. BCEWithLogitsLoss instead of CrossEntropyLoss
    3. Multi-label attention diversity computation
    4. Per-sample CAM extraction for multiple positive classes
    """

    def __init__(self, args=None):
        super(MultiLabelMEDAF, self).__init__()
        backbone, feature_dim, self.cam_size = build_backbone(
            img_size=args["img_size"],
            backbone_name=args["backbone"],
            projection_dim=-1,
            inchan=3,
        )
        self.img_size = args["img_size"]
        self.gate_temp = args["gate_temp"]
        self.num_classes = args[
            "num_classes"
        ]  # Changed from num_known to num_classes for clarity
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Shared layers (L1-L3)
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-6])

        # Expert branch 1
        self.branch1_l4 = nn.Sequential(*list(backbone.children())[-6:-3])
        self.branch1_l5 = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_classes)

        # Expert branch 2 (deep copy)
        self.branch2_l4 = copy.deepcopy(self.branch1_l4)
        self.branch2_l5 = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_classes)

        # Expert branch 3 (deep copy)
        self.branch3_l4 = copy.deepcopy(self.branch1_l4)
        self.branch3_l5 = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_classes)

        # Gating network
        self.gate_l3 = copy.deepcopy(self.shared_l3)
        self.gate_l4 = copy.deepcopy(self.branch1_l4)
        self.gate_l5 = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(
            Classifier(feature_dim, int(feature_dim / 4), bias=True),
            Classifier(int(feature_dim / 4), 3, bias=True),  # 3 experts
        )

    def forward(self, x, y=None, return_ft=False):
        """
        Forward pass for multi-label MEDAF

        Args:
            x: Input tensor [B, C, H, W]
            y: Multi-hot labels [B, num_classes] or None
            return_ft: Whether to return features

        Returns:
            Dictionary containing logits, gate predictions, and CAMs/features
        """
        b = x.size(0)
        ft_till_l3 = self.shared_l3(x)

        # Expert branch 1
        branch1_l4 = self.branch1_l4(ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_ft_cams = self.branch1_cls(branch1_l5)  # [B, num_classes, H, W]
        b1_logits = self.avg_pool(b1_ft_cams).view(b, -1)

        # Expert branch 2
        branch2_l4 = self.branch2_l4(ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_ft_cams = self.branch2_cls(branch2_l5)  # [B, num_classes, H, W]
        b2_logits = self.avg_pool(b2_ft_cams).view(b, -1)

        # Expert branch 3
        branch3_l4 = self.branch3_l4(ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_ft_cams = self.branch3_cls(branch3_l5)  # [B, num_classes, H, W]
        b3_logits = self.avg_pool(b3_ft_cams).view(b, -1)

        # Store CAMs for diversity loss computation
        cams_list = [b1_ft_cams, b2_ft_cams, b3_ft_cams]

        # Multi-label CAM extraction for positive classes
        if y is not None:
            # Extract CAMs for all positive classes across all experts
            # This will be used for attention diversity computation
            multi_label_cams = self._extract_multilabel_cams(cams_list, y)
        else:
            multi_label_cams = None

        if return_ft:
            # Aggregate features from all experts
            fts = (
                b1_ft_cams.detach().clone()
                + b2_ft_cams.detach().clone()
                + b3_ft_cams.detach().clone()
            )

        # Gating network
        gate_l5 = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_pool = self.avg_pool(gate_l5).view(b, -1)
        gate_pred = F.softmax(self.gate_cls(gate_pool) / self.gate_temp, dim=1)

        # Adaptive fusion using gating weights
        gate_logits = torch.stack(
            [b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1
        )
        gate_logits = gate_logits * gate_pred.view(
            gate_pred.size(0), 1, gate_pred.size(1)
        )
        gate_logits = gate_logits.sum(-1)

        logits_list = [b1_logits, b2_logits, b3_logits, gate_logits]

        if return_ft and y is None:
            outputs = {
                "logits": logits_list,
                "gate_pred": gate_pred,
                "fts": fts,
                "cams_list": cams_list,
            }
        else:
            outputs = {
                "logits": logits_list,
                "gate_pred": gate_pred,
                "multi_label_cams": multi_label_cams,
                "cams_list": cams_list,
            }

        return outputs

    def _extract_multilabel_cams(self, cams_list, targets):
        """
        Extract CAMs for all positive classes in multi-label setting

        Args:
            cams_list: List of CAMs from 3 experts [B, num_classes, H, W]
            targets: Multi-hot labels [B, num_classes]

        Returns:
            extracted_cams: List of CAMs for positive classes per expert
        """
        batch_size = targets.size(0)
        extracted_cams = []

        for expert_idx, expert_cams in enumerate(cams_list):
            expert_extracted = []

            for batch_idx in range(batch_size):
                # Find positive class indices for this sample
                positive_classes = torch.where(targets[batch_idx] == 1)[0]

                if len(positive_classes) > 0:
                    # Extract CAMs for positive classes
                    sample_cams = expert_cams[
                        batch_idx, positive_classes
                    ]  # [num_positive, H, W]
                    expert_extracted.append(sample_cams)
                else:
                    # If no positive classes, create zero tensor
                    H, W = expert_cams.shape[-2:]
                    expert_extracted.append(
                        torch.zeros(1, H, W, device=expert_cams.device)
                    )

            extracted_cams.append(expert_extracted)

        return extracted_cams

    def get_params(self, prefix="extractor"):
        """Get model parameters for different learning rates"""
        extractor_params = (
            list(self.shared_l3.parameters())
            + list(self.branch1_l4.parameters())
            + list(self.branch1_l5.parameters())
            + list(self.branch2_l4.parameters())
            + list(self.branch2_l5.parameters())
            + list(self.branch3_l4.parameters())
            + list(self.branch3_l5.parameters())
            + list(self.gate_l3.parameters())
            + list(self.gate_l4.parameters())
            + list(self.gate_l5.parameters())
        )
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(
            lambda p: id(p) not in extractor_params_ids, self.parameters()
        )

        if prefix in ["extractor", "extract"]:
            return extractor_params
        elif prefix in ["classifier"]:
            return classifier_params

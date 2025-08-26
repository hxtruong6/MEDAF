import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import build_backbone, conv1x1, Classifier


class PerClassGating(nn.Module):
    """
    Per-class gating mechanism for multi-label classification
    
    Instead of global gating that applies the same weights across all classes,
    this module learns class-specific gating weights, allowing different
    experts to specialize in different classes.
    """
    
    def __init__(self, feature_dim, num_classes, num_experts=3, 
                 hidden_dim=None, dropout=0.1):
        super(PerClassGating, self).__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        
        if hidden_dim is None:
            hidden_dim = feature_dim // 4
        
        # Shared feature transformation
        self.shared_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Per-class gating networks
        self.class_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_experts)
            ) for _ in range(num_classes)
        ])
        
        # Initialize with small weights to start with uniform gating
        for gate in self.class_gates:
            for layer in gate:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, features, temperature=1.0):
        """
        Args:
            features: [B, feature_dim] - Global features from gating network
            temperature: Temperature for softmax (higher = more uniform)
            
        Returns:
            gate_weights: [B, num_classes, num_experts] - Per-class gating weights
        """
        batch_size = features.size(0)
        
        # Shared transformation
        shared_features = self.shared_transform(features)  # [B, hidden_dim]
        
        # Compute per-class gating weights
        gate_logits = []
        for class_gate in self.class_gates:
            logits = class_gate(shared_features)  # [B, num_experts]
            gate_logits.append(logits)
        
        gate_logits = torch.stack(gate_logits, dim=1)  # [B, num_classes, num_experts]
        
        # Apply temperature and softmax
        gate_weights = F.softmax(gate_logits / temperature, dim=-1)
        
        return gate_weights, gate_logits


class LabelCorrelationModule(nn.Module):
    """
    Module to capture label co-occurrence patterns for better gating
    """
    
    def __init__(self, num_classes, embedding_dim=64):
        super(LabelCorrelationModule, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Label embeddings
        self.label_embeddings = nn.Embedding(num_classes, embedding_dim)
        
        # Correlation attention
        self.correlation_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=4, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, predicted_labels=None):
        """
        Args:
            predicted_labels: [B, num_classes] - Soft predictions (optional)
            
        Returns:
            correlation_features: [B, num_classes, embedding_dim]
        """
        # Get all label embeddings
        label_indices = torch.arange(self.num_classes, device=self.label_embeddings.weight.device)
        all_embeddings = self.label_embeddings(label_indices)  # [num_classes, embedding_dim]
        
        batch_size = 1 if predicted_labels is None else predicted_labels.size(0)
        
        # Expand for batch
        label_emb = all_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_classes, embedding_dim]
        
        # Self-attention to capture correlations
        corr_emb, _ = self.correlation_attention(label_emb, label_emb, label_emb)
        
        # Final projection
        correlation_features = self.output_proj(corr_emb)
        
        return correlation_features


class MultiLabelMEDAFv2(nn.Module):
    """
    Enhanced Multi-Label MEDAF with configurable per-class gating
    
    Key improvements:
    1. Configurable gating: Global vs Per-class
    2. Label correlation modeling
    3. Enhanced attention diversity
    4. Comparative evaluation support
    """
    
    def __init__(self, args=None):
        super(MultiLabelMEDAFv2, self).__init__()
        
        # Configuration
        self.use_per_class_gating = args.get("use_per_class_gating", False)
        self.use_label_correlation = args.get("use_label_correlation", False)
        self.enhanced_diversity = args.get("enhanced_diversity", False)
        
        # Model architecture
        backbone, feature_dim, self.cam_size = build_backbone(
            img_size=args["img_size"],
            backbone_name=args["backbone"],
            projection_dim=-1,
            inchan=3,
        )
        
        self.img_size = args["img_size"]
        self.gate_temp = args["gate_temp"]
        self.num_classes = args["num_classes"]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        print(f"Initializing MultiLabelMEDAFv2 with:")
        print(f"  - Per-class gating: {self.use_per_class_gating}")
        print(f"  - Label correlation: {self.use_label_correlation}")
        print(f"  - Enhanced diversity: {self.enhanced_diversity}")
        
        # Shared layers (L1-L3)
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-6])

        # Expert branches
        self.branch1_l4 = nn.Sequential(*list(backbone.children())[-6:-3])
        self.branch1_l5 = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_classes)

        self.branch2_l4 = copy.deepcopy(self.branch1_l4)
        self.branch2_l5 = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_classes)

        self.branch3_l4 = copy.deepcopy(self.branch1_l4)
        self.branch3_l5 = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_classes)

        # Gating network backbone
        self.gate_l3 = copy.deepcopy(self.shared_l3)
        self.gate_l4 = copy.deepcopy(self.branch1_l4)
        self.gate_l5 = copy.deepcopy(self.branch1_l5)
        
        # Configurable gating mechanism
        if self.use_per_class_gating:
            # Per-class gating
            self.per_class_gating = PerClassGating(
                feature_dim=feature_dim,
                num_classes=self.num_classes,
                num_experts=3,
                dropout=args.get("gating_dropout", 0.1)
            )
            print(f"  - Using per-class gating with {self.num_classes} class-specific gates")
        else:
            # Global gating (original MEDAF style)
            self.gate_cls = nn.Sequential(
                Classifier(feature_dim, int(feature_dim / 4), bias=True),
                Classifier(int(feature_dim / 4), 3, bias=True),
            )
            print(f"  - Using global gating (original MEDAF style)")
        
        # Optional label correlation module
        if self.use_label_correlation:
            self.label_correlation = LabelCorrelationModule(
                num_classes=self.num_classes,
                embedding_dim=args.get("label_embedding_dim", 64)
            )
            print(f"  - Using label correlation module")

    def forward(self, x, y=None, return_ft=False, return_attention_weights=False):
        """
        Forward pass with configurable gating
        
        Args:
            x: Input tensor [B, C, H, W]
            y: Multi-hot labels [B, num_classes] or None
            return_ft: Whether to return features
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary with outputs and optional attention weights
        """
        b = x.size(0)
        ft_till_l3 = self.shared_l3(x)

        # Expert branches
        branch1_l4 = self.branch1_l4(ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_ft_cams = self.branch1_cls(branch1_l5)  # [B, num_classes, H, W]
        b1_logits = self.avg_pool(b1_ft_cams).view(b, -1)

        branch2_l4 = self.branch2_l4(ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_ft_cams = self.branch2_cls(branch2_l5)
        b2_logits = self.avg_pool(b2_ft_cams).view(b, -1)

        branch3_l4 = self.branch3_l4(ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_ft_cams = self.branch3_cls(branch3_l5)
        b3_logits = self.avg_pool(b3_ft_cams).view(b, -1)

        # Store for diversity computation
        cams_list = [b1_ft_cams, b2_ft_cams, b3_ft_cams]
        expert_logits = [b1_logits, b2_logits, b3_logits]

        # Extract multi-label CAMs
        if y is not None:
            multi_label_cams = self._extract_multilabel_cams(cams_list, y)
        else:
            multi_label_cams = None

        if return_ft:
            fts = (
                b1_ft_cams.detach().clone()
                + b2_ft_cams.detach().clone()
                + b3_ft_cams.detach().clone()
            )

        # Gating network feature extraction
        gate_l5 = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_features = self.avg_pool(gate_l5).view(b, -1)

        # Configurable gating mechanism
        if self.use_per_class_gating:
            # Per-class gating
            gate_weights, gate_logits = self.per_class_gating(gate_features, self.gate_temp)
            
            # Apply per-class weights
            expert_stack = torch.stack(expert_logits, dim=-1)  # [B, num_classes, 3]
            fused_logits = (expert_stack * gate_weights).sum(dim=-1)  # [B, num_classes]
            
            # For compatibility with original interface
            gate_pred = gate_weights.mean(dim=1)  # [B, 3] - average across classes
            
        else:
            # Global gating (original MEDAF)
            gate_pred = F.softmax(self.gate_cls(gate_features) / self.gate_temp, dim=1)
            
            gate_logits_stack = torch.stack(
                [b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1
            )
            gate_logits_stack = gate_logits_stack * gate_pred.view(
                gate_pred.size(0), 1, gate_pred.size(1)
            )
            fused_logits = gate_logits_stack.sum(-1)

        # Prepare outputs
        logits_list = expert_logits + [fused_logits]
        
        outputs = {
            "logits": logits_list,
            "gate_pred": gate_pred,
            "cams_list": cams_list,
            "gating_type": "per_class" if self.use_per_class_gating else "global"
        }
        
        if y is not None:
            outputs["multi_label_cams"] = multi_label_cams
        
        if return_ft and y is None:
            outputs["fts"] = fts
            
        if return_attention_weights and self.use_per_class_gating:
            outputs["per_class_weights"] = gate_weights
            outputs["gate_logits"] = gate_logits
        
        # Label correlation features (if enabled)
        if self.use_label_correlation:
            corr_features = self.label_correlation()
            outputs["correlation_features"] = corr_features

        return outputs

    def _extract_multilabel_cams(self, cams_list, targets):
        """Extract CAMs for positive classes - same as v1"""
        batch_size = targets.size(0)
        extracted_cams = []
        
        for expert_idx, expert_cams in enumerate(cams_list):
            expert_extracted = []
            
            for batch_idx in range(batch_size):
                positive_classes = torch.where(targets[batch_idx] == 1)[0]
                
                if len(positive_classes) > 0:
                    sample_cams = expert_cams[batch_idx, positive_classes]
                    expert_extracted.append(sample_cams)
                else:
                    H, W = expert_cams.shape[-2:]
                    expert_extracted.append(torch.zeros(1, H, W, device=expert_cams.device))
            
            extracted_cams.append(expert_extracted)
        
        return extracted_cams

    def get_params(self, prefix="extractor"):
        """Get model parameters with proper grouping"""
        base_extractor_params = (
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
        
        # Add gating-specific parameters to extractor
        if self.use_per_class_gating:
            base_extractor_params.extend(list(self.per_class_gating.parameters()))
        
        if self.use_label_correlation:
            base_extractor_params.extend(list(self.label_correlation.parameters()))
        
        extractor_params_ids = list(map(id, base_extractor_params))
        classifier_params = filter(
            lambda p: id(p) not in extractor_params_ids, self.parameters()
        )

        if prefix in ["extractor", "extract"]:
            return base_extractor_params
        elif prefix in ["classifier"]:
            return list(classifier_params)

    def get_gating_summary(self):
        """Get summary of current gating configuration"""
        summary = {
            "gating_type": "per_class" if self.use_per_class_gating else "global",
            "num_classes": self.num_classes,
            "use_label_correlation": self.use_label_correlation,
            "enhanced_diversity": self.enhanced_diversity,
        }
        
        if self.use_per_class_gating:
            total_gate_params = sum(p.numel() for p in self.per_class_gating.parameters())
            summary["gating_parameters"] = total_gate_params
        
        return summary

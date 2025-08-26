# Multi-Label MEDAF Phase 2: Per-Class Gating Enhancement

## Table of Contents

1. [Overview](#overview)
2. [Phase 2 Enhancements](#phase-2-enhancements)  
3. [Architecture Deep Dive](#architecture-deep-dive)
4. [Configuration System](#configuration-system)
5. [Comparative Analysis Framework](#comparative-analysis-framework)
6. [Implementation Guide](#implementation-guide)
7. [Experimental Setup](#experimental-setup)
8. [Performance Analysis](#performance-analysis)
9. [Usage Examples](#usage-examples)
10. [Research Extensions](#research-extensions)

## Overview

Phase 2 of Multi-Label MEDAF introduces **per-class gating mechanisms** that enable class-specific expert selection and fusion. Unlike the global gating in Phase 1, this approach allows different classes to leverage different expert combinations, leading to more sophisticated and specialized representations for multi-label classification tasks.

### Key Innovations

1. **🎯 Per-Class Gating**: Class-specific expert selection and weighting
2. **🔄 Configurable Architecture**: Toggle between global and per-class gating
3. **📊 Comparative Framework**: Built-in evaluation system for approach comparison
4. **🧠 Label Correlation Modeling**: Capture label co-occurrence patterns
5. **📈 Enhanced Diversity Loss**: Gating-aware attention diversity computation

## Phase 2 Enhancements

### 1. Per-Class Gating Mechanism

**Motivation**: Different classes in multi-label scenarios may benefit from different expert combinations. For example, in medical imaging:
- Expert 1: Specializes in anatomical structures
- Expert 2: Focuses on pathological patterns  
- Expert 3: Analyzes texture and fine details

**Implementation**:
```python
class PerClassGating(nn.Module):
    def __init__(self, feature_dim, num_classes, num_experts=3):
        # Shared feature transformation
        self.shared_transform = nn.Sequential(...)
        
        # Per-class gating networks
        self.class_gates = nn.ModuleList([
            nn.Sequential(...) for _ in range(num_classes)
        ])
```

**Mathematical Formulation**:
```
For class c and input features f:
g_c = softmax(W_c^T · σ(W_shared · f) / τ)
where g_c ∈ ℝ^3 represents expert weights for class c
```

### 2. Label Correlation Module

**Purpose**: Capture semantic relationships between labels to improve gating decisions.

```python
class LabelCorrelationModule(nn.Module):
    def __init__(self, num_classes, embedding_dim=64):
        self.label_embeddings = nn.Embedding(num_classes, embedding_dim)
        self.correlation_attention = nn.MultiheadAttention(...)
```

**Benefits**:
- Improved gating decisions based on label relationships
- Better handling of label co-occurrence patterns
- Enhanced semantic understanding

### 3. Enhanced Attention Diversity Loss

**Original (Phase 1)**:
```python
def multiLabelAttnDiv(cams_list, targets):
    # Standard cosine similarity between expert attention maps
```

**Enhanced (Phase 2)**:
```python
def enhancedMultiLabelAttnDiv(cams_list, targets, gate_weights=None):
    # Weight attention maps by their gating importance
    if gate_weights is not None:
        expert_cams = expert_cams * class_gate_weights.unsqueeze(-1)
    # Support multiple diversity measures: cosine, L2, KL-divergence
```

## Architecture Deep Dive

### 1. MultiLabelMEDAFv2 Structure

```
Input Image [B, 3, H, W]
         ↓
   Shared Layers (L1-L3)
         ↓
    ┌─────────┬─────────┬─────────┐
    │Expert 1 │Expert 2 │Expert 3 │
    │(L4-L5)  │(L4-L5)  │(L4-L5)  │
    └─────────┴─────────┴─────────┘
         ↓         ↓         ↓
    [CAMs_1]  [CAMs_2]  [CAMs_3]
         ↓         ↓         ↓
    [Logits_1][Logits_2][Logits_3]
                   ↓
              Gating Network
                   ↓
         ┌─────────────────────┐
         │ Global vs Per-Class │
         │    Gating Choice    │
         └─────────────────────┘
                   ↓
            Adaptive Fusion
                   ↓
           [Fused Logits]
```

### 2. Gating Mechanism Comparison

| Feature | Global Gating (Phase 1) | Per-Class Gating (Phase 2) |
|---------|-------------------------|----------------------------|
| **Weights** | [B, 3] | [B, num_classes, 3] |
| **Specialization** | Same for all classes | Class-specific |
| **Parameters** | O(d → 3) | O(d → 3 × C) |
| **Flexibility** | Limited | High |
| **Computational Cost** | Low | Medium |

### 3. Configuration Matrix

```python
# Configuration combinations
configs = {
    "baseline": {
        "use_per_class_gating": False,
        "use_label_correlation": False,
        "enhanced_diversity": False
    },
    "per_class_only": {
        "use_per_class_gating": True,
        "use_label_correlation": False, 
        "enhanced_diversity": False
    },
    "full_enhanced": {
        "use_per_class_gating": True,
        "use_label_correlation": True,
        "enhanced_diversity": True
    }
}
```

## Configuration System

### 1. Model Configuration

```python
args = {
    # Basic setup
    "img_size": 224,
    "backbone": "resnet18",
    "num_classes": 20,
    "gate_temp": 100,
    
    # Phase 2 features
    "use_per_class_gating": True,      # Enable per-class gating
    "use_label_correlation": True,      # Enable label correlation
    "enhanced_diversity": True,         # Enhanced diversity loss
    
    # Advanced options
    "gating_dropout": 0.1,             # Dropout in gating networks
    "label_embedding_dim": 64,         # Label embedding dimension
    "diversity_type": "cosine",        # Diversity measure type
    "gating_regularization": 0.01,     # Gating regularization weight
    
    # Training
    "loss_wgts": [0.7, 1.0, 0.01],    # [expert, gate, diversity]
}
```

### 2. Dynamic Model Creation

```python
def create_model(config_name, args):
    """Create model based on configuration"""
    if config_name == "global":
        args["use_per_class_gating"] = False
    elif config_name == "per_class":
        args["use_per_class_gating"] = True
    elif config_name == "enhanced":
        args.update({
            "use_per_class_gating": True,
            "use_label_correlation": True,
            "enhanced_diversity": True
        })
    
    return MultiLabelMEDAFv2(args)
```

## Comparative Analysis Framework

### 1. ComparativeTrainingFramework

```python
framework = ComparativeTrainingFramework(args)

# Train global model
model_global = create_model("global", args)
metrics_global = train_multilabel_v2(
    train_loader, model_global, criterion, optimizer, 
    args, device, framework
)

# Train per-class model  
model_per_class = create_model("per_class", args)
metrics_per_class = train_multilabel_v2(
    train_loader, model_per_class, criterion, optimizer,
    args, device, framework
)

# Compare results
framework.print_comparison()
```

### 2. Metrics Tracking

The framework automatically tracks:
- **Subset Accuracy**: Exact match accuracy
- **Hamming Accuracy**: Label-wise accuracy  
- **F1 Score**: Macro F1 across labels
- **Diversity Loss**: Attention diversity
- **Gating Entropy**: Expert specialization measure
- **Training Time**: Computational efficiency

### 3. Statistical Analysis

```python
def analyze_expert_specialization(gate_weights):
    """
    Analyze how experts specialize across classes
    
    Args:
        gate_weights: [N, num_classes, 3] - Per-class expert weights
        
    Returns:
        specialization_metrics: Dictionary with analysis
    """
    # Compute expert preference per class
    expert_preferences = gate_weights.mean(dim=0)  # [num_classes, 3]
    
    # Measure specialization
    entropy = -(expert_preferences * torch.log(expert_preferences + 1e-8)).sum(dim=-1)
    specialization = expert_preferences.std(dim=0).mean()
    
    return {
        "expert_entropy": entropy.mean().item(),
        "specialization_score": specialization.item(),
        "dominant_expert_ratio": (expert_preferences.max(dim=-1)[0] > 0.5).float().mean().item()
    }
```

## Implementation Guide

### 1. Step-by-Step Implementation

#### Step 1: Basic Setup
```python
from core.multilabel_net_v2 import MultiLabelMEDAFv2
from core.multilabel_train_v2 import train_multilabel_v2, ComparativeTrainingFramework

# Configuration
args = {
    "img_size": 32,
    "backbone": "resnet18", 
    "num_classes": 10,
    "use_per_class_gating": True,  # Enable Phase 2
}
```

#### Step 2: Model Creation
```python
model = MultiLabelMEDAFv2(args)
print(model.get_gating_summary())
```

#### Step 3: Training Setup
```python
criterion = {"bce": nn.BCEWithLogitsLoss()}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
framework = ComparativeTrainingFramework(args)
```

#### Step 4: Training
```python
for epoch in range(num_epochs):
    metrics = train_multilabel_v2(
        train_loader, model, criterion, optimizer,
        args, device, framework
    )
    
    if epoch % 5 == 0:
        eval_metrics = evaluate_multilabel_v2(
            model, test_loader, criterion, args, device
        )
        print(f"Epoch {epoch}: {eval_metrics}")
```

### 2. Configuration Examples

#### Research Experiment Setup
```python
# Compare all approaches
experiments = {
    "baseline_global": {
        "use_per_class_gating": False,
        "use_label_correlation": False
    },
    "per_class_basic": {
        "use_per_class_gating": True,
        "use_label_correlation": False
    },
    "per_class_enhanced": {
        "use_per_class_gating": True,
        "use_label_correlation": True,
        "enhanced_diversity": True
    }
}

results = {}
for exp_name, config in experiments.items():
    args.update(config)
    model = MultiLabelMEDAFv2(args)
    results[exp_name] = run_experiment(model, data_loader)
```

## Experimental Setup

### 1. Datasets

**Recommended Multi-Label Datasets**:
- **PASCAL VOC 2007**: 20 object classes
- **MS-COCO**: 80 object categories  
- **NUS-WIDE**: 81 concept labels
- **FLICKR-25K**: 25 concept labels
- **Synthetic**: Controllable label correlations

### 2. Evaluation Protocol

```python
def comprehensive_evaluation(model, test_loader, args):
    """Comprehensive evaluation protocol"""
    
    # Standard metrics
    metrics = evaluate_multilabel_v2(model, test_loader, criterion, args, device)
    
    # Per-class analysis
    per_class_metrics = evaluate_per_class_performance(model, test_loader)
    
    # Attention visualization
    attention_analysis = analyze_attention_patterns(model, sample_batch)
    
    # Computational efficiency
    efficiency_metrics = measure_computational_cost(model)
    
    return {
        "standard": metrics,
        "per_class": per_class_metrics,
        "attention": attention_analysis,
        "efficiency": efficiency_metrics
    }
```

### 3. Ablation Studies

| Study | Configuration | Purpose |
|-------|---------------|---------|
| **A1** | Global vs Per-Class | Core mechanism comparison |
| **A2** | Diversity Types | cosine vs L2 vs KL |
| **A3** | Label Correlation | With/without correlation module |
| **A4** | Gating Regularization | Different λ values |
| **A5** | Temperature Scaling | τ ∈ {10, 50, 100, 200} |

## Performance Analysis

### 1. Expected Performance Gains

**Per-Class Gating Benefits**:
- **Subset Accuracy**: +5-15% improvement
- **Class-specific F1**: +10-20% for complex label combinations
- **Attention Diversity**: Higher specialization scores
- **Computational Cost**: 1.5-2x increase in parameters

### 2. When Per-Class Gating Helps Most

1. **High Label Correlation**: Datasets with strong co-occurrence patterns
2. **Diverse Semantics**: Labels representing different visual concepts
3. **Imbalanced Labels**: Some labels much rarer than others
4. **Complex Scenes**: Multiple objects with spatial relationships

### 3. Performance Monitoring

```python
def monitor_training_progress(framework):
    """Monitor key training indicators"""
    
    summary = framework.get_comparison_summary()
    
    # Check for improvement
    if 'per_class' in summary and 'global' in summary:
        pc_acc = summary['per_class']['final_subset_acc']
        global_acc = summary['global']['final_subset_acc']
        
        improvement = pc_acc - global_acc
        
        if improvement > 0.02:  # 2% improvement
            print("✅ Per-class gating shows significant improvement")
        elif improvement > 0:
            print("✳️ Per-class gating shows modest improvement")
        else:
            print("⚠️ Per-class gating may be overfitting")
```

## Usage Examples

### 1. Quick Start Example

```python
#!/usr/bin/env python3
"""Quick start example for Multi-Label MEDAF Phase 2"""

import torch
import torch.nn as nn
from core.multilabel_net_v2 import MultiLabelMEDAFv2
from core.multilabel_train_v2 import train_multilabel_v2

# Configuration
args = {
    "img_size": 64,
    "backbone": "resnet18",
    "num_classes": 20,
    "gate_temp": 100,
    "use_per_class_gating": True,
    "use_label_correlation": True,
    "enhanced_diversity": True,
    "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
    "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
    "loss_wgts": [0.7, 1.0, 0.01]
}

# Create model
model = MultiLabelMEDAFv2(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print configuration
print("Model Configuration:")
print(model.get_gating_summary())

# Training setup
criterion = {"bce": nn.BCEWithLogitsLoss()}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training (assuming you have train_loader)
# for epoch in range(num_epochs):
#     metrics = train_multilabel_v2(
#         train_loader, model, criterion, optimizer, args, device
#     )
```

### 2. Comparative Experiment

```python
#!/usr/bin/env python3
"""Comparative experiment: Global vs Per-Class Gating"""

from core.multilabel_train_v2 import ComparativeTrainingFramework

def run_comparative_experiment(train_loader, test_loader, args):
    """Run comparison between global and per-class gating"""
    
    framework = ComparativeTrainingFramework(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    criterion = {"bce": nn.BCEWithLogitsLoss()}
    
    # Experiment 1: Global Gating
    print("Training Global Gating Model...")
    args_global = args.copy()
    args_global["use_per_class_gating"] = False
    
    model_global = MultiLabelMEDAFv2(args_global).to(device)
    optimizer_global = torch.optim.Adam(model_global.parameters(), lr=0.001)
    
    for epoch in range(10):  # Quick training
        framework.current_epoch = epoch
        train_multilabel_v2(
            train_loader, model_global, criterion, optimizer_global,
            args_global, device, framework
        )
    
    # Experiment 2: Per-Class Gating
    print("\nTraining Per-Class Gating Model...")
    args_pc = args.copy()
    args_pc["use_per_class_gating"] = True
    
    model_pc = MultiLabelMEDAFv2(args_pc).to(device)
    optimizer_pc = torch.optim.Adam(model_pc.parameters(), lr=0.001)
    
    for epoch in range(10):
        framework.current_epoch = epoch
        train_multilabel_v2(
            train_loader, model_pc, criterion, optimizer_pc,
            args_pc, device, framework
        )
    
    # Print comparison
    framework.print_comparison()
    
    return framework.get_comparison_summary()

# Usage:
# results = run_comparative_experiment(train_loader, test_loader, args)
```

### 3. Advanced Research Setup

```python
#!/usr/bin/env python3
"""Advanced research setup with full ablation studies"""

def run_ablation_study(base_args, train_loader, test_loader):
    """Run comprehensive ablation study"""
    
    ablation_configs = {
        "baseline": {
            "use_per_class_gating": False,
            "use_label_correlation": False,
            "enhanced_diversity": False
        },
        "per_class_only": {
            "use_per_class_gating": True,
            "use_label_correlation": False,
            "enhanced_diversity": False
        },
        "with_correlation": {
            "use_per_class_gating": True,
            "use_label_correlation": True,
            "enhanced_diversity": False
        },
        "full_model": {
            "use_per_class_gating": True,
            "use_label_correlation": True,
            "enhanced_diversity": True,
            "diversity_type": "cosine"
        },
        "kl_diversity": {
            "use_per_class_gating": True,
            "use_label_correlation": True,
            "enhanced_diversity": True,
            "diversity_type": "kl"
        }
    }
    
    results = {}
    
    for config_name, config in ablation_configs.items():
        print(f"\n{'='*50}")
        print(f"Running: {config_name}")
        print(f"{'='*50}")
        
        # Update args
        experiment_args = base_args.copy()
        experiment_args.update(config)
        
        # Create and train model
        model = MultiLabelMEDAFv2(experiment_args)
        model.to(device)
        
        # Training loop (simplified)
        criterion = {"bce": nn.BCEWithLogitsLoss()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_f1 = 0
        for epoch in range(20):
            train_metrics = train_multilabel_v2(
                train_loader, model, criterion, optimizer,
                experiment_args, device
            )
            
            if epoch % 5 == 0:
                eval_metrics = evaluate_multilabel_v2(
                    model, test_loader, criterion, experiment_args, device
                )
                
                if eval_metrics['f1_score'] > best_f1:
                    best_f1 = eval_metrics['f1_score']
        
        results[config_name] = {
            "best_f1": best_f1,
            "final_metrics": eval_metrics,
            "config": config
        }
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    for config_name, result in results.items():
        print(f"{config_name:<20}: F1={result['best_f1']:.4f}")
    
    return results
```

## Research Extensions

### 1. Advanced Gating Mechanisms

#### Hierarchical Per-Class Gating
```python
class HierarchicalGating(nn.Module):
    """Hierarchical gating for taxonomic label structures"""
    
    def __init__(self, label_hierarchy, feature_dim):
        self.hierarchy = label_hierarchy
        self.level_gates = nn.ModuleDict()
        
        for level, classes in label_hierarchy.items():
            self.level_gates[level] = PerClassGating(feature_dim, len(classes))
```

#### Attention-Based Gating
```python
class AttentionGating(nn.Module):
    """Use attention mechanism for expert selection"""
    
    def __init__(self, feature_dim, num_classes, num_experts):
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(num_classes, feature_dim)
        self.value_proj = nn.Linear(num_experts, feature_dim)
```

### 2. Dynamic Expert Architecture

#### Conditional Expert Activation
```python
class ConditionalExperts(nn.Module):
    """Activate experts based on predicted label complexity"""
    
    def forward(self, x, predicted_complexity):
        if predicted_complexity < threshold:
            return self.lightweight_expert(x)
        else:
            return self.full_expert_ensemble(x)
```

### 3. Meta-Learning Extensions

#### Few-Shot Label Learning
```python
class MetaLabelLearning(nn.Module):
    """Quickly adapt to new label combinations"""
    
    def adapt_to_new_labels(self, support_set, query_set):
        # Use meta-learning to adapt gating for new label combinations
        pass
```

### 4. Uncertainty Quantification

#### Gating Uncertainty
```python
def compute_gating_uncertainty(gate_logits):
    """Measure uncertainty in expert selection"""
    gate_probs = F.softmax(gate_logits, dim=-1)
    entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1)
    return entropy  # Higher entropy = more uncertainty
```

## Conclusion

Multi-Label MEDAF Phase 2 represents a significant advancement in multi-label classification through:

1. **🎯 Class-Specific Intelligence**: Per-class gating enables specialized expert utilization
2. **🔄 Flexible Architecture**: Easy toggling between approaches for research
3. **📊 Comprehensive Analysis**: Built-in comparative evaluation framework
4. **🚀 Research Ready**: Extensible design for advanced research

The configurable nature allows researchers to:
- Compare global vs per-class approaches
- Conduct comprehensive ablation studies  
- Extend with advanced research components
- Apply to diverse multi-label domains

This implementation provides both practical improvements and a solid foundation for future multi-label learning research.

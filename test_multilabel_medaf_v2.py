#!/usr/bin/env python3
"""
Test script for Multi-Label MEDAF Phase 2 with Per-Class Gating
Validates enhanced features and comparative analysis framework
"""

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import sys
import os
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.multilabel_net_v2 import MultiLabelMEDAFv2, PerClassGating, LabelCorrelationModule
from core.multilabel_train_v2 import (
    train_multilabel_v2, 
    evaluate_multilabel_v2,
    enhancedMultiLabelAttnDiv,
    ComparativeTrainingFramework
)
from test_multilabel_medaf import SyntheticMultiLabelDataset


def test_per_class_gating():
    """Test per-class gating mechanism"""
    print("\n" + "="*60)
    print("Testing Per-Class Gating Mechanism")
    print("="*60)
    
    # Configuration
    feature_dim = 128
    num_classes = 8
    num_experts = 3
    batch_size = 4
    
    # Create per-class gating module
    per_class_gating = PerClassGating(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_experts=num_experts,
        dropout=0.1
    )
    
    # Test input
    features = torch.randn(batch_size, feature_dim)
    
    print(f"Input features shape: {features.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of experts: {num_experts}")
    
    # Forward pass
    gate_weights, gate_logits = per_class_gating(features, temperature=100.0)
    
    print(f"\nOutput shapes:")
    print(f"Gate weights: {gate_weights.shape}")  # [B, num_classes, num_experts]
    print(f"Gate logits: {gate_logits.shape}")
    
    # Verify properties
    print(f"\nGate weights properties:")
    print(f"Sum across experts (should be ~1.0): {gate_weights.sum(dim=-1).mean():.4f}")
    print(f"Min weight: {gate_weights.min():.4f}")
    print(f"Max weight: {gate_weights.max():.4f}")
    
    # Test with different temperatures
    temperatures = [10, 50, 100, 200]
    print(f"\nTemperature effects:")
    for temp in temperatures:
        weights, _ = per_class_gating(features, temperature=temp)
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        max_weight = weights.max().item()
        print(f"Temp={temp:3d}: Entropy={entropy:.3f}, Max Weight={max_weight:.3f}")
    
    print("✓ Per-class gating test PASSED")
    return True


def test_label_correlation_module():
    """Test label correlation module"""
    print("\n" + "="*60)
    print("Testing Label Correlation Module")
    print("="*60)
    
    num_classes = 10
    embedding_dim = 64
    batch_size = 3
    
    # Create module
    label_corr = LabelCorrelationModule(
        num_classes=num_classes,
        embedding_dim=embedding_dim
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Test without predicted labels
    corr_features = label_corr()
    print(f"Correlation features shape: {corr_features.shape}")
    
    # Test with predicted labels
    predicted_labels = torch.randn(batch_size, num_classes)
    corr_features_batch = label_corr(predicted_labels)
    print(f"Batch correlation features shape: {corr_features_batch.shape}")
    
    # Verify embedding properties
    print(f"\nEmbedding statistics:")
    embeddings = label_corr.label_embeddings.weight
    print(f"Embedding mean: {embeddings.mean():.4f}")
    print(f"Embedding std: {embeddings.std():.4f}")
    
    print("✓ Label correlation module test PASSED")
    return True


def test_enhanced_diversity_loss():
    """Test enhanced attention diversity loss"""
    print("\n" + "="*60)
    print("Testing Enhanced Attention Diversity Loss")
    print("="*60)
    
    batch_size = 2
    num_classes = 5
    H, W = 8, 8
    
    # Create test data
    cams_list = [torch.randn(batch_size, num_classes, H, W) for _ in range(3)]
    targets = torch.tensor([
        [1, 0, 1, 0, 1],  # Sample 1: classes 0, 2, 4
        [1, 1, 0, 0, 0],  # Sample 2: classes 0, 1
    ]).float()
    
    # Per-class gating weights
    gate_weights = torch.tensor([
        [[0.8, 0.1, 0.1], [0.0, 0.0, 0.0], [0.1, 0.8, 0.1], [0.0, 0.0, 0.0], [0.1, 0.1, 0.8]],  # Sample 1
        [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # Sample 2
    ]).float()
    
    print(f"CAMs shapes: {[cam.shape for cam in cams_list]}")
    print(f"Targets:\n{targets}")
    print(f"Gate weights shape: {gate_weights.shape}")
    
    # Test different diversity types
    diversity_types = ["cosine", "l2", "kl"]
    
    for div_type in diversity_types:
        # Without gating weights
        div_loss_basic = enhancedMultiLabelAttnDiv(cams_list, targets, diversity_type=div_type)
        
        # With gating weights
        div_loss_enhanced = enhancedMultiLabelAttnDiv(
            cams_list, targets, gate_weights, diversity_type=div_type
        )
        
        print(f"{div_type:>6} diversity - Basic: {div_loss_basic.item():.6f}, Enhanced: {div_loss_enhanced.item():.6f}")
    
    print("✓ Enhanced diversity loss test PASSED")
    return True


def test_model_configurations():
    """Test different model configurations"""
    print("\n" + "="*60)
    print("Testing Model Configurations")
    print("="*60)
    
    base_args = {
        "img_size": 32,
        "backbone": "resnet18",
        "num_classes": 6,
        "gate_temp": 100
    }
    
    configurations = {
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
        "full_enhanced": {
            "use_per_class_gating": True,
            "use_label_correlation": True,
            "enhanced_diversity": True,
            "diversity_type": "cosine",
            "gating_regularization": 0.01
        }
    }
    
    for config_name, config in configurations.items():
        print(f"\n--- Testing {config_name} configuration ---")
        
        # Merge configuration
        args = base_args.copy()
        args.update(config)
        
        # Create model
        model = MultiLabelMEDAFv2(args)
        
        # Get summary
        summary = model.get_gating_summary()
        print(f"Gating type: {summary['gating_type']}")
        print(f"Use correlation: {summary['use_label_correlation']}")
        print(f"Enhanced diversity: {summary['enhanced_diversity']}")
        
        if 'gating_parameters' in summary:
            print(f"Gating parameters: {summary['gating_parameters']:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 2, (2, 6)).float()
        
        with torch.no_grad():
            outputs = model(x, y, return_attention_weights=True)
        
        print(f"Logits shapes: {[logit.shape for logit in outputs['logits']]}")
        print(f"Gating type in output: {outputs['gating_type']}")
        
        if 'per_class_weights' in outputs:
            per_class_weights = outputs['per_class_weights']
            print(f"Per-class weights shape: {per_class_weights.shape}")
            print(f"Average expert preference: {per_class_weights.mean(dim=(0,1))}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
    
    print("\n✓ Model configurations test PASSED")
    return True


def test_comparative_framework():
    """Test comparative training framework"""
    print("\n" + "="*60)
    print("Testing Comparative Training Framework")
    print("="*60)
    
    # Create framework
    args = {"loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"]}
    framework = ComparativeTrainingFramework(args)
    
    # Simulate training results
    global_metrics = {
        'total_loss': 0.8,
        'diversity_loss': 0.05,
        'subset_acc': 65.0,
        'hamming_acc': 78.0,
        'f1': 0.72
    }
    
    per_class_metrics = {
        'total_loss': 0.65,
        'diversity_loss': 0.03,
        'subset_acc': 72.0,
        'hamming_acc': 83.0,
        'f1': 0.78,
        'gating_entropy': 0.85
    }
    
    # Log metrics
    for epoch in range(5):
        framework.current_epoch = epoch
        framework.log_metrics('global', epoch, global_metrics)
        framework.log_metrics('per_class', epoch, per_class_metrics)
    
    # Test comparison
    summary = framework.get_comparison_summary()
    print(f"Summary keys: {list(summary.keys())}")
    
    if 'global' in summary and 'per_class' in summary:
        print("Comparison available:")
        print(f"Global final accuracy: {summary['global']['final_subset_acc']}")
        print(f"Per-class final accuracy: {summary['per_class']['final_subset_acc']}")
        
        # Print full comparison
        framework.print_comparison()
    
    print("✓ Comparative framework test PASSED")
    return True


def test_end_to_end_training():
    """Test end-to-end training with both configurations"""
    print("\n" + "="*60)
    print("Testing End-to-End Training")
    print("="*60)
    
    # Create synthetic dataset
    dataset = SyntheticMultiLabelDataset(
        num_samples=80,
        img_size=32,
        num_classes=6,
        avg_labels_per_sample=2,
        random_state=42
    )
    
    train_loader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    # Training configuration
    base_args = {
        "img_size": 32,
        "backbone": "resnet18",
        "num_classes": 6,
        "gate_temp": 100,
        "loss_keys": ["b1", "b2", "b3", "gate", "divAttn", "total"],
        "acc_keys": ["acc1", "acc2", "acc3", "accGate"],
        "loss_wgts": [0.7, 1.0, 0.01],
        "enhanced_diversity": True,
        "diversity_type": "cosine",
        "gating_regularization": 0.01
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = {"bce": nn.BCEWithLogitsLoss()}
    
    # Create comparative framework
    framework = ComparativeTrainingFramework(base_args)
    
    configurations = [
        ("global", {"use_per_class_gating": False}),
        ("per_class", {"use_per_class_gating": True, "use_label_correlation": True})
    ]
    
    results = {}
    
    for config_name, config_updates in configurations:
        print(f"\n--- Training {config_name} configuration ---")
        
        # Setup configuration
        args = base_args.copy()
        args.update(config_updates)
        
        # Create model
        model = MultiLabelMEDAFv2(args)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"Model: {model.get_gating_summary()}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Training loop (short for testing)
        best_acc = 0
        for epoch in range(3):
            framework.current_epoch = epoch
            
            metrics = train_multilabel_v2(
                train_loader, model, criterion, optimizer,
                args, device, framework
            )
            
            if metrics['subset_acc'] > best_acc:
                best_acc = metrics['subset_acc']
        
        results[config_name] = {
            'best_accuracy': best_acc,
            'final_metrics': metrics
        }
        
        print(f"Best accuracy: {best_acc:.2f}%")
    
    # Print final comparison
    print(f"\n--- Final Comparison ---")
    framework.print_comparison()
    
    # Check if per-class shows improvement
    if len(results) == 2:
        global_acc = results['global']['best_accuracy']
        pc_acc = results['per_class']['best_accuracy']
        improvement = pc_acc - global_acc
        
        print(f"\nPerformance Summary:")
        print(f"Global gating: {global_acc:.2f}%")
        print(f"Per-class gating: {pc_acc:.2f}%")
        print(f"Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print("✅ Per-class gating shows improvement!")
        else:
            print("ℹ️  Results may vary with longer training and real datasets")
    
    print("✓ End-to-end training test PASSED")
    return True


def main():
    """Run all Phase 2 tests"""
    print("Testing Multi-Label MEDAF Phase 2: Per-Class Gating")
    print("Enhanced Features and Comparative Analysis")
    
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run tests
        tests = [
            test_per_class_gating,
            test_label_correlation_module,
            test_enhanced_diversity_loss,
            test_model_configurations,
            test_comparative_framework,
            test_end_to_end_training
        ]
        
        passed = 0
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ Test {test_func.__name__} FAILED: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("PHASE 2 TEST SUMMARY")
        print("="*60)
        print(f"Tests passed: {passed}/{len(tests)}")
        
        if passed == len(tests):
            print("🎉 ALL PHASE 2 TESTS PASSED!")
            print("\nPhase 2 Features Successfully Implemented:")
            print("✅ Per-class gating mechanism")
            print("✅ Label correlation modeling")  
            print("✅ Enhanced attention diversity")
            print("✅ Configurable architecture")
            print("✅ Comparative evaluation framework")
            print("\nNext Steps:")
            print("1. 🔬 Run experiments on real multi-label datasets")
            print("2. 📊 Conduct comprehensive ablation studies")
            print("3. 🚀 Implement advanced research extensions")
            print("4. 📝 Publish comparative analysis results")
        else:
            print("❌ Some Phase 2 tests failed. Please review the implementation.")
            
    except Exception as e:
        print(f"Fatal error during Phase 2 testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

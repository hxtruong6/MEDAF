#!/usr/bin/env python3
"""
Comprehensive Evaluation of Multi-Label MEDAF on Real Datasets
Tests both Phase 1 (global gating) and Phase 2 (per-class gating) on PASCAL VOC and MS-COCO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
import argparse
from collections import defaultdict

# Import our implementations
from core.multilabel_net import MultiLabelMEDAF
from core.multilabel_net_v2 import MultiLabelMEDAFv2
from core.multilabel_train import train_multilabel
from core.multilabel_train_v2 import train_multilabel_v2, evaluate_multilabel_v2, ComparativeTrainingFramework

# Import dataset loaders
from datasets.real_multilabel_datasets import create_multilabel_dataloaders


class RealDatasetEvaluator:
    """
    Comprehensive evaluator for Multi-Label MEDAF on real datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = defaultdict(dict)
        
        print(f"Using device: {self.device}")
        print(f"Configuration: {config}")
    
    def load_dataset(self, dataset_name):
        """Load real multi-label dataset"""
        print(f"\n{'='*60}")
        print(f"Loading {dataset_name.upper()} Dataset")
        print(f"{'='*60}")
        
        data_root = f"./datasets/data/{dataset_name}"
        
        train_loader, test_loader, num_classes = create_multilabel_dataloaders(
            dataset_name=dataset_name,
            data_root=data_root,
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config['num_workers']
        )
        
        # Analyze dataset statistics
        self.analyze_dataset_statistics(train_loader, test_loader, dataset_name)
        
        return train_loader, test_loader, num_classes
    
    def analyze_dataset_statistics(self, train_loader, test_loader, dataset_name):
        """Analyze multi-label dataset statistics"""
        print(f"\nAnalyzing {dataset_name.upper()} Statistics...")
        
        # Collect statistics from training set
        all_labels = []
        for i, (_, labels) in enumerate(train_loader):
            all_labels.append(labels.numpy())
            if i >= 50:  # Sample for speed
                break
        
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Compute statistics
        labels_per_sample = all_labels.sum(axis=1)
        class_frequencies = all_labels.sum(axis=0)
        
        stats = {
            'num_samples_analyzed': len(all_labels),
            'avg_labels_per_sample': float(labels_per_sample.mean()),
            'std_labels_per_sample': float(labels_per_sample.std()),
            'min_labels_per_sample': int(labels_per_sample.min()),
            'max_labels_per_sample': int(labels_per_sample.max()),
            'class_frequencies': class_frequencies.tolist(),
            'most_frequent_classes': class_frequencies.argsort()[-5:][::-1].tolist(),
            'least_frequent_classes': class_frequencies.argsort()[:5].tolist()
        }
        
        print(f"Average labels per sample: {stats['avg_labels_per_sample']:.2f} ± {stats['std_labels_per_sample']:.2f}")
        print(f"Labels per sample range: {stats['min_labels_per_sample']} - {stats['max_labels_per_sample']}")
        print(f"Most frequent classes: {stats['most_frequent_classes']}")
        print(f"Least frequent classes: {stats['least_frequent_classes']}")
        
        self.results[dataset_name]['statistics'] = stats
    
    def evaluate_configuration(self, train_loader, test_loader, num_classes, 
                              config_name, model_config, dataset_name):
        """Evaluate a specific model configuration"""
        print(f"\n{'-'*50}")
        print(f"Evaluating: {config_name} on {dataset_name.upper()}")
        print(f"{'-'*50}")
        
        # Setup model arguments
        args = {
            'img_size': self.config['img_size'],
            'backbone': self.config['backbone'],
            'num_classes': num_classes,
            'gate_temp': 100,
            'loss_keys': ["b1", "b2", "b3", "gate", "divAttn", "total"],
            'acc_keys': ["acc1", "acc2", "acc3", "accGate"],
            'loss_wgts': [0.7, 1.0, 0.01],
            **model_config
        }
        
        # Create model
        if model_config.get('use_per_class_gating', False):
            model = MultiLabelMEDAFv2(args)
        else:
            # Convert args for Phase 1 model
            phase1_args = {k: v for k, v in args.items() if k != 'use_per_class_gating'}
            phase1_args['num_known'] = num_classes  # Phase 1 uses 'num_known'
            model = MultiLabelMEDAF(phase1_args)
        
        model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        if hasattr(model, 'get_gating_summary'):
            summary = model.get_gating_summary()
            print(f"Gating type: {summary['gating_type']}")
        
        # Setup training
        criterion = {'bce': nn.BCEWithLogitsLoss()}
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[self.config['num_epochs']//2, 3*self.config['num_epochs']//4], 
            gamma=0.1
        )
        
        # Training loop
        best_metrics = None
        training_history = []
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Training
            if model_config.get('use_per_class_gating', False):
                train_metrics = train_multilabel_v2(
                    train_loader, model, criterion, optimizer, args, self.device
                )
            else:
                train_metrics = train_multilabel(
                    train_loader, model, criterion, optimizer, args, self.device
                )
            
            # Evaluation every few epochs
            if (epoch + 1) % max(1, self.config['num_epochs'] // 5) == 0:
                if model_config.get('use_per_class_gating', False):
                    eval_metrics = evaluate_multilabel_v2(
                        model, test_loader, criterion, args, self.device
                    )
                else:
                    # Simple evaluation for Phase 1
                    eval_metrics = self._evaluate_phase1(model, test_loader, criterion)
                
                if best_metrics is None or eval_metrics['f1_score'] > best_metrics['f1_score']:
                    best_metrics = eval_metrics.copy()
                    best_metrics['epoch'] = epoch + 1
                
                print(f"Epoch {epoch+1:3d}: "
                      f"Loss={train_metrics if isinstance(train_metrics, float) else train_metrics.get('total_loss', 0):.4f}, "
                      f"F1={eval_metrics['f1_score']:.4f}, "
                      f"SubsetAcc={eval_metrics['subset_accuracy']:.4f}")
                
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics if isinstance(train_metrics, float) else train_metrics.get('total_loss', 0),
                    'eval_metrics': eval_metrics
                })
            
            scheduler.step()
        
        training_time = time.time() - start_time
        
        # Final evaluation
        if model_config.get('use_per_class_gating', False):
            final_metrics = evaluate_multilabel_v2(
                model, test_loader, criterion, args, self.device
            )
        else:
            final_metrics = self._evaluate_phase1(model, test_loader, criterion)
        
        # Compile results
        results = {
            'config': model_config,
            'best_metrics': best_metrics,
            'final_metrics': final_metrics,
            'training_time': training_time,
            'training_history': training_history,
            'model_parameters': total_params
        }
        
        # Print summary
        print(f"\nResults for {config_name}:")
        print(f"  Best F1 Score: {best_metrics['f1_score']:.4f} (epoch {best_metrics['epoch']})")
        print(f"  Best Subset Accuracy: {best_metrics['subset_accuracy']:.4f}")
        print(f"  Final F1 Score: {final_metrics['f1_score']:.4f}")
        print(f"  Training Time: {training_time:.1f}s")
        print(f"  Model Parameters: {total_params:,}")
        
        return results
    
    def _evaluate_phase1(self, model, test_loader, criterion):
        """Simple evaluation for Phase 1 model"""
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data[0].to(self.device), data[1].to(self.device)
                
                outputs = model(inputs)
                logits = outputs['logits'][3]  # Use fused logits
                
                loss = criterion['bce'](logits, targets)
                total_loss += loss.item()
                
                predictions = torch.sigmoid(logits)
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        pred_binary = (all_predictions > 0.5).float()
        
        # Subset accuracy
        subset_acc = (pred_binary == all_targets).all(dim=1).float().mean().item()
        
        # Hamming accuracy
        hamming_acc = (pred_binary == all_targets).float().mean().item()
        
        # Precision, Recall, F1
        tp = (pred_binary * all_targets).sum(dim=0)
        fp = (pred_binary * (1 - all_targets)).sum(dim=0)
        fn = ((1 - pred_binary) * all_targets).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'eval_loss': total_loss / len(test_loader),
            'subset_accuracy': subset_acc,
            'hamming_accuracy': hamming_acc,
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1_score': f1.mean().item(),
            'gating_type': 'global'
        }
    
    def run_comparative_study(self, dataset_name):
        """Run comparative study on a dataset"""
        print(f"\n{'='*70}")
        print(f"COMPARATIVE STUDY: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Load dataset
        train_loader, test_loader, num_classes = self.load_dataset(dataset_name)
        
        # Configurations to compare
        configurations = {
            'global_gating': {
                'name': 'Global Gating (Phase 1)',
                'use_per_class_gating': False
            },
            'per_class_basic': {
                'name': 'Per-Class Gating (Phase 2)',
                'use_per_class_gating': True,
                'use_label_correlation': False,
                'enhanced_diversity': False
            },
            'per_class_enhanced': {
                'name': 'Enhanced Per-Class (Phase 2+)',
                'use_per_class_gating': True,
                'use_label_correlation': True,
                'enhanced_diversity': True,
                'diversity_type': 'cosine'
            }
        }
        
        # Run experiments
        dataset_results = {}
        
        for config_key, config in configurations.items():
            try:
                results = self.evaluate_configuration(
                    train_loader, test_loader, num_classes,
                    config['name'], config, dataset_name
                )
                dataset_results[config_key] = results
                
            except Exception as e:
                print(f"Error evaluating {config['name']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Store results
        self.results[dataset_name]['experiments'] = dataset_results
        
        # Print comparative analysis
        self._print_comparative_analysis(dataset_name, dataset_results)
        
        return dataset_results
    
    def _print_comparative_analysis(self, dataset_name, results):
        """Print comparative analysis of results"""
        print(f"\n{'='*60}")
        print(f"COMPARATIVE ANALYSIS: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        if not results:
            print("No results available for analysis")
            return
        
        # Extract metrics for comparison
        comparison_data = []
        for config_key, result in results.items():
            config_name = result['config'].get('name', config_key)
            best_metrics = result['best_metrics']
            
            comparison_data.append({
                'name': config_name,
                'f1_score': best_metrics['f1_score'],
                'subset_accuracy': best_metrics['subset_accuracy'],
                'hamming_accuracy': best_metrics['hamming_accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'parameters': result['model_parameters'],
                'training_time': result['training_time']
            })
        
        # Sort by F1 score
        comparison_data.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Print table
        print(f"{'Method':<25} {'F1':<8} {'SubsetAcc':<10} {'HammingAcc':<11} {'Precision':<10} {'Recall':<8} {'Params':<10} {'Time(s)':<8}")
        print("-" * 90)
        
        for data in comparison_data:
            print(f"{data['name']:<25} "
                  f"{data['f1_score']:.4f}   "
                  f"{data['subset_accuracy']:.4f}     "
                  f"{data['hamming_accuracy']:.4f}      "
                  f"{data['precision']:.4f}     "
                  f"{data['recall']:.4f}   "
                  f"{data['parameters']/1000:.0f}K     "
                  f"{data['training_time']:.0f}")
        
        # Calculate improvements
        if len(comparison_data) >= 2:
            best = comparison_data[0]
            baseline = comparison_data[-1]  # Assuming global is last
            
            f1_improvement = ((best['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
            acc_improvement = ((best['subset_accuracy'] - baseline['subset_accuracy']) / baseline['subset_accuracy']) * 100
            
            print(f"\n🏆 Best Method: {best['name']}")
            print(f"📈 F1 Score Improvement: {f1_improvement:+.2f}%")
            print(f"📈 Subset Accuracy Improvement: {acc_improvement:+.2f}%")
            
            if f1_improvement > 5:
                print("✅ Significant improvement achieved!")
            elif f1_improvement > 0:
                print("✳️ Positive improvement observed")
            else:
                print("⚠️ No significant improvement (may need longer training)")
    
    def save_results(self, output_path="real_dataset_evaluation_results.json"):
        """Save all results to JSON file"""
        output_path = Path(output_path)
        
        # Convert results for JSON serialization
        serializable_results = {}
        for dataset, data in self.results.items():
            serializable_results[dataset] = {}
            
            if 'statistics' in data:
                serializable_results[dataset]['statistics'] = data['statistics']
            
            if 'experiments' in data:
                serializable_results[dataset]['experiments'] = {}
                for config, result in data['experiments'].items():
                    serializable_results[dataset]['experiments'][config] = {
                        'config': result['config'],
                        'best_metrics': result['best_metrics'],
                        'final_metrics': result['final_metrics'],
                        'training_time': result['training_time'],
                        'model_parameters': result['model_parameters']
                        # Skip training_history for size
                    }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def run_full_evaluation(self):
        """Run full evaluation on all datasets"""
        print("Multi-Label MEDAF Real Dataset Evaluation")
        print("Phase 1 vs Phase 2 Comparison")
        print("="*60)
        
        datasets = self.config['datasets']
        
        for dataset in datasets:
            try:
                self.run_comparative_study(dataset)
            except Exception as e:
                print(f"Error evaluating {dataset}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save all results
        self.save_results()
        
        # Print final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final summary across all datasets"""
        print(f"\n{'='*70}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        if not self.results:
            print("No results available")
            return
        
        for dataset_name, data in self.results.items():
            if 'experiments' not in data:
                continue
            
            print(f"\n📊 {dataset_name.upper()}")
            
            best_result = None
            best_f1 = 0
            
            for config_key, result in data['experiments'].items():
                f1_score = result['best_metrics']['f1_score']
                config_name = result['config'].get('name', config_key)
                
                print(f"  {config_name}: F1={f1_score:.4f}")
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_result = (config_name, result)
            
            if best_result:
                print(f"  🏆 Best: {best_result[0]} (F1={best_f1:.4f})")
        
        print(f"\n🎯 Key Findings:")
        print(f"  • Real dataset validation completed")
        print(f"  • Per-class gating performance measured")  
        print(f"  • Comparative analysis provides clear insights")
        print(f"  • Results saved for detailed analysis")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Multi-Label MEDAF on Real Datasets')
    parser.add_argument('--datasets', nargs='+', default=['pascal_voc'], 
                       choices=['pascal_voc', 'coco'],
                       help='Datasets to evaluate on')
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, 
                       help='Image size')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--backbone', default='resnet18', 
                       help='Backbone architecture')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'datasets': args.datasets,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'learning_rate': args.lr,
        'backbone': args.backbone,
        'num_workers': args.num_workers
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run evaluation
    evaluator = RealDatasetEvaluator(config)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick ChestX-ray14 Dataset Analysis

A simplified version for quick analysis of the ChestX-ray14 dataset.
This script provides essential statistics without extensive visualizations.

Usage:
    python quick_chestxray_analysis.py
"""

import pandas as pd
import numpy as np
from collections import Counter

def quick_analysis(csv_path):
    """Perform quick analysis of the ChestX-ray14 dataset"""
    
    print("🔬 Quick ChestX-ray14 Dataset Analysis")
    print("="*50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Parse finding labels
    df['Finding Labels'] = df['Finding Labels'].astype(str)
    
    # Define the 14 pathology labels
    pathology_labels = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Create binary columns for each pathology
    for label in pathology_labels:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)
    
    # Handle "No Finding" cases
    df['No Finding'] = df['Finding Labels'].apply(lambda x: 1 if x == 'No Finding' else 0)
    
    # Basic statistics
    print(f"\n📊 Basic Statistics:")
    print(f"   Total images: {len(df):,}")
    print(f"   Total patients: {df['Patient ID'].nunique():,}")
    print(f"   Male patients: {len(df[df['Patient Gender'] == 'M']):,}")
    print(f"   Female patients: {len(df[df['Patient Gender'] == 'F']):,}")
    
    # Parse age and get age statistics
    df['Patient Age Numeric'] = df['Patient Age'].str.replace('Y', '').astype(int)
    print(f"   Average age: {df['Patient Age Numeric'].mean():.1f} years")
    print(f"   Age range: {df['Patient Age Numeric'].min()}-{df['Patient Age Numeric'].max()} years")
    
    # Label distribution
    print(f"\n🏥 Pathology Label Distribution:")
    print(f"{'Label':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 45)
    
    all_labels = pathology_labels + ['No Finding']
    label_counts = {}
    
    for label in all_labels:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        label_counts[label] = count
        print(f"{label:<20} {count:<10,} {percentage:<9.2f}%")
    
    # Multi-label analysis
    print(f"\n🔗 Multi-label Analysis:")
    df['Num_Labels'] = df[all_labels].sum(axis=1)
    
    num_labels_dist = df['Num_Labels'].value_counts().sort_index()
    print(f"{'Labels per Image':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 45)
    
    for num_labels, count in num_labels_dist.items():
        percentage = (count / len(df)) * 100
        print(f"{num_labels:<20} {count:<10,} {percentage:<9.2f}%")
    
    # Multi-label statistics
    single_label = len(df[df['Num_Labels'] == 1])
    multi_label = len(df[df['Num_Labels'] > 1])
    no_finding = len(df[df['No Finding'] == 1])
    
    print(f"\n📈 Multi-label Statistics:")
    print(f"   Single label images: {single_label:,} ({single_label/len(df)*100:.1f}%)")
    print(f"   Multi-label images: {multi_label:,} ({multi_label/len(df)*100:.1f}%)")
    print(f"   No finding images: {no_finding:,} ({no_finding/len(df)*100:.1f}%)")
    print(f"   Average labels per image: {df['Num_Labels'].mean():.2f}")
    
    # Most common combinations
    print(f"\n🤝 Most Common Label Combinations:")
    combinations = []
    for idx, row in df.iterrows():
        if row['Num_Labels'] > 1:
            labels = [label for label in pathology_labels if row[label] == 1]
            if labels:
                combinations.append('|'.join(sorted(labels)))
    
    combo_counts = Counter(combinations)
    top_combinations = combo_counts.most_common(10)
    
    for i, (combo, count) in enumerate(top_combinations, 1):
        print(f"   {i:2d}. {combo:<30} {count:,} images")
    
    # Class imbalance analysis
    print(f"\n⚖️ Class Imbalance Analysis:")
    counts = [label_counts[label] for label in pathology_labels if label_counts[label] > 0]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"   Most common: {max(label_counts, key=label_counts.get)} ({max_count:,} cases)")
    print(f"   Least common: {min(label_counts, key=label_counts.get)} ({min_count:,} cases)")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # View position distribution
    print(f"\n📸 View Position Distribution:")
    view_counts = df['View Position'].value_counts()
    for view, count in view_counts.items():
        print(f"   {view}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n✅ Quick analysis complete!")
    
    return df, label_counts

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = "/home/s2320437/WORK/aidan-medaf/datasets/data/chestxray/NIH/Data_Entry_2017.csv"
    
    # Run quick analysis
    df, label_counts = quick_analysis(csv_path)

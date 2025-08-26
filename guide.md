```sh
qsub -q GPU-S -I

module load cuda/12.1

nvcc --version # check CUDA version
nvidia-smi # check GPU
```



## Implementation Priority
### Phase 1: Core Modifications
✅ Replace CrossEntropy with BCEWithLogitsLoss
✅ Modify CAM extraction for multi-hot labels
✅ Implement multi-label attention diversity loss
### Phase 2: Advanced Features
✅ Per-class gating mechanism
✅ Multi-label evaluation metrics
✅ Label-aware expert selection
### Phase 3: Research Extensions
🔬 Hierarchical attention mechanisms
🔬 Dynamic expert architectures
🔬 Label correlation modeling

## Expected Benefits
Enhanced Diversity: Experts specialize in different label combinations
Better Generalization: Improved handling of unseen label combinations (crucial for open-set multi-label)
Semantic Understanding: Class-specific gating captures label relationships
Scalability: Architecture scales to large label vocabularies
This modification transforms MEDAF from a single-label to a sophisticated multi-label learning framework while preserving its core strengths: diverse expert representations and adaptive fusion. The attention diversity mechanism becomes even more powerful in multi-label settings by encouraging experts to focus on different semantic aspects across multiple co-occurring labels.
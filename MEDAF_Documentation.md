# MEDAF: Multi-Expert Diverse Attention Fusion - Implementation Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Core Components](#core-components)
4. [Training Process](#training-process)
5. [Configuration System](#configuration-system)
6. [Dataset Handling](#dataset-handling)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Modification Guide](#modification-guide)
9. [File Structure](#file-structure)
10. [Usage Examples](#usage-examples)

## Overview

MEDAF (Multi-Expert Diverse Attention Fusion) is a discriminative approach for Open Set Recognition (OSR) that learns diverse representations through multiple experts with attention diversity regularization. The key innovation is using attention maps to ensure different experts focus on different aspects of the input, leading to more robust open space handling.

### Key Features

- **Multi-Expert Architecture**: Three parallel branches with shared early layers
- **Attention Diversity Regularization**: Ensures experts learn different attention patterns
- **Adaptive Fusion**: Gating network dynamically combines expert predictions
- **Discriminative Approach**: Achieves SOTA performance without generative components

## Architecture Deep Dive

### 1. MultiBranchNet Structure

The main architecture is implemented in `core/net.py` as `MultiBranchNet`:

```python
class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        # Shared layers (L1-L3)
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-6])
        
        # Three expert branches (L4-L5 + classifier)
        self.branch1_l4 = nn.Sequential(*list(backbone.children())[-6:-3])
        self.branch1_l5 = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_known)
        
        # Similar for branch2 and branch3 (deep copies)
        
        # Gating network
        self.gate_l3 = copy.deepcopy(self.shared_l3)
        self.gate_l4 = copy.deepcopy(self.branch1_l4)
        self.gate_l5 = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(
            Classifier(feature_dim, int(feature_dim/4), bias=True), 
            Classifier(int(feature_dim/4), 3, bias=True)
        )
```

### 2. Forward Pass Flow

```python
def forward(self, x, y=None, return_ft=False):
    # 1. Shared feature extraction (L1-L3)
    ft_till_l3 = self.shared_l3(x)
    
    # 2. Parallel expert processing
    branch1_l4 = self.branch1_l4(ft_till_l3.clone())
    branch1_l5 = self.branch1_l5(branch1_l4)
    b1_ft_cams = self.branch1_cls(branch1_l5)  # Class activation maps
    b1_logits = self.avg_pool(b1_ft_cams).view(b, -1)
    
    # Similar for branch2 and branch3
    
    # 3. Gating network
    gate_l5 = self.gate_l5(self.gate_l4(self.gate_l3(x)))
    gate_pool = self.avg_pool(gate_l5).view(b, -1)
    gate_pred = F.softmax(self.gate_cls(gate_pool)/self.gate_temp, dim=1)
    
    # 4. Adaptive fusion
    gate_logits = torch.stack([b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1)
    gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
    gate_logits = gate_logits.sum(-1)
```

## Core Components

### 1. Backbone Network (`ResNet`)

Located in `core/net.py`, the backbone is a custom ResNet implementation:

```python
class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], 
                 avg_output=False, output_dim=-1, resprestride=1, 
                 res1ststride=1, res2ndstride=1, inchan=3):
        # Configurable architecture for different datasets
        self.conv1 = nn.Sequential(...)  # Initial conv layer
        self.conv2_x = self._make_layer(block, 64, num_block[0], res1ststride)
        self.conv3_x = self._make_layer(block, 128, num_block[1], res2ndstride)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
```

**Key Features:**

- Configurable stride parameters for different input sizes
- Flexible output dimension for feature projection
- Support for different ResNet variants (ResNet18, ResNet34)

### 2. Attention Diversity Regularization

Implemented in `core/train.py`:

```python
def attnDiv(cams):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    orthogonal_loss = 0
    bs = cams.shape[0]
    num_part = cams.shape[1]  # Number of experts (3)
    cams = cams.view(bs, num_part, -1)
    cams = F.normalize(cams, p=2, dim=-1)
    
    # Remove mean activation
    mean = cams.mean(dim=-1).view(bs, num_part, -1).expand(size=[bs, num_part, cams.shape[-1]])
    cams = F.relu(cams-mean)
    
    # Compute cosine similarity between all expert pairs
    for i in range(cams.shape[1]):
        for j in range(i+1, cams.shape[1]):
            orthogonal_loss += cos(cams[:,i,:].view(bs,1,-1), 
                                 cams[:,j,:].view(bs,1,-1)).mean()
    return orthogonal_loss/(i*(i-1)/2)
```

**Purpose:** Ensures that different experts learn to focus on different spatial regions of the input, promoting diverse representations.

### 3. Gating Network

The gating network learns to adaptively combine expert predictions:

```python
self.gate_cls = nn.Sequential(
    Classifier(feature_dim, int(feature_dim/4), bias=True),  # Bottleneck
    Classifier(int(feature_dim/4), 3, bias=True)             # 3 experts
)
```

**Temperature Scaling:** The gating predictions are scaled by `gate_temp` (default: 100) to control the sharpness of the softmax distribution.

## Training Process

### 1. Loss Function Components

The training loss consists of multiple components:

```python
# In core/train.py
loss_values = [
    criterion['entropy'](logit.float(), target.long()) for logit in logits  # Classification loss for each expert
]
loss_values.append(attnDiv(branch_cams))  # Attention diversity loss
loss_values.append(
    args['loss_wgts'][0] * sum(loss_values[:3]) +      # Classification loss weight
    args['loss_wgts'][1] * loss_values[-2] +           # Gating loss weight  
    args['loss_wgts'][2] * loss_values[-1]             # Diversity loss weight
)
```

**Default Weights:** `[0.7, 1.0, 0.01]` - Classification, Gating, Diversity

### 2. Training Loop

Located in `osr_main.py`:

```python
def trainLoop(options):
    # 1. Data loading
    train_loader, test_loader, out_loader = getLoader(options)
    
    # 2. Model initialization
    model = get_model(options)
    model = nn.DataParallel(model).cuda()
    
    # 3. Optimizer setup with different learning rates
    extractor_params = model.module.get_params(prefix='extractor')
    classifier_params = model.module.get_params(prefix='classifier')
    params = [
        {'params': classifier_params, 'lr': lr_cls},
        {'params': extractor_params, 'lr': lr_extractor}
    ]
    
    # 4. Training epochs
    for epoch in range(epoch_start, options['epoch_num']):
        train_loss = train(train_loader, model, criterion, optimizer, args=options)
        if (epoch + 1) % options['test_step'] == 0:
            result_list = evaluation(model, test_loader, out_loader, **options)
        scheduler.step()
```

## Configuration System

### 1. Configuration File (`misc/osr.yml`)

```yaml
# Training parameters
loss_wgts: [0.7, 1.0, 0.01]    # [classification, gating, diversity]
score_wgts: [1, 0, 0]          # [msp, mls, feature_norm]
branch_opt: -1                  # Which branch to use for evaluation (-1 = gated)
gate_temp: 100                  # Temperature for gating softmax

# Dataset and model
dataset: "tiny_imagenet"
backbone: "resnet18"
batch_size: 128
epoch_num: 150

# Optimization
optimizer: "SGD"
lr: 0.1
gamma: 0.1
milestones: [130]
lgs_temp: 100                   # Temperature for logit scaling
```

### 2. Parameter Management (`misc/param.py`)

The configuration system supports:

- YAML file loading
- Command-line argument overrides
- Environment variable support
- Automatic parameter merging

## Dataset Handling

### 1. Dataset Classes

Located in `datasets/osr_loader.py`:

```python
class CIFAR10_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, 
                 batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))
        
        # Filter datasets for known/unknown classes
        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
```

### 2. Data Splits

Predefined splits in `misc/util.py`:

```python
splits_AUROC = {
    'cifar10': [
        [0, 1, 2, 4, 5, 9],  # Known classes for split 1
        [0, 3, 5, 7, 8, 9],  # Known classes for split 2
        # ... more splits
    ]
}
```

### 3. Data Augmentation

Implemented in `datasets/tools.py`:

```python
def predata(img_size):
    trans = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=img_size, padding=int(img_size*0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),  # RandAugment
        transforms.ToTensor(),
        simple_norm,
        manual_ctr  # Contrast adjustment
    ]
    return transforms.Compose(trans)
```

## Evaluation Metrics

### 1. OSR Metrics

Implemented in `core/test.py`:

```python
def evaluation(model, test_loader, out_loader, **options):
    # 1. Closed set accuracy
    correct = (pred_label == labels.data).sum()
    acc = float(correct) * 100. / float(total)
    
    # 2. Open set detection
    fpr, tpr, thresholds = roc_curve(open_labels, prob)
    auroc = auc(fpr, tpr)
    
    # 3. Precision-Recall curves
    precision, recall, _ = precision_recall_curve(open_labels, prob)
    aupr_in = auc(recall, precision)
    
    # 4. Macro F1-score
    macro_f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')
```

### 2. Score Computation

```python
def compute_score(logit_list, softmax_list, score_wgts, branch_opt, fts=None):
    msp = softmax_list[branch_opt].max(1)[0]      # Maximum softmax probability
    mls = logit_list[branch_opt].max(1)[0]        # Maximum logit score
    if score_wgts[2] != 0:
        ftl = fts.mean(dim=[2,3]).norm(dim=1, p=2)  # Feature norm
        temp = (score_wgts[0]*msp + score_wgts[1]*mls + score_wgts[2]*ftl)
    else:
        temp = (score_wgts[0]*msp + score_wgts[1]*mls)
    return temp
```

## Modification Guide

### 1. Adding New Experts

To add a fourth expert:

```python
# In MultiBranchNet.__init__()
self.branch4_l4 = copy.deepcopy(self.branch1_l4)
self.branch4_l5 = copy.deepcopy(self.branch1_l5)
self.branch4_cls = conv1x1(feature_dim, self.num_known)

# In forward()
branch4_l4 = self.branch4_l4(ft_till_l3.clone())
branch4_l5 = self.branch4_l5(branch4_l4)
b4_ft_cams = self.branch4_cls(branch4_l5)
b4_logits = self.avg_pool(b4_ft_cams).view(b, -1)

# Update gating network output dimension
self.gate_cls = nn.Sequential(
    Classifier(feature_dim, int(feature_dim/4), bias=True),
    Classifier(int(feature_dim/4), 4, bias=True)  # 4 experts
)
```

### 2. Changing Backbone Architecture

```python
# In build_backbone() function
def build_backbone(img_size, backbone_name, projection_dim, inchan=3):
    if backbone_name == 'resnet50':
        backbone = ResNet(output_dim=projection_dim, inchan=inchan, 
                         num_block=[3,4,6,3], resprestride=1, 
                         res1ststride=2, res2ndstride=2)
        cam_size = int(img_size / 32)
    # Add more backbone options
```

### 3. Modifying Attention Diversity

```python
# Alternative diversity measures in attnDiv()
def attnDiv_alternative(cams):
    # L2 distance between attention maps
    bs, num_part = cams.shape[0], cams.shape[1]
    cams = cams.view(bs, num_part, -1)
    
    diversity_loss = 0
    for i in range(num_part):
        for j in range(i+1, num_part):
            dist = torch.norm(cams[:,i,:] - cams[:,j,:], p=2, dim=1)
            diversity_loss += dist.mean()
    
    return diversity_loss / (num_part * (num_part - 1) / 2)
```

### 4. Custom Fusion Strategies

```python
# Replace adaptive fusion with different strategies
def weighted_fusion(logits_list, weights):
    # Fixed weight fusion
    fused_logits = sum(w * logits for w, logits in zip(weights, logits_list))
    return fused_logits

def attention_fusion(logits_list, attention_weights):
    # Attention-based fusion
    logits_stack = torch.stack(logits_list, dim=-1)
    fused_logits = (logits_stack * attention_weights.unsqueeze(1)).sum(-1)
    return fused_logits
```

## File Structure

```
MEDAF/
├── core/                    # Core model and training code
│   ├── __init__.py         # Model imports
│   ├── net.py              # Network architectures
│   ├── train.py            # Training functions
│   └── test.py             # Evaluation functions
├── datasets/               # Data loading and preprocessing
│   ├── osr_loader.py       # Dataset classes
│   └── tools.py            # Data augmentation
├── misc/                   # Utilities and configuration
│   ├── __init__.py
│   ├── osr.yml            # Configuration file
│   ├── param.py           # Parameter management
│   └── util.py            # Utility functions
├── figs/                   # Figures and visualizations
├── osr_main.py            # Main training script
├── requirements.txt       # Dependencies
└── README.md             # Project overview
```

## Usage Examples

### 1. Basic Training

```bash
# Train on CIFAR10 with default settings
python osr_main.py -g 0 -d cifar10

# Train on Tiny ImageNet with custom parameters
python osr_main.py -g 0 -d tiny_imagenet -b 64 --epoch_num 200
```

### 2. Resume Training

```bash
# Resume from checkpoint
python osr_main.py -g 0 -d cifar10 -r -c ./ckpt/osr/cifar10/checkpoint.pth
```

### 3. Custom Configuration

```yaml
# Modify misc/osr.yml
loss_wgts: [0.8, 1.0, 0.05]    # Increase diversity weight
gate_temp: 50                   # Sharper gating
lgs_temp: 50                    # Sharper logit scaling
```

### 4. Adding New Datasets

```python
# In datasets/osr_loader.py
class CustomDataset_OSR(object):
    def __init__(self, known, dataroot, use_gpu=True, 
                 batch_size=128, img_size=32, options=None):
        # Implement dataset loading logic
        pass

# In osr_main.py getLoader()
elif 'custom_dataset' == options['dataset']:
    options['img_size'] = 64
    Data = CustomDataset_OSR(known=options['known'], 
                            batch_size=options['batch_size'], 
                            img_size=options['img_size'], 
                            options=options)
```

## Key Insights for Modification

1. **Expert Diversity**: The attention diversity loss is crucial for performance. Consider different diversity measures based on your use case.

2. **Gating Temperature**: Lower temperatures make the gating more decisive, while higher temperatures make it more uniform.

3. **Loss Weights**: The balance between classification, gating, and diversity losses significantly affects performance.

4. **Feature Extraction**: The shared layers (L1-L3) capture common features, while expert-specific layers (L4-L5) learn specialized representations.

5. **Evaluation Strategy**: The `branch_opt` parameter controls which expert's output is used for evaluation. `-1` uses the gated fusion.

This documentation provides a comprehensive understanding of the MEDAF implementation, enabling you to modify components effectively while maintaining the core architectural principles.

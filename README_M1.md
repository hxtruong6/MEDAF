# MEDAF for macOS M1

This document provides instructions for running the MEDAF (Multi-Expert Diverse Attention Fusion) project on macOS M1 with Apple Silicon.

## Prerequisites

### 1. Install PyTorch for macOS M1

Make sure you have PyTorch installed with MPS (Metal Performance Shaders) support:

```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 2. Install other dependencies

```bash
pip install -r requirements.txt
```

## Changes Made for macOS M1 Compatibility

The following changes were made to adapt the original CUDA-based code for macOS M1:

### 1. Device Detection

- Added automatic device detection that prioritizes:
  1. CUDA (if available)
  2. MPS (Apple Silicon GPU)
  3. CPU (fallback)

### 2. Removed CUDA-specific Code

- Replaced all `.cuda()` calls with `.to(device)`
- Removed `nn.DataParallel` (not needed for single device)
- Made CUDA environment variables conditional

### 3. Updated Configuration

- Reduced batch size from 128 to 32 (for memory efficiency)
- Set `num_workers` to 0 (better compatibility on macOS)
- Changed default dataset to `cifar10` (smaller, faster for testing)

## Running the Code

### 1. Test Device Detection

First, test that your PyTorch setup works correctly:

```bash
python test_device.py
```

You should see output indicating that MPS is available and basic tensor operations work.

### 2. Run Training

```bash
python osr_main.py
```

The script will automatically detect your device and use MPS if available.

## Configuration

The main configuration file is `misc/osr.yml`. Key settings for M1:

```yaml
dataset: "cifar10"        # Smaller dataset for testing
batch_size: 32           # Reduced for memory efficiency
num_workers: 0           # Better macOS compatibility
```

## Performance Notes

- **MPS Performance**: Apple Silicon GPU should provide good performance, though not as fast as CUDA
- **Memory**: M1 has unified memory, so monitor memory usage
- **Batch Size**: Start with 32, increase if memory allows
- **Dataset**: CIFAR-10 is recommended for initial testing

## Troubleshooting

### Common Issues

1. **MPS not available**: Update PyTorch to latest version
2. **Memory errors**: Reduce batch size further
3. **Slow performance**: Ensure you're using MPS, not CPU

### Debugging

Run the device test script to verify setup:

```bash
python test_device.py
```

## Original vs Adapted Code

| Original (CUDA) | Adapted (M1) |
|----------------|---------------|
| `model.cuda()` | `model.to(device)` |
| `nn.DataParallel(model)` | `model` (single device) |
| `data.cuda()` | `data.to(device)` |
| Fixed batch size 128 | Configurable batch size |
| CUDA-specific optimizations | Device-agnostic |

## Files Modified

- `osr_main.py`: Main training script with device detection
- `core/train.py`: Training loop with device parameter
- `core/test.py`: Evaluation with device parameter
- `misc/osr.yml`: Configuration for M1 compatibility
- `test_device.py`: Device detection test script

## Performance Comparison

Expected performance on M1:

- **Training**: ~2-5x slower than CUDA, but much faster than CPU
- **Memory**: More efficient than CUDA due to unified memory
- **Compatibility**: Better than CPU-only, not as optimized as CUDA

For production use, consider:

- Using a smaller model architecture
- Reducing batch size further if needed
- Running on CPU if MPS causes issues

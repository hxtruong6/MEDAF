#!/usr/bin/env python3
"""
Test script to verify device detection and PyTorch setup for macOS M1
"""

import torch
import sys


def test_device_detection():
    """Test device detection function"""
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if hasattr(torch.backends, "mps"):
        print("MPS available:", torch.backends.mps.is_available())
    else:
        print("MPS not available (PyTorch version too old)")

    # Test device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    print(f"Selected device: {device}")

    # Test basic tensor operations
    try:
        x = torch.randn(3, 3).to(device)
        y = torch.randn(3, 3).to(device)
        z = torch.mm(x, y)
        print("✓ Basic tensor operations work on", device)
        print("  Sample result:", z[0, 0].item())
    except Exception as e:
        print("✗ Error with tensor operations:", e)
        return False

    return True


if __name__ == "__main__":
    print("Testing PyTorch setup for macOS M1...")
    success = test_device_detection()

    if success:
        print("\n✓ Device detection test passed!")
        print("You can now run the main training script.")
    else:
        print("\n✗ Device detection test failed!")
        print("Please check your PyTorch installation.")
        sys.exit(1)

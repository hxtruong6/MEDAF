#!/usr/bin/env python3
"""
Chest X-Ray Multi-Label Dataset Loader using TorchXRayVision
Real medical data only - no synthetic fallbacks
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

try:
    import torchxrayvision as xrv
    import skimage

    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False
    raise ImportError(
        "TorchXRayVision is required. Install with: pip install torchxrayvision"
    )


class ChestXRayMultiLabel(data.Dataset):
    """
    Chest X-Ray Multi-Label Dataset using TorchXRayVision

    Features:
    - Real medical data with multiple pathologies per image
    - NIH ChestX-ray14, CheXpert, and MIMIC-CXR datasets supported
    - 18 pathology classes (multi-label classification)
    - Automatic train/validation splits
    """

    def __init__(
        self,
        split="train",
        transform=None,
        dataset_name="nih",
        data_root="./datasets/data/chestxray",
        max_samples=None,
        train_ratio=0.8,
    ):

        if not XRV_AVAILABLE:
            raise ImportError(
                "TorchXRayVision is required but not installed. "
                "Install with: pip install torchxrayvision"
            )

        self.split = split
        self.transform = transform
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.train_ratio = train_ratio

        # Standard pathologies (mapped from TorchXRayVision)
        self.pathologies = [
            "Atelectasis",
            "Consolidation",
            "Infiltration",
            "Pneumothorax",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Effusion",
            "Pneumonia",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Nodule",
            "Mass",
            "Hernia",
            "Lung Lesion",
            "Fracture",
            "Lung Opacity",
            "Enlarged Cardiomediastinum",
        ]
        self.num_classes = len(self.pathologies)

        print(f"Loading Chest X-Ray dataset: {dataset_name} ({split} split)")
        print(f"Pathologies: {self.num_classes} classes")
        print(f"Data root: {self.data_root}")

        # Load and process dataset
        self.dataset = self._load_xrv_dataset(dataset_name)
        self.images = []
        self.labels = []

        # Process and split dataset
        self._process_dataset(max_samples)

        print(f"Loaded {len(self.images)} {split} samples")
        if len(self.labels) > 0:
            avg_labels = np.mean([label.sum() for label in self.labels])
            print(f"Average pathologies per image: {avg_labels:.2f}")

    def _load_xrv_dataset(self, dataset_name):
        """Load TorchXRayVision dataset"""

        print(f"Initializing {dataset_name.upper()} dataset...")

        if dataset_name.lower() == "nih":
            # NIH ChestX-ray14 Dataset
            imgpath = str(self.data_root / "NIH" / "images")
            csvpath = str(self.data_root / "NIH" / "Data_Entry_2017.csv")

            # Check if data exists
            if not os.path.exists(imgpath):
                raise FileNotFoundError(
                    f"NIH images directory not found: {imgpath}\n"
                    f"Please download NIH ChestX-ray14 dataset first.\n"
                    f"Run: python download_real_chestxray.py"
                )

            if not os.path.exists(csvpath):
                raise FileNotFoundError(
                    f"NIH metadata CSV not found: {csvpath}\n"
                    f"Please download NIH ChestX-ray14 dataset first.\n"
                    f"Run: python download_real_chestxray.py"
                )

            dataset = xrv.datasets.NIH_Dataset(
                imgpath=imgpath,
                csvpath=csvpath,
                views=["PA", "AP"],  # Posterior-Anterior and Anterior-Posterior views
            )

        elif dataset_name.lower() == "chexpert":
            # CheXpert Dataset
            chexpert_path = str(self.data_root / "CheXpert")

            if not os.path.exists(chexpert_path):
                raise FileNotFoundError(
                    f"CheXpert directory not found: {chexpert_path}\n"
                    f"Please download CheXpert dataset manually from:\n"
                    f"https://stanfordmlgroup.github.io/competitions/chexpert/"
                )

            dataset = xrv.datasets.CheX_Dataset(
                imgpath=chexpert_path, views=["PA", "AP"]
            )

        elif dataset_name.lower() == "mimic":
            # MIMIC-CXR Dataset
            mimic_path = str(self.data_root / "MIMIC")

            if not os.path.exists(mimic_path):
                raise FileNotFoundError(
                    f"MIMIC-CXR directory not found: {mimic_path}\n"
                    f"Please download MIMIC-CXR dataset from PhysioNet"
                )

            dataset = xrv.datasets.MIMIC_Dataset(
                imgpath=mimic_path,
                csvpath=str(self.data_root / "MIMIC" / "mimic-cxr-2.0.0-metadata.csv"),
                metacsvpath=str(
                    self.data_root / "MIMIC" / "mimic-cxr-2.0.0-chexpert.csv"
                ),
                views=["PA", "AP"],
            )

        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: 'nih', 'chexpert', 'mimic'"
            )

        print(f"Dataset initialized: {len(dataset)} total samples")
        return dataset

    def _process_dataset(self, max_samples):
        """Process XRV dataset and create train/test splits"""

        print("Processing dataset...")

        # Determine sample size
        total_samples = len(self.dataset)
        if max_samples:
            total_samples = min(total_samples, max_samples)

        # Create train/test split
        indices = np.random.RandomState(42).permutation(len(self.dataset))[
            :total_samples
        ]
        split_idx = int(len(indices) * self.train_ratio)

        if self.split == "train":
            selected_indices = indices[:split_idx]
        else:  # test
            selected_indices = indices[split_idx:]

        print(f"Processing {len(selected_indices)} samples for {self.split} split...")

        processed = 0
        for i in selected_indices:
            try:
                sample = self.dataset[i]
                image = sample["img"]  # Already normalized by XRV
                label = sample["lab"]  # Multi-hot label vector

                # Convert to PIL Image for transforms
                if isinstance(image, np.ndarray):
                    # Handle grayscale
                    if image.ndim == 2:
                        image = np.stack([image] * 3, axis=-1)  # Convert to RGB

                    # Convert to uint8
                    if image.dtype != np.uint8:
                        image = (
                            (image - image.min()) / (image.max() - image.min()) * 255
                        ).astype(np.uint8)

                    image = Image.fromarray(image)

                # Map XRV labels to our pathology set
                our_label = self._map_labels(label)

                # Ensure at least one positive label for multi-label training
                if our_label.sum() == 0:
                    # Use most common pathology if no labels
                    our_label[0] = 1  # Atelectasis (common)

                self.images.append(image)
                self.labels.append(our_label)
                processed += 1

                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(selected_indices)} samples...")

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        print(f"Successfully processed {len(self.images)} samples")

    def _map_labels(self, xrv_labels):
        """Map TorchXRayVision labels to our pathology classes"""

        our_label = np.zeros(self.num_classes, dtype=np.float32)

        # Get pathology mapping from dataset
        if hasattr(self.dataset, "pathologies"):
            dataset_pathologies = self.dataset.pathologies

            for i, xrv_pathology in enumerate(dataset_pathologies):
                if i < len(xrv_labels):
                    # Map to our pathology list
                    if xrv_pathology in self.pathologies:
                        our_idx = self.pathologies.index(xrv_pathology)
                        our_label[our_idx] = float(xrv_labels[i])

                    # Handle common variations
                    pathology_mapping = {
                        "Pleural Thickening": "Pleural_Thickening",
                        "No Finding": None,  # Skip
                        "Consolidation": "Consolidation",
                        "Infiltration": "Infiltration",
                    }

                    mapped_name = pathology_mapping.get(xrv_pathology, xrv_pathology)
                    if mapped_name and mapped_name in self.pathologies:
                        our_idx = self.pathologies.index(mapped_name)
                        our_label[our_idx] = float(xrv_labels[i])
        else:
            # Fallback: use first N labels
            n = min(len(xrv_labels), self.num_classes)
            our_label[:n] = xrv_labels[:n]

        return our_label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_chestxray_transforms(img_size=224, is_training=True):
    """Get transforms for chest X-ray data"""

    if is_training:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return transform


def create_chestxray_dataloaders(
    dataset_name="nih",
    data_root="./datasets/data/chestxray",
    batch_size=32,
    img_size=224,
    num_workers=4,
    max_samples=None,
    train_ratio=0.8,
):
    """
    Create train/test dataloaders for chest X-ray datasets

    Args:
        dataset_name: 'nih', 'chexpert', or 'mimic'
        data_root: Root directory for dataset
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of worker processes
        max_samples: Maximum samples total (split between train/test)
        train_ratio: Ratio of samples for training

    Returns:
        train_loader, test_loader, num_classes, pathologies
    """

    # Get transforms
    train_transform = get_chestxray_transforms(img_size, is_training=True)
    test_transform = get_chestxray_transforms(img_size, is_training=False)

    # Create datasets
    train_dataset = ChestXRayMultiLabel(
        split="train",
        transform=train_transform,
        dataset_name=dataset_name,
        data_root=data_root,
        max_samples=max_samples,
        train_ratio=train_ratio,
    )

    test_dataset = ChestXRayMultiLabel(
        split="test",
        transform=test_transform,
        dataset_name=dataset_name,
        data_root=data_root,
        max_samples=max_samples,
        train_ratio=train_ratio,
    )

    num_classes = train_dataset.num_classes
    pathologies = train_dataset.pathologies

    # Create dataloaders
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\nChest X-Ray Dataset: {dataset_name.upper()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of pathologies: {num_classes}")
    print(f"Batch size: {batch_size}")
    print(f"Pathologies: {pathologies}")

    return train_loader, test_loader, num_classes, pathologies


def check_dataset_availability(
    dataset_name="nih", data_root="./datasets/data/chestxray"
):
    """Check if a dataset is available for use"""

    data_root = Path(data_root)

    if dataset_name.lower() == "nih":
        imgpath = data_root / "NIH" / "images"
        csvpath = data_root / "NIH" / "Data_Entry_2017.csv"

        if imgpath.exists() and csvpath.exists():
            num_images = len(list(imgpath.glob("*.png")))
            return True, f"NIH dataset available: {num_images} images"
        else:
            return (
                False,
                "NIH dataset not found. Run: python download_real_chestxray.py",
            )

    elif dataset_name.lower() == "chexpert":
        chexpert_path = data_root / "CheXpert"

        if chexpert_path.exists():
            return True, "CheXpert dataset available"
        else:
            return False, "CheXpert dataset not found. Download manually from Stanford."

    elif dataset_name.lower() == "mimic":
        mimic_path = data_root / "MIMIC"

        if mimic_path.exists():
            return True, "MIMIC-CXR dataset available"
        else:
            return False, "MIMIC-CXR dataset not found. Download from PhysioNet."

    else:
        return False, f"Unknown dataset: {dataset_name}"


if __name__ == "__main__":
    # Test chest X-ray dataset loading
    print("Testing Chest X-Ray Multi-Label Dataset (Real Data Only)")
    print("=" * 60)

    # Check TorchXRayVision installation
    if not XRV_AVAILABLE:
        print("❌ TorchXRayVision not installed")
        print("Install with: pip install torchxrayvision")
        sys.exit(1)

    # Check dataset availability
    available, message = check_dataset_availability("nih")
    print(f"Dataset check: {message}")

    if not available:
        print("\n❌ Real dataset not available")
        print("Please download first:")
        print("  python download_real_chestxray.py")
        sys.exit(1)

    # Test dataset creation
    try:
        train_loader, test_loader, num_classes, pathologies = (
            create_chestxray_dataloaders(
                dataset_name="nih",
                batch_size=8,
                img_size=224,
                num_workers=0,
                max_samples=100,  # Small for testing
            )
        )

        # Test a batch
        batch = next(iter(train_loader))
        images, labels = batch

        print(f"\n✅ Dataset loaded successfully (REAL DATA)")
        print(f"Sample batch: {images.shape}, {labels.shape}")
        print(f"Labels per sample: {labels.sum(dim=1).tolist()}")

        # Show sample pathologies
        first_labels = labels[0]
        positive_pathologies = [
            pathologies[i] for i, val in enumerate(first_labels) if val > 0.5
        ]
        print(f"Sample pathologies: {positive_pathologies}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. TorchXRayVision is installed: pip install torchxrayvision")
        print("2. NIH dataset is downloaded: python download_real_chestxray.py")

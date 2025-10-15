"""
PyTorch Lightning DataModule for MEDAF Multi-Label Classification
"""

import torch
import torch.utils.data as data
import pytorch_lightning as pl
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split

from test_multilabel_medaf import (
    ChestXrayKnownDataset,
    ChestXrayUnknownDataset,
    ChestXrayFullDataset,
)


class MEDAFDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MEDAF Multi-Label Classification

    This module handles:
    - Dataset loading and preprocessing
    - Train/validation/test splits
    - Data loaders creation
    - Class weight calculation
    - Stratified splitting for multi-label data
    """

    def __init__(
        self,
        train_csv: str,
        test_csv: str = None,
        image_root: str = None,
        img_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 1,
        val_ratio: float = 0.1,
        use_stratified_split: bool = True,
        max_samples: Optional[int] = None,
        pin_memory: bool = True,
        train_list: str = None,
        test_list: str = None,
        use_full_dataset: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Store configuration
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.image_root = image_root
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.use_stratified_split = use_stratified_split
        self.max_samples = max_samples
        self.pin_memory = pin_memory
        self.train_list = train_list
        self.test_list = test_list
        self.use_full_dataset = use_full_dataset

        # Will be set during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.pos_weight = None

    def prepare_data(self):
        """Download or prepare data if needed"""
        # This is called only on one GPU in distributed training
        # We don't need to download anything, but we can validate paths
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage"""
        fit_stage = stage == "fit" or stage is None
        if fit_stage and (self.train_dataset is None or self.val_dataset is None):
            # Load training dataset
            if self.use_full_dataset:
                self.train_dataset = ChestXrayFullDataset(
                    csv_path=self.train_csv,
                    image_root=self.image_root,
                    img_size=self.img_size,
                    max_samples=self.max_samples,
                    train_list=self.train_list,
                )
            else:
                self.train_dataset = ChestXrayKnownDataset(
                    csv_path=self.train_csv,
                    image_root=self.image_root,
                    img_size=self.img_size,
                    max_samples=self.max_samples,
                )

            # Create validation split
            if self.use_stratified_split:
                train_subset, val_subset = self._create_stratified_split(
                    self.train_dataset, self.val_ratio
                )
            else:
                # Use random split
                val_size = max(1, int(len(self.train_dataset) * self.val_ratio))
                train_size = len(self.train_dataset) - val_size

                train_subset, val_subset = data.random_split(
                    self.train_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )

            self.train_dataset = train_subset
            self.val_dataset = val_subset

            # Calculate class weights for training dataset
            self._calculate_class_weights()

        test_stage = stage == "test" or stage is None
        if test_stage and self.test_dataset is None:
            # Load test dataset
            if self.use_full_dataset:
                self.test_dataset = ChestXrayFullDataset(
                    csv_path=self.train_csv,  # Use same CSV for both train and test
                    image_root=self.image_root,
                    img_size=self.img_size,
                    max_samples=self.max_samples,
                    test_list=self.test_list,
                )
            else:
                self.test_dataset = ChestXrayKnownDataset(
                    csv_path=self.test_csv,
                    image_root=self.image_root,
                    img_size=self.img_size,
                    max_samples=self.max_samples,
                )

    def _extract_labels_for_stratification(self, dataset) -> np.ndarray:
        """
        Extract labels from dataset for stratification purposes.
        For multi-label data, we create a stratification strategy based on label combinations.
        """
        labels_list = []

        # Extract labels from all samples
        for i in range(len(dataset)):
            _, labels = dataset[i]
            # Convert to binary array and then to string for stratification
            label_str = "".join(map(str, labels.int().tolist()))
            labels_list.append(label_str)

        return np.array(labels_list)

    def _create_stratified_split(
        self, dataset, val_ratio: float, random_state: int = 42
    ) -> Tuple[data.Subset, data.Subset]:
        """
        Create stratified train-validation split for multi-label data.
        """
        try:
            # Extract labels for stratification
            labels_for_stratify = self._extract_labels_for_stratification(dataset)

            # Get indices
            indices = np.arange(len(dataset))

            # Use train_test_split with stratification
            train_indices, val_indices = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=random_state,
                stratify=labels_for_stratify,
            )

        except ValueError as e:
            # If stratification fails, fall back to random split
            print(f"Warning: Stratified split failed: {e}")
            print("Falling back to random split...")

            indices = np.arange(len(dataset))
            train_indices, val_indices = train_test_split(
                indices, test_size=val_ratio, random_state=random_state
            )

        # Create subsets
        train_subset = data.Subset(dataset, train_indices)
        val_subset = data.Subset(dataset, val_indices)

        return train_subset, val_subset

    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced multi-label classification"""
        # This is a simplified version - in practice, you might want to use
        # the more sophisticated class weight calculation from the original trainer
        if self.train_dataset is None:
            return

        try:
            # Collect all labels from training dataset
            all_labels = []
            for i in range(len(self.train_dataset)):
                _, labels = self.train_dataset[i]
                # Handle both tensor and numpy inputs
                if torch.is_tensor(labels):
                    all_labels.append(labels.cpu().float().numpy())
                else:
                    all_labels.append(labels)

            all_labels = np.array(all_labels, dtype=np.float32)

            # Calculate positive class frequencies
            pos_freq = all_labels.mean(axis=0)

            # Calculate inverse frequency weights
            # Avoid division by zero
            pos_freq = np.clip(pos_freq, 1e-7, 1.0)
            neg_freq = 1.0 - pos_freq
            neg_freq = np.clip(neg_freq, 1e-7, 1.0)

            # Calculate weights (positive class weight = neg_freq / pos_freq)
            pos_weight = neg_freq / pos_freq

            # Convert to tensor
            self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

            print("Class weights calculated:")
            for i, weight in enumerate(pos_weight):
                print(f"  Class {i}: {weight:.4f}")

        except Exception as e:
            print(f"Warning: Failed to calculate class weights: {e}")
            print("Continuing without class weights...")
            self.pos_weight = None

    def train_dataloader(self) -> data.DataLoader:
        """Create training data loader"""
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> data.DataLoader:
        """Create validation data loader"""
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> data.DataLoader:
        """Create test data loader"""
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get calculated class weights"""
        return self.pos_weight

    def create_unknown_dataloader(
        self, novelty_type: str = "all", max_samples: Optional[int] = None
    ) -> data.DataLoader:
        """
        Create data loader for unknown/novel samples for novelty detection evaluation.

        Args:
            novelty_type: Type of novelty samples to load ("all", "independent", "mixed", "known_only")
            max_samples: Maximum number of samples (None = all)

        Returns:
            DataLoader for unknown samples
        """
        # Use the same CSV file for both known and unknown datasets in full dataset mode
        csv_path = self.test_csv if self.test_csv is not None else self.train_csv

        unknown_dataset = ChestXrayUnknownDataset(
            csv_path=csv_path,
            image_root=self.image_root,
            img_size=self.img_size,
            max_samples=max_samples,
            novelty_type=novelty_type,
        )

        def custom_collate_fn(batch):
            """Custom collate function to handle 3-item returns from ChestXrayUnknownDataset"""
            # Separate the batch into images, labels, and metadata
            images = []
            labels = []
            metadata_list = []

            for item in batch:
                if len(item) == 3:
                    image, label, metadata = item
                    images.append(image)
                    labels.append(label)
                    metadata_list.append(metadata)
                else:
                    # Fallback for 2-item returns
                    image, label = item
                    images.append(image)
                    labels.append(label)
                    metadata_list.append({})

            # Stack images and labels
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)

            return images, labels

        return data.DataLoader(
            unknown_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues with custom collate
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=False,
            collate_fn=custom_collate_fn,
        )

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the datasets"""
        info = {}

        if self.train_dataset is not None:
            info["train_size"] = len(self.train_dataset)
        if self.val_dataset is not None:
            info["val_size"] = len(self.val_dataset)
        if self.test_dataset is not None:
            info["test_size"] = len(self.test_dataset)

        info["batch_size"] = self.batch_size
        info["num_workers"] = self.num_workers
        info["img_size"] = self.img_size

        return info

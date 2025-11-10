import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from typing import Tuple, Optional, List
from .augmentation import ImageAugmenter


class ImageDataset(Dataset):
    """
    PyTorch dataset for image classification
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            image_size: Image size for resizing
            augment: Whether to apply augmentation
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Initialize augmenter
        self.augmenter = ImageAugmenter(image_size=image_size)
        
        # Get all image paths and labels
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
        
        print(f"📊 Dataset loaded: {split}")
        print(f"   Images: {len(self.images)}")
        print(f"   Classes: {len(self.class_to_idx)}")
        print(f"   Augmentation: {self.augment}")
    
    def _load_dataset(self):
        """Load all image paths and labels"""
        # Get all category directories
        categories = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        # Create class mappings
        for idx, category_dir in enumerate(categories):
            category = category_dir.name
            self.class_to_idx[category] = idx
            self.idx_to_class[idx] = category
        
        # Load image paths
        for category_dir in categories:
            category = category_dir.name
            label = self.class_to_idx[category]
            
            # Get all images
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in category_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(label)
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Apply augmentation
        mode = 'train' if self.augment else 'val'
        image = self.augmenter.augment(image, mode=mode)
        
        # Convert to tensor (augmenter already normalized)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Get label
        label = self.labels[idx]
        
        return image_tensor, label
    
    def get_class_distribution(self) -> dict:
        """Get distribution of classes"""
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


class DataLoaderFactory:
    """
    Factory for creating dataloaders
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize factory
        
        Args:
            data_dir: Root data directory
            image_size: Image size
            batch_size: Batch size for training
            num_workers: Number of data loading workers
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print(f"🏭 DataLoader factory initialized")
        print(f"   Batch size: {batch_size}")
        print(f"   Num workers: {num_workers}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = ImageDataset(
            self.data_dir,
            split='train',
            image_size=self.image_size,
            augment=True
        )
        
        val_dataset = ImageDataset(
            self.data_dir,
            split='val',
            image_size=self.image_size,
            augment=False
        )
        
        test_dataset = ImageDataset(
            self.data_dir,
            split='test',
            image_size=self.image_size,
            augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        print(f"\n✅ DataLoaders created!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader, train_dataset.class_to_idx
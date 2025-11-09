import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt


class ImageAugmenter:
    """
    Image augmentation for training robustness
    """
    
    def __init__(self, image_size: int = 224):
        """
        Initialize augmenter
        
        Args:
            image_size: Target image size
        """
        self.image_size = image_size
        
        # Training augmentations
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
            ], p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Validation/test augmentations (no augmentation, just resize)
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"🎨 Image augmenter initialized")
        print(f"   Image size: {image_size}x{image_size}")
    
    def augment(
        self,
        image: np.ndarray,
        mode: str = 'train'
    ) -> np.ndarray:
        """
        Apply augmentation
        
        Args:
            image: Input image (BGR format)
            mode: 'train' or 'val'
            
        Returns:
            Augmented image
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform
        if mode == 'train':
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        
        return transformed['image']
    
    def visualize_augmentations(
        self,
        image_path: str,
        n_augmentations: int = 6,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize multiple augmentations
        
        Args:
            image_path: Path to image
            n_augmentations: Number of augmentations to show
            save_path: Where to save visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        cols = 3
        rows = (n_augmentations + 2) // cols  # +1 for original
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Augmented versions
        for i in range(1, n_augmentations + 1):
            # Apply training augmentation (without normalization for visualization)
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ], p=0.5),
            ])
            
            augmented = transform(image=image.copy())['image']
            
            axes[i].imshow(augmented)
            axes[i].set_title(f"Augmentation {i}", fontsize=12)
            axes[i].axis('off')
        
        # Hide remaining axes
        for i in range(n_augmentations + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        else:
            plt.savefig('data/augmentation_demo.png', dpi=150, bbox_inches='tight')
            print("✅ Visualization saved: data/augmentation_demo.png")
        
        plt.close()
    
    def get_augmentation_info(self) -> Dict:
        """Get information about augmentations"""
        return {
            'image_size': self.image_size,
            'train_augmentations': [
                'Resize',
                'HorizontalFlip',
                'RandomRotate90',
                'ShiftScaleRotate',
                'Noise/Blur',
                'Brightness/Contrast/Hue',
                'Normalization'
            ],
            'val_augmentations': [
                'Resize',
                'Normalization'
            ]
        }
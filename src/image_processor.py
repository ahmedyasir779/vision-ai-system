import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt


class ImageProcessor:
    """
    Core image processing operations
    """
    
    def __init__(self):
        """Initialize image processor"""
        print(" Image Processor initialized")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file
        
        Args:
            image_path: Path to image
            
        Returns:
            Image as numpy array (BGR format)
        """
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return image
    
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """Save image to file"""
        cv2.imwrite(output_path, image)
        print(f"✅ Image saved: {output_path}")
    
    def resize(
        self,
        image: np.ndarray,
        size: Tuple[int, int],
        keep_aspect_ratio: bool = False
    ) -> np.ndarray:
        """
        Resize image
        
        Args:
            image: Input image
            size: Target size (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = size
            
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Pad to target size
            delta_w = target_w - new_w
            delta_h = target_h - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            resized = cv2.resize(image, size)
        
        return resized
    
    def crop(
        self,
        image: np.ndarray,
        x: int, y: int,
        width: int, height: int
    ) -> np.ndarray:
        """
        Crop image
        
        Args:
            image: Input image
            x, y: Top-left corner
            width, height: Crop dimensions
            
        Returns:
            Cropped image
        """
        return image[y:y+height, x:x+width]
    
    def rotate(
        self,
        image: np.ndarray,
        angle: float,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Rotate image
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            scale: Scaling factor
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, matrix, (w, h))
        
        return rotated
    
    def flip(self, image: np.ndarray, direction: str = "horizontal") -> np.ndarray:
        """
        Flip image
        
        Args:
            image: Input image
            direction: 'horizontal', 'vertical', or 'both'
            
        Returns:
            Flipped image
        """
        if direction == "horizontal":
            return cv2.flip(image, 1)
        elif direction == "vertical":
            return cv2.flip(image, 0)
        elif direction == "both":
            return cv2.flip(image, -1)
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def adjust_brightness(
        self,
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input image
            factor: Brightness factor (1.0 = no change, <1 darker, >1 brighter)
            
        Returns:
            Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def adjust_contrast(
        self,
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """
        Adjust image contrast
        
        Args:
            image: Input image
            factor: Contrast factor (1.0 = no change)
            
        Returns:
            Adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)
    
    def convert_color(
        self,
        image: np.ndarray,
        conversion: str
    ) -> np.ndarray:
        """
        Convert image color space
        
        Args:
            image: Input image
            conversion: Color conversion ('bgr2rgb', 'bgr2gray', 'bgr2hsv', etc.)
            
        Returns:
            Converted image
        """
        conversions = {
            'bgr2rgb': cv2.COLOR_BGR2RGB,
            'rgb2bgr': cv2.COLOR_RGB2BGR,
            'bgr2gray': cv2.COLOR_BGR2GRAY,
            'gray2bgr': cv2.COLOR_GRAY2BGR,
            'bgr2hsv': cv2.COLOR_BGR2HSV,
            'hsv2bgr': cv2.COLOR_HSV2BGR,
        }
        
        if conversion not in conversions:
            raise ValueError(f"Unknown conversion: {conversion}")
        
        return cv2.cvtColor(image, conversions[conversion])
    
    def apply_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
        blur_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Apply blur to image
        
        Args:
            image: Input image
            kernel_size: Blur kernel size (must be odd)
            blur_type: 'gaussian', 'median', or 'average'
            
        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if blur_type == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif blur_type == "median":
            return cv2.medianBlur(image, kernel_size)
        elif blur_type == "average":
            return cv2.blur(image, (kernel_size, kernel_size))
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")
    
    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges using Canny algorithm
        
        Args:
            image: Input image
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge map
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        return edges
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """Get image statistics"""
        stats = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min": int(np.min(image)),
            "max": int(np.max(image)),
            "mean": float(np.mean(image)),
            "std": float(np.std(image))
        }
        
        if len(image.shape) == 3:
            stats["channels"] = image.shape[2]
            stats["size_mb"] = image.nbytes / (1024 * 1024)
        
        return stats
    
    def visualize_transformations(
        self,
        image: np.ndarray,
        transformations: List[Tuple[str, dict]]
    ) -> None:
        """
        Visualize multiple transformations
        
        Args:
            image: Original image
            transformations: List of (method_name, kwargs) tuples
        """
        n_transforms = len(transformations) + 1
        cols = min(3, n_transforms)
        rows = (n_transforms + cols - 1) // cols
        
        plt.figure(figsize=(5 * cols, 5 * rows))
        
        # Original image
        plt.subplot(rows, cols, 1)
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(display_img)
        plt.title("Original")
        plt.axis('off')
        
        # Transformations
        for idx, (method_name, kwargs) in enumerate(transformations, 2):
            plt.subplot(rows, cols, idx)
            
            # Apply transformation
            method = getattr(self, method_name)
            transformed = method(image.copy(), **kwargs)
            
            # Display
            if len(transformed.shape) == 2:
                plt.imshow(transformed, cmap='gray')
            else:
                display_img = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
                plt.imshow(display_img)
            
            plt.title(f"{method_name}: {kwargs}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('data/transformations_demo.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved: data/transformations_demo.png")
        plt.close()
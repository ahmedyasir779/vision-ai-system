import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .augmentation import ImageAugmenter


class InferenceEngine:
    """
    Run inference on new images
    """
    
    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = 'cuda',
        image_size: int = 224
    ):
        """
        Initialize inference engine
        
        Args:
            model: Trained model
            class_names: List of class names
            device: Device to use
            image_size: Input image size
        """
        self.model = model.to(device)
        self.model.eval()
        
        self.class_names = class_names
        self.device = device
        self.image_size = image_size
        
        # Initialize augmenter (for preprocessing)
        self.augmenter = ImageAugmenter(image_size=image_size)
        
        print(f"🔮 Inference engine initialized")
        print(f"   Classes: {class_names}")
        print(f"   Device: {device}")
    
    def predict_single(
        self,
        image_path: str
    ) -> Dict:
        """
        Predict on single image
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess
        preprocessed = self.augmenter.augment(image, mode='val')
        
        # Convert to tensor
        image_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probs, 3)
        top3_classes = [self.class_names[idx] for idx in top3_indices[0].cpu().numpy()]
        top3_confidences = top3_probs[0].cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_value,
            'top3_classes': top3_classes,
            'top3_confidences': top3_confidences.tolist(),
            'all_probabilities': probs[0].cpu().numpy().tolist()
        }
    
    def predict_batch(
        self,
        image_paths: List[str]
    ) -> List[Dict]:
        """
        Predict on batch of images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def predict_and_visualize(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ):
        """
        Predict and visualize result
        
        Args:
            image_path: Path to image
            save_path: Where to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Predict
        result = self.predict_single(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Show image
        ax1.imshow(image_rgb)
        ax1.set_title(
            f"Prediction: {result['predicted_class']}\n"
            f"Confidence: {result['confidence']*100:.2f}%",
            fontsize=12,
            fontweight='bold'
        )
        ax1.axis('off')
        
        # Show probabilities
        classes = self.class_names
        probs = result['all_probabilities']
        
        bars = ax2.barh(classes, probs, color='skyblue')
        
        # Highlight predicted class
        predicted_idx = classes.index(result['predicted_class'])
        bars[predicted_idx].set_color('green')
        
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Class Probabilities', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add value labels
        for idx, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(
                width, bar.get_y() + bar.get_height()/2,
                f'{probs[idx]*100:.1f}%',
                ha='left', va='center', fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
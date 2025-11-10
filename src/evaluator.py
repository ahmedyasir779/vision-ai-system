import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class ModelEvaluator:
    """
    Evaluate trained model
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str],
        device: str = 'cuda'
    ):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            test_loader: Test dataloader
            class_names: List of class names
            device: Device to use
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        
        print(f"📊 Model evaluator initialized")
        print(f"   Classes: {len(class_names)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary with predictions and metrics
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\n🔍 Evaluating model...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        print(f"\n✅ Evaluation complete!")
        print(f"   Accuracy:  {accuracy*100:.2f}%")
        print(f"   Precision: {precision*100:.2f}%")
        print(f"   Recall:    {recall*100:.2f}%")
        print(f"   F1 Score:  {f1*100:.2f}%")
        
        return {
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = 'results/plots/confusion_matrix.png'
    ):
        """
        Plot confusion matrix
        
        Args:
            predictions: Predicted labels
            labels: True labels
            save_path: Where to save plot
        """
        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0]
        )
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1]
        )
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
        plt.close()
    
    def plot_per_class_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = 'results/plots/per_class_metrics.png'
    ):
        """
        Plot per-class metrics
        
        Args:
            predictions: Predicted labels
            labels: True labels
            save_path: Where to save plot
        """
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Create DataFrame for plotting
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8
                )
        
        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Per-class metrics saved: {save_path}")
        plt.close()
    
    def generate_classification_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        save_path: str = 'results/classification_report.txt'
    ):
        """
        Generate and save classification report
        
        Args:
            predictions: Predicted labels
            labels: True labels
            save_path: Where to save report
        """
        report = classification_report(
            labels,
            predictions,
            target_names=self.class_names,
            digits=4
        )
        
        # Print report
        print("\n" + "="*60)
        print("📋 CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        
        # Save report
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
        
        print(f"✅ Report saved: {save_path}")
    
    def plot_top_errors(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        n_errors: int = 5,
        save_path: str = 'results/plots/top_errors.png'
    ):
        """
        Plot images with highest prediction errors
        
        Args:
            predictions: Predicted labels
            labels: True labels
            probabilities: Prediction probabilities
            n_errors: Number of errors to show
            save_path: Where to save plot
        """
        # Find misclassified samples
        errors = predictions != labels
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            print("✅ No errors to plot!")
            return
        
        # Get confidence for wrong predictions
        error_confidences = []
        for idx in error_indices:
            conf = probabilities[idx][predictions[idx]]
            error_confidences.append(conf)
        
        # Sort by confidence (high confidence but wrong = worst errors)
        sorted_indices = np.argsort(error_confidences)[::-1]
        top_error_indices = error_indices[sorted_indices[:n_errors]]
        
        print(f"\n🔍 Found {len(error_indices)} misclassified samples")
        print(f"   Showing top {min(n_errors, len(error_indices))} confident errors")


def plot_training_history(
    history_path: str = 'logs/training_history.json',
    save_path: str = 'results/plots/training_history.png'
):
    """
    Plot training history
    
    Args:
        history_path: Path to training history JSON
        save_path: Where to save plot
    """
    import json
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history saved: {save_path}")
    plt.close()
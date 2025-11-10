import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification
    Good for learning fundamentals
    """
    
    def __init__(self, num_classes: int = 5):
        """
        Initialize simple CNN
        
        Args:
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class TransferLearningModel(nn.Module):
    """
    Transfer learning model using pre-trained ResNet
    Better performance with less training time
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize transfer learning model
        
        Args:
            num_classes: Number of output classes
            model_name: Pre-trained model to use
            pretrained: Whether to use pre-trained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super(TransferLearningModel, self).__init__()
        
        self.model_name = model_name
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze final layer
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)


class ModelFactory:
    """
    Factory for creating models
    """
    
    @staticmethod
    def create_model(
        model_type: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ) -> nn.Module:
        """
        Create model
        
        Args:
            model_type: 'simple' or 'resnet18/34/50'
            num_classes: Number of classes
            pretrained: Use pre-trained weights
            freeze_backbone: Freeze backbone
            
        Returns:
            Model
        """
        print(f"\n🏗️ Creating model: {model_type}")
        print(f"   Classes: {num_classes}")
        print(f"   Pretrained: {pretrained}")
        
        if model_type == 'simple':
            model = SimpleCNN(num_classes=num_classes)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            model = TransferLearningModel(
                num_classes=num_classes,
                model_name=model_type,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone
            )
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
        print(f"   Frozen params: {total_params - trainable_params:,}")
        
        return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset_loader import DataLoaderFactory
from src.cnn_model import ModelFactory, count_parameters
from src.trainer import Trainer


def test_dataset_loader():
    """Test dataset loading"""
    print("\n" + "="*60)
    print("🧪 TEST 1: DATASET LOADER")
    print("="*60 + "\n")
    
    factory = DataLoaderFactory(
        data_dir='data/processed',
        batch_size=8,
        num_workers=0  # 0 for testing
    )
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_to_idx = factory.create_dataloaders()
    
    print(f"\n✅ DataLoaders created!")
    print(f"   Classes: {class_to_idx}")
    
    # Test batch
    print("\n📦 Testing batch...")
    images, labels = next(iter(train_loader))
    print(f"   Batch images shape: {images.shape}")
    print(f"   Batch labels shape: {labels.shape}")
    print(f"   Image value range: [{images.min():.2f}, {images.max():.2f}]")
    print("   ✅ Batch loading works\n")
    
    print("="*60)
    print("✅ DATASET LOADER TEST PASSED!")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader, class_to_idx


def test_models(num_classes):
    """Test model creation"""
    print("\n" + "="*60)
    print("🧪 TEST 2: MODEL CREATION")
    print("="*60 + "\n")
    
    # Test simple CNN
    print("1. Testing Simple CNN...")
    simple_model = ModelFactory.create_model('simple', num_classes)
    params = count_parameters(simple_model)
    print(f"   Parameters: {params['total']:,}")
    print("   ✅ Simple CNN created\n")
    
    # Test transfer learning
    print("2. Testing Transfer Learning (ResNet18)...")
    transfer_model = ModelFactory.create_model(
        'resnet18',
        num_classes,
        pretrained=True
    )
    params = count_parameters(transfer_model)
    print(f"   Parameters: {params['total']:,}")
    print("   ✅ ResNet18 created\n")
    
    # Test forward pass
    print("3. Testing forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    output = simple_model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected classes: {num_classes}")
    assert output.shape == (2, num_classes), "Output shape mismatch!"
    print("   ✅ Forward pass works\n")
    
    print("="*60)
    print("✅ MODEL CREATION TEST PASSED!")
    print("="*60 + "\n")
    
    return simple_model


def test_training(model, train_loader, val_loader):
    """Test training loop"""
    print("\n" + "="*60)
    print("🧪 TEST 3: TRAINING")
    print("="*60 + "\n")
    
    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # Train for 2 epochs (quick test)
    print("\n🚀 Training for 2 epochs (test)...")
    history = trainer.train(num_epochs=2)
    
    print("\n📊 Training History:")
    print(f"   Epoch 1 - Train Acc: {history['train_acc'][0]:.2f}%, "
          f"Val Acc: {history['val_acc'][0]:.2f}%")
    print(f"   Epoch 2 - Train Acc: {history['train_acc'][1]:.2f}%, "
          f"Val Acc: {history['val_acc'][1]:.2f}%")
    
    # Save history
    trainer.save_history()
    
    print("\n   ✅ Training completed\n")
    
    print("="*60)
    print("✅ TRAINING TEST PASSED!")
    print("="*60 + "\n")


def main():
    print("\n🚀 CNN ARCHITECTURE & TRAINING TESTS")
    print("Testing dataset loading, model creation, and training\n")
    
    try:
        # Test 1: Dataset Loader
        train_loader, val_loader, test_loader, class_to_idx = test_dataset_loader()
        num_classes = len(class_to_idx)
        
        # Test 2: Model Creation
        model = test_models(num_classes)
        
        # Test 3: Training
        test_training(model, train_loader, val_loader)
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Dataset Loader: PASSED")
        print("✅ Model Creation: PASSED")
        print("✅ Training: PASSED")
        print("\n🎯 CNN training complete!")
        print("\n💡 Key Skills Learned:")
        print("  - PyTorch Dataset & DataLoader")
        print("  - CNN architecture design")
        print("  - Transfer learning with ResNet")
        print("  - Training loop with validation")
        print("  - Model checkpointing")
        print("  - Training history tracking")
        print("\n📊 Model Performance:")
        print("  - Simple CNN created")
        print("  - ResNet18 transfer learning")
        print("  - Training completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
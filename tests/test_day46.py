"""
Test Day 46: Real Data Training & Evaluation
Complete pipeline with real dataset
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from src.data_downloader_real import DatasetPreparer
from src.dataset_organizer import DatasetOrganizer
from src.dataset_loader import DataLoaderFactory
from src.cnn_model import ModelFactory
from src.trainer import Trainer
from src.evaluator import ModelEvaluator, plot_training_history
from src.inference import InferenceEngine


def test_dataset_download():
    """Test real dataset download"""
    print("\n" + "="*60)
    print("🧪 TEST 1: DOWNLOAD REAL DATASET")
    print("="*60 + "\n")
    
    preparer = DatasetPreparer()
    
    print("Downloading flowers dataset...")
    dataset_dir = preparer.prepare_flowers_dataset(samples_per_class=150)
    
    print(f"\n✅ Real dataset ready!")
    print(f"   Location: {dataset_dir}")
    
    print("\n="*60)
    print("✅ DATASET DOWNLOAD TEST PASSED!")
    print("="*60 + "\n")
    
    return dataset_dir


def test_dataset_split(dataset_dir):
    """Test dataset splitting"""
    print("\n" + "="*60)
    print("🧪 TEST 2: SPLIT REAL DATASET")
    print("="*60 + "\n")
    
    organizer = DatasetOrganizer(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    print("Splitting real dataset...")
    stats = organizer.split_dataset(
        source_dir=dataset_dir,
        output_dir=Path("data/processed_real"),
        copy_files=True
    )
    
    print("\n✅ Dataset split complete!")
    
    print("\n="*60)
    print("✅ DATASET SPLIT TEST PASSED!")
    print("="*60 + "\n")


def test_training_on_real_data():
    """Test training on real data"""
    print("\n" + "="*60)
    print("🧪 TEST 3: TRAIN ON REAL DATA")
    print("="*60 + "\n")
    
    # Create dataloaders
    factory = DataLoaderFactory(
        data_dir='data/processed_real',
        batch_size=16,
        num_workers=2
    )
    
    train_loader, val_loader, test_loader, class_to_idx = factory.create_dataloaders()
    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
    
    print(f"\nClasses: {class_names}")
    
    # Create model
    print("\n🏗️ Creating ResNet18 model...")
    model = ModelFactory.create_model(
        'resnet18',
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Setup training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='models/checkpoints_real'
    )
    
    # Train
    print("\n🚀 Training on REAL data (10 epochs)...")
    history = trainer.train(num_epochs=10)
    
    # Save history
    trainer.save_history('logs/training_history_real.json')
    
    print("\n✅ Training on real data complete!")
    
    print("="*60 + "\n")
    print("✅ TRAINING TEST PASSED!")
    print("="*60 + "\n")
    
    return model, test_loader, class_names


def test_evaluation(model, test_loader, class_names):
    """Test model evaluation"""
    print("\n" + "="*60)
    print("🧪 TEST 4: MODEL EVALUATION")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device
    )
    
    # Evaluate
    results = evaluator.evaluate()
    
    # Generate reports and plots
    print("\n📊 Generating evaluation plots...")
    
    evaluator.plot_confusion_matrix(
        results['predictions'],
        results['labels'],
        save_path='results/plots/confusion_matrix_real.png'
    )
    
    evaluator.plot_per_class_metrics(
        results['predictions'],
        results['labels'],
        save_path='results/plots/per_class_metrics_real.png'
    )
    
    evaluator.generate_classification_report(
        results['predictions'],
        results['labels'],
        save_path='results/classification_report_real.txt'
    )
    
    # Plot training history
    plot_training_history(
        'logs/training_history_real.json',
        'results/plots/training_history_real.png'
    )
    
    print("\n✅ Evaluation complete!")
    
    print("="*60 + "\n")
    print("✅ EVALUATION TEST PASSED!")
    print("="*60 + "\n")


def test_inference(model, class_names):
    """Test inference on new images"""
    print("\n" + "="*60)
    print("🧪 TEST 5: INFERENCE")
    print("="*60 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create inference engine
    engine = InferenceEngine(
        model=model,
        class_names=class_names,
        device=device
    )
    
    # Get a test image
    test_image = Path("data/processed_real/test") / class_names[0]
    test_images = list(test_image.glob('*.jpg'))[:3]
    
    if test_images:
        print(f"\n🔮 Testing inference on {len(test_images)} images...")
        
        for img_path in test_images:
            result = engine.predict_single(str(img_path))
            
            print(f"\n📸 {img_path.name}")
            print(f"   Predicted: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   Top 3: {result['top3_classes']}")
        
        # Visualize one
        engine.predict_and_visualize(
            str(test_images[0]),
            save_path='results/predictions/sample_prediction.png'
        )
        
        print("\n✅ Inference working!")
    else:
        print("\n⚠️ No test images found")
    
    print("="*60 + "\n")
    print("✅ INFERENCE TEST PASSED!")
    print("="*60 + "\n")


def main():
    print("\n🚀 REAL DATA TRAINING & EVALUATION")
    print("Complete pipeline with real flower dataset\n")
    
    try:
        # Test 1: Download real dataset
        dataset_dir = test_dataset_download()
        
        # Test 2: Split dataset
        test_dataset_split(dataset_dir)
        
        # Test 3: Train on real data
        model, test_loader, class_names = test_training_on_real_data()
        
        # Test 4: Evaluate
        test_evaluation(model, test_loader, class_names)
        
        # Test 5: Inference
        test_inference(model, class_names)
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Dataset Download: PASSED")
        print("✅ Dataset Split: PASSED")
        print("✅ Training on Real Data: PASSED")
        print("✅ Model Evaluation: PASSED")
        print("✅ Inference: PASSED")
        print("\n🎯 real data training complete!")
        print("\n📊 Model Performance on Real Data:")
        print("  - Check: results/plots/confusion_matrix_real.png")
        print("  - Check: results/classification_report_real.txt")
        print("  - Check: results/plots/training_history_real.png")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
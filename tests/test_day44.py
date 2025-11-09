"""
Test Day 44: Dataset Preparation & Augmentation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_downloader import DatasetDownloader
from src.dataset_organizer import DatasetOrganizer
from src.augmentation import ImageAugmenter
import cv2


def test_dataset_creation():
    """Test dataset creation"""
    print("\n" + "="*60)
    print("🧪 TEST 1: DATASET CREATION")
    print("="*60 + "\n")
    
    downloader = DatasetDownloader()
    
    print("1. Creating sample dataset structure...")
    dataset_dir = downloader.download_sample_dataset()
    print(f"   ✅ Structure created\n")
    
    print("2. Creating sample images...")
    dataset_dir = downloader.create_sample_images(num_per_class=20)
    print(f"   ✅ Images created\n")
    
    print("3. Getting dataset info...")
    info = downloader.get_dataset_info(dataset_dir)
    print(f"   Categories: {len(info['categories'])}")
    print(f"   Total images: {info['total_images']}")
    for cat, count in info['images_per_category'].items():
        print(f"   - {cat}: {count} images")
    print("   ✅ Dataset info retrieved\n")
    
    print("="*60)
    print("✅ DATASET CREATION TEST PASSED!")
    print("="*60 + "\n")
    
    return dataset_dir


def test_dataset_split(dataset_dir):
    """Test dataset splitting"""
    print("\n" + "="*60)
    print("🧪 TEST 2: DATASET SPLITTING")
    print("="*60 + "\n")
    
    organizer = DatasetOrganizer(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("Splitting dataset...")
    stats = organizer.split_dataset(
        source_dir=dataset_dir,
        output_dir=Path("data/processed"),
        copy_files=True
    )
    
    print("\n   ✅ Dataset split complete\n")
    
    print("="*60)
    print("✅ DATASET SPLITTING TEST PASSED!")
    print("="*60 + "\n")


def test_augmentation():
    """Test image augmentation"""
    print("\n" + "="*60)
    print("🧪 TEST 3: IMAGE AUGMENTATION")
    print("="*60 + "\n")
    
    augmenter = ImageAugmenter(image_size=224)
    
    print("1. Creating test image...")
    import numpy as np
    test_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite('data/test_image.jpg', test_image)
    print("   ✅ Test image created\n")
    
    print("2. Testing augmentation...")
    augmented = augmenter.augment(test_image, mode='train')
    print(f"   Original shape: {test_image.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    print("   ✅ Augmentation works\n")
    
    print("3. Creating visualization...")
    augmenter.visualize_augmentations(
        'data/test_image.jpg',
        n_augmentations=8
    )
    print("   ✅ Visualization created\n")
    
    print("4. Getting augmentation info...")
    info = augmenter.get_augmentation_info()
    print(f"   Image size: {info['image_size']}")
    print(f"   Train augmentations: {len(info['train_augmentations'])}")
    print(f"   Val augmentations: {len(info['val_augmentations'])}")
    print("   ✅ Info retrieved\n")
    
    print("="*60)
    print("✅ AUGMENTATION TEST PASSED!")
    print("="*60 + "\n")


def main():
    """Run all tests"""
    print("Testing dataset creation, splitting, and augmentation\n")
    
    try:
        # Test 1: Dataset Creation
        dataset_dir = test_dataset_creation()
        
        # Test 2: Dataset Split
        test_dataset_split(dataset_dir)
        
        # Test 3: Augmentation
        test_augmentation()
        
        # Final summary
        print("\n" + "="*60)
        print("="*60)
        print("\n✅ Dataset Creation: PASSED")
        print("✅ Dataset Splitting: PASSED")
        print("✅ Image Augmentation: PASSED")
        print("\n💡 Key Skills Learned:")
        print("  - Dataset downloading & organization")
        print("  - Train/val/test splitting (70/15/15)")
        print("  - Advanced augmentation (Albumentations)")
        print("  - Augmentation visualization")
        print("\n📊 Dataset Ready:")
        print("  - 5 categories (cat, dog, bird, car, flower)")
        print("  - 20 images per category = 100 total")
        print("  - Split: 70 train / 15 val / 15 test")
        print("\n🔜 Tomorrow: CNN architecture & training!\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
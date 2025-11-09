import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_processor import ImageProcessor
import cv2
import numpy as np


def create_test_image():
    """Create a test image"""
    # Create colorful test image (300x300)
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Red square
    image[50:150, 50:150] = [0, 0, 255]
    
    # Green circle
    cv2.circle(image, (225, 100), 50, (0, 255, 0), -1)
    
    # Blue triangle
    pts = np.array([[225, 200], [175, 250], [275, 250]], np.int32)
    cv2.fillPoly(image, [pts], (255, 0, 0))
    
    return image


def test_basic_operations():
    """Test basic image operations"""
    print("\n" + "="*60)
    print("🧪 TEST 1: BASIC OPERATIONS")
    print("="*60 + "\n")
    
    processor = ImageProcessor()
    image = create_test_image()
    
    print("1. Test image creation...")
    print(f"   Shape: {image.shape}")
    print(f"   Type: {image.dtype}")
    print("   ✅ Image created\n")
    
    print("2. Test resize...")
    resized = processor.resize(image, (150, 150))
    print(f"   Original: {image.shape}")
    print(f"   Resized: {resized.shape}")
    print("   ✅ Resize works\n")
    
    print("3. Test crop...")
    cropped = processor.crop(image, 50, 50, 100, 100)
    print(f"   Cropped shape: {cropped.shape}")
    print("   ✅ Crop works\n")
    
    print("4. Test rotate...")
    rotated = processor.rotate(image, 45)
    print(f"   Rotated shape: {rotated.shape}")
    print("   ✅ Rotate works\n")
    
    print("5. Test flip...")
    flipped_h = processor.flip(image, "horizontal")
    flipped_v = processor.flip(image, "vertical")
    print(f"   Horizontal flip: {flipped_h.shape}")
    print(f"   Vertical flip: {flipped_v.shape}")
    print("   ✅ Flip works\n")
    
    print("="*60)
    print("✅ BASIC OPERATIONS TEST PASSED!")
    print("="*60 + "\n")


def test_color_operations():
    """Test color and intensity operations"""
    print("\n" + "="*60)
    print("🧪 TEST 2: COLOR OPERATIONS")
    print("="*60 + "\n")
    
    processor = ImageProcessor()
    image = create_test_image()
    
    print("1. Test brightness adjustment...")
    brighter = processor.adjust_brightness(image, 1.5)
    darker = processor.adjust_brightness(image, 0.5)
    print("   ✅ Brightness adjustment works\n")
    
    print("2. Test contrast adjustment...")
    high_contrast = processor.adjust_contrast(image, 2.0)
    print("   ✅ Contrast adjustment works\n")
    
    print("3. Test color conversion...")
    gray = processor.convert_color(image, 'bgr2gray')
    hsv = processor.convert_color(image, 'bgr2hsv')
    print(f"   Grayscale shape: {gray.shape}")
    print(f"   HSV shape: {hsv.shape}")
    print("   ✅ Color conversion works\n")
    
    print("="*60)
    print("✅ COLOR OPERATIONS TEST PASSED!")
    print("="*60 + "\n")


def test_filters():
    """Test image filters"""
    print("\n" + "="*60)
    print("🧪 TEST 3: FILTERS")
    print("="*60 + "\n")
    
    processor = ImageProcessor()
    image = create_test_image()
    
    print("1. Test Gaussian blur...")
    gaussian = processor.apply_blur(image, 15, "gaussian")
    print(f"   Blurred shape: {gaussian.shape}")
    print("   ✅ Gaussian blur works\n")
    
    print("2. Test median blur...")
    median = processor.apply_blur(image, 15, "median")
    print(f"   Median blur shape: {median.shape}")
    print("   ✅ Median blur works\n")
    
    print("3. Test edge detection...")
    edges = processor.detect_edges(image)
    print(f"   Edges shape: {edges.shape}")
    print("   ✅ Edge detection works\n")
    
    print("="*60)
    print("✅ FILTERS TEST PASSED!")
    print("="*60 + "\n")


def test_statistics():
    """Test image statistics"""
    print("\n" + "="*60)
    print("🧪 TEST 4: IMAGE STATISTICS")
    print("="*60 + "\n")
    
    processor = ImageProcessor()
    image = create_test_image()
    
    print("Getting image statistics...")
    stats = processor.get_image_stats(image)
    
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n   ✅ Statistics works\n")
    
    print("="*60)
    print("✅ STATISTICS TEST PASSED!")
    print("="*60 + "\n")


def test_visualization():
    """Test visualization"""
    print("\n" + "="*60)
    print("🧪 TEST 5: VISUALIZATION")
    print("="*60 + "\n")
    
    processor = ImageProcessor()
    image = create_test_image()
    
    print("Creating transformation visualization...")
    
    transformations = [
        ('resize', {'size': (150, 150)}),
        ('rotate', {'angle': 45}),
        ('flip', {'direction': 'horizontal'}),
        ('adjust_brightness', {'factor': 1.5}),
        ('apply_blur', {'kernel_size': 15}),
        ('detect_edges', {}),
    ]
    
    processor.visualize_transformations(image, transformations)
    
    print("\n   ✅ Visualization created\n")
    
    print("="*60)
    print("✅ VISUALIZATION TEST PASSED!")
    print("="*60 + "\n")


def main():
    """Run all tests"""
    print("Testing all image operations with OpenCV\n")
    
    try:
        # Test 1: Basic Operations
        test_basic_operations()
        
        # Test 2: Color Operations
        test_color_operations()
        
        # Test 3: Filters
        test_filters()
        
        # Test 4: Statistics
        test_statistics()
        
        # Test 5: Visualization
        test_visualization()
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ALL IMAGE PROCESSING TESTS COMPLETED!")
        print("="*60)
        print("\n✅ Basic Operations: PASSED")
        print("✅ Color Operations: PASSED")
        print("✅ Filters: PASSED")
        print("✅ Statistics: PASSED")
        print("✅ Visualization: PASSED")
        print("\n🎯 Day 43 image processing fundamentals complete!")
        print("\n💡 Key Skills Learned:")
        print("  - Image loading & saving")
        print("  - Geometric transformations")
        print("  - Color space conversions")
        print("  - Filtering & edge detection")
        print("  - Image statistics")
        print("  - Visualization techniques\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
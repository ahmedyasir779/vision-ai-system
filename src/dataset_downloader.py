import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import shutil


class DatasetDownloader:
    """
    Download public image datasets
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize downloader
        
        Args:
            data_dir: Directory to save datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"📥 Dataset downloader initialized")
        print(f"   Data directory: {self.data_dir}")
    
    def download_file(
        self,
        url: str,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Download file with progress bar
        
        Args:
            url: URL to download from
            output_path: Where to save (optional)
            
        Returns:
            Path to downloaded file
        """
        if output_path is None:
            filename = url.split('/')[-1]
            output_path = self.data_dir / filename
        else:
            output_path = Path(output_path)
        
        print(f"\n📥 Downloading: {url}")
        print(f"   Saving to: {output_path}")
        
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {output_path.name}")
        return output_path
    
    def extract_archive(
        self,
        archive_path: Path,
        extract_to: Optional[Path] = None
    ) -> Path:
        """
        Extract zip or tar archive
        
        Args:
            archive_path: Path to archive
            extract_to: Where to extract (optional)
            
        Returns:
            Path to extracted directory
        """
        if extract_to is None:
            extract_to = archive_path.parent / archive_path.stem
        
        print(f"\n📦 Extracting: {archive_path.name}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
        
        print(f"✅ Extracted to: {extract_to}")
        return extract_to
    
    def download_sample_dataset(self) -> Path:
        """
        Download a small sample dataset for testing
        
        We'll use Intel Image Classification dataset (small subset)
        Categories: buildings, forest, glacier, mountain, sea, street
        """
        print("\n" + "="*60)
        print("📥 DOWNLOADING SAMPLE DATASET")
        print("="*60)
        
        # For this tutorial, we'll create a simple dataset structure
        # In practice, you'd download from Kaggle or other sources
        
        dataset_dir = self.data_dir / "sample_images"
        dataset_dir.mkdir(exist_ok=True)
        
        categories = ['cat', 'dog', 'bird', 'car', 'flower']
        
        print(f"\n📁 Creating sample dataset structure...")
        print(f"   Categories: {', '.join(categories)}")
        
        for category in categories:
            category_dir = dataset_dir / category
            category_dir.mkdir(exist_ok=True)
            print(f"   ✅ Created: {category}/")
        
        print(f"\n✅ Dataset structure ready!")
        print(f"   Location: {dataset_dir}")
        print(f"\n💡 Next steps:")
        print(f"   1. Add your own images to: {dataset_dir}/[category]/")
        print(f"   2. Or use the sample generator below")
        
        return dataset_dir
    
    def create_sample_images(self, num_per_class: int = 10) -> Path:
        """
        Create synthetic sample images for testing
        
        Args:
            num_per_class: Number of images per category
            
        Returns:
            Path to dataset directory
        """
        import cv2
        import numpy as np
        
        print("\n" + "="*60)
        print("🎨 CREATING SAMPLE IMAGES")
        print("="*60)
        
        dataset_dir = self.data_dir / "sample_images"
        
        categories = {
            'cat': (255, 100, 50),      # Orange-ish
            'dog': (150, 100, 50),      # Brown-ish
            'bird': (100, 150, 255),    # Blue-ish
            'car': (100, 100, 100),     # Gray
            'flower': (255, 150, 200)   # Pink-ish
        }
        
        total_images = len(categories) * num_per_class
        
        with tqdm(total=total_images, desc="Creating images") as pbar:
            for category, color in categories.items():
                category_dir = dataset_dir / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                for i in range(num_per_class):
                    # Create 224x224 image with category color + noise
                    img = np.ones((224, 224, 3), dtype=np.uint8)
                    
                    # Base color
                    img[:, :] = color
                    
                    # Add random noise
                    noise = np.random.randint(-30, 30, (224, 224, 3))
                    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
                    
                    # Add some shapes for variety
                    if i % 3 == 0:
                        cv2.circle(img, (112, 112), 50, (255, 255, 255), -1)
                    elif i % 3 == 1:
                        cv2.rectangle(img, (50, 50), (174, 174), (255, 255, 255), -1)
                    else:
                        pts = np.array([[112, 50], [50, 174], [174, 174]], np.int32)
                        cv2.fillPoly(img, [pts], (255, 255, 255))
                    
                    # Save
                    filename = f"{category}_{i:03d}.jpg"
                    cv2.imwrite(str(category_dir / filename), img)
                    
                    pbar.update(1)
        
        print(f"\n✅ Created {total_images} sample images!")
        print(f"   Location: {dataset_dir}")
        print(f"   {num_per_class} images per category")
        
        return dataset_dir
    
    def get_dataset_info(self, dataset_dir: Path) -> dict:
        """Get information about dataset"""
        info = {
            'path': str(dataset_dir),
            'categories': [],
            'total_images': 0,
            'images_per_category': {}
        }
        
        if not dataset_dir.exists():
            return info
        
        for category_dir in sorted(dataset_dir.iterdir()):
            if category_dir.is_dir():
                category = category_dir.name
                images = list(category_dir.glob('*.jpg')) + \
                         list(category_dir.glob('*.png')) + \
                         list(category_dir.glob('*.jpeg'))
                
                count = len(images)
                
                info['categories'].append(category)
                info['images_per_category'][category] = count
                info['total_images'] += count
        
        return info
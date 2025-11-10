import urllib.request
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil


class RealDatasetDownloader:
    """
    Download real image datasets
    """
    
    def __init__(self, data_dir: str = "data/real_dataset"):
        """
        Initialize downloader
        
        Args:
            data_dir: Directory to save dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Real dataset downloader initialized")
        print(f"   Data directory: {self.data_dir}")
    
    def download_flowers_dataset(self) -> Path:
        """
        Download Flowers dataset
        
        5 categories: daisy, dandelion, roses, sunflowers, tulips
        ~800 images total
        
        Returns:
            Path to dataset directory
        """
        print("\n" + "="*60)
        print("🌸 DOWNLOADING FLOWERS DATASET")
        print("="*60)
        
        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        
        output_file = self.data_dir / "flower_photos.tgz"
        extract_dir = self.data_dir / "flower_photos"
        
        # Check if already downloaded
        if extract_dir.exists():
            print(f"\n✅ Dataset already exists: {extract_dir}")
            return extract_dir
        
        # Download
        print(f"\n📥 Downloading from: {dataset_url}")
        print(f"   This may take a few minutes...")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(
                dataset_url,
                filename=output_file,
                reporthook=t.update_to
            )
        
        print(f"✅ Downloaded: {output_file.name}")
        
        # Extract
        print(f"\n📦 Extracting archive...")
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall(self.data_dir)
        
        print(f"✅ Extracted to: {extract_dir}")
        
        # Clean up archive
        output_file.unlink()
        print(f"🗑️ Cleaned up archive file")
        
        # Get dataset info
        self._print_dataset_info(extract_dir)
        
        return extract_dir
    
    def _print_dataset_info(self, dataset_dir: Path):
        """Print dataset information"""
        print("\n" + "="*60)
        print("📊 DATASET INFO")
        print("="*60)
        
        categories = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        
        total_images = 0
        
        print(f"\n{'Category':<15} {'Images':<10}")
        print("-" * 30)
        
        for category_dir in categories:
            category = category_dir.name
            
            # Skip LICENSE file
            if category == 'LICENSE.txt':
                continue
            
            images = list(category_dir.glob('*.jpg')) + \
                     list(category_dir.glob('*.jpeg')) + \
                     list(category_dir.glob('*.png'))
            
            count = len(images)
            total_images += count
            
            print(f"{category:<15} {count:<10}")
        
        print("-" * 30)
        print(f"{'TOTAL':<15} {total_images:<10}")
        print()
    
    def create_subset(
        self,
        source_dir: Path,
        output_dir: Path,
        samples_per_class: int = 150
    ) -> Path:
        """
        Create a smaller subset for faster training
        
        Args:
            source_dir: Source dataset directory
            output_dir: Output directory
            samples_per_class: Number of samples per class
            
        Returns:
            Path to subset directory
        """
        print("\n" + "="*60)
        print("✂️ CREATING DATASET SUBSET")
        print("="*60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        categories = sorted([d for d in source_dir.iterdir() if d.is_dir()])
        
        print(f"\n📊 Creating subset with {samples_per_class} images per class...")
        
        for category_dir in tqdm(categories, desc="Processing categories"):
            category = category_dir.name
            
            # Skip LICENSE
            if category == 'LICENSE.txt':
                continue
            
            # Create output category dir
            output_category = output_dir / category
            output_category.mkdir(exist_ok=True)
            
            # Get all images
            images = list(category_dir.glob('*.jpg')) + \
                     list(category_dir.glob('*.jpeg')) + \
                     list(category_dir.glob('*.png'))
            
            # Take subset
            subset_images = images[:samples_per_class]
            
            # Copy images
            for img in subset_images:
                shutil.copy2(img, output_category / img.name)
        
        self._print_dataset_info(output_dir)
        
        return output_dir


class DatasetPreparer:
    """
    Prepare dataset for training
    """
    
    def __init__(self):
        """Initialize preparer"""
        print("🔧 Dataset preparer initialized")
    
    def prepare_flowers_dataset(
        self,
        samples_per_class: int = 150,
        force_download: bool = False
    ) -> Path:
        """
        Download and prepare flowers dataset
        
        Args:
            samples_per_class: Images per class
            force_download: Force re-download
            
        Returns:
            Path to prepared dataset
        """
        downloader = RealDatasetDownloader()
        
        # Download full dataset
        full_dataset = downloader.download_flowers_dataset()
        
        # Create subset
        subset_dir = Path("data/real_dataset/flowers_subset")
        
        if not subset_dir.exists() or force_download:
            subset_dir = downloader.create_subset(
                full_dataset,
                subset_dir,
                samples_per_class=samples_per_class
            )
        else:
            print(f"\n✅ Subset already exists: {subset_dir}")
        
        return subset_dir
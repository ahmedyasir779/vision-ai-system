import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import random
from collections import defaultdict
from tqdm import tqdm


class DatasetOrganizer:
    """
    Organize dataset into train/val/test splits
    """
    
    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        """
        Initialize organizer
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        random.seed(random_seed)
        
        print(f"📊 Dataset organizer initialized")
        print(f"   Train: {train_ratio:.0%}")
        print(f"   Val: {val_ratio:.0%}")
        print(f"   Test: {test_ratio:.0%}")
    
    def split_dataset(
        self,
        source_dir: Path,
        output_dir: Path,
        copy_files: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """
        Split dataset into train/val/test
        
        Args:
            source_dir: Source dataset directory
            output_dir: Output directory for splits
            copy_files: If True, copy files; if False, move files
            
        Returns:
            Dictionary with split statistics
        """
        print("\n" + "="*60)
        print("📊 SPLITTING DATASET")
        print("="*60)
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        # Create output structure
        splits = ['train', 'val', 'test']
        for split in splits:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        
        stats = defaultdict(lambda: defaultdict(int))
        
        # Get all categories
        categories = [d for d in source_dir.iterdir() if d.is_dir()]
        
        print(f"\n📁 Found {len(categories)} categories")
        
        for category_dir in tqdm(categories, desc="Processing categories"):
            category = category_dir.name
            
            # Create category dirs in each split
            for split in splits:
                (output_dir / split / category).mkdir(parents=True, exist_ok=True)
            
            # Get all images
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(list(category_dir.glob(ext)))
            
            # Shuffle
            random.shuffle(images)
            
            # Calculate split indices
            n_images = len(images)
            n_train = int(n_images * self.train_ratio)
            n_val = int(n_images * self.val_ratio)
            
            # Split
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy/move files
            self._transfer_files(
                train_images,
                output_dir / 'train' / category,
                copy_files
            )
            self._transfer_files(
                val_images,
                output_dir / 'val' / category,
                copy_files
            )
            self._transfer_files(
                test_images,
                output_dir / 'test' / category,
                copy_files
            )
            
            # Update stats
            stats[category]['train'] = len(train_images)
            stats[category]['val'] = len(val_images)
            stats[category]['test'] = len(test_images)
            stats[category]['total'] = n_images
        
        # Print summary
        self._print_split_summary(stats)
        
        return dict(stats)
    
    def _transfer_files(
        self,
        files: List[Path],
        dest_dir: Path,
        copy: bool
    ) -> None:
        """Transfer files to destination"""
        for file in files:
            dest_file = dest_dir / file.name
            if copy:
                shutil.copy2(file, dest_file)
            else:
                shutil.move(str(file), str(dest_file))
    
    def _print_split_summary(self, stats: Dict) -> None:
        """Print split summary"""
        print("\n" + "="*60)
        print("📊 SPLIT SUMMARY")
        print("="*60)
        
        print(f"\n{'Category':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
        print("-" * 60)
        
        total_train = 0
        total_val = 0
        total_test = 0
        total_all = 0
        
        for category, counts in sorted(stats.items()):
            print(f"{category:<15} {counts['train']:<10} {counts['val']:<10} "
                  f"{counts['test']:<10} {counts['total']:<10}")
            
            total_train += counts['train']
            total_val += counts['val']
            total_test += counts['test']
            total_all += counts['total']
        
        print("-" * 60)
        print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} "
              f"{total_test:<10} {total_all:<10}")
        
        print(f"\n✅ Dataset split complete!")
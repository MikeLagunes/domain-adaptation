"""
CORe50 Dataset Implementation
PyTorch Dataset class for CORe50 with hierarchical structure
"""

import os
from pathlib import Path
from typing import List, Optional, Callable, Dict, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image


class CORe50Dataset(Dataset):
    """
    PyTorch Dataset for CORe50 dataset with hierarchical structure:
    CORe50/s{session}/o{object}/C_{session}_{object}_{image_id}.png
    
    Args:
        root_dir (str): Path to the CORe50 directory
        sessions (list): List of sessions to include (e.g., [1, 2, 3])
        objects (list, optional): List of objects to include (e.g., [1, 2, 3]). 
                                  If None, all 50 objects are used.
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(
        self, 
        root_dir: str, 
        sessions: List[int],
        objects: Optional[List[int]] = None,
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.sessions = sessions
        self.objects = objects if objects is not None else list(range(1, 51))  # o1-o50
        self.transform = transform
        
        # Build the dataset index
        self.samples = []
        self.class_to_idx = {}
        self._build_dataset()
    
    def _build_dataset(self):
        """Build the dataset by scanning the directory structure"""
        class_idx = 0
        
        for session in self.sessions:
            session_dir = self.root_dir / f"s{session}"
            
            if not session_dir.exists():
                print(f"Warning: Session directory {session_dir} does not exist, skipping...")
                continue
                
            for obj in self.objects:
                obj_dir = session_dir / f"o{obj}"
                
                if not obj_dir.exists():
                    print(f"Warning: Object directory {obj_dir} does not exist, skipping...")
                    continue
                
                # Create class label - same object across sessions = same class
                class_name = f"o{obj}"
                
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = class_idx
                    class_idx += 1
                
                # Find all PNG files in the object directory
                image_files = list(obj_dir.glob("C_*.png"))
                
                for img_file in image_files:
                    self.samples.append({
                        'path': img_file,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'session': session - 1,
                        'object': obj
                    })
        
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                f"Please check that the directory structure is correct:\n"
                f"CORe50/s{{session}}/o{{object}}/C_*.png"
            )
        
        print(f"CORe50 Dataset loaded:")
        print(f"  Images: {len(self.samples)}")
        print(f"  Classes: {len(self.class_to_idx)}")
        print(f"  Sessions: {sorted(set(s['session'] for s in self.samples))}")
        print(f"  Objects: {sorted(set(s['object'] for s in self.samples))}")
    
    def __len__(self) -> int:
        """Return the total number of samples"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        sample = self.samples[idx]
        
        # Load image
        # print(f"Loading image: {sample['path']}")
        image = Image.open(sample['path']).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['class_idx'], sample['session']
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get detailed information about a sample
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary with keys 'path', 'class_name', 'class_idx', 'session', 'object'
        """
        return self.samples[idx].copy()
    
    def get_classes(self) -> List[str]:
        """Get list of class names"""
        return sorted(self.class_to_idx.keys())
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class"""
        distribution = {}
        for sample in self.samples:
            class_name = sample['class_name']
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def get_session_distribution(self) -> Dict[int, int]:
        """Get distribution of samples per session"""
        distribution = {}
        for sample in self.samples:
            session = sample['session']
            distribution[session] = distribution.get(session, 0) + 1
        return distribution


# Example usage and testing
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    print("Creating dataset...")
    dataset = CORe50Dataset(
        root_dir="path/to/CORe50",  # Update this path
        sessions=[1, 2, 3],
        objects=None,  # Use all objects
        transform=transform
    )
    
    # Print dataset info
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_to_idx)}")
    print(f"Classes: {dataset.get_classes()}")
    
    # Print class distribution
    print("\nClass distribution:")
    class_dist = dataset.get_class_distribution()
    for class_name, count in sorted(class_dist.items())[:5]:  # Show first 5
        print(f"  {class_name}: {count} images")
    
    # Print session distribution
    print("\nSession distribution:")
    session_dist = dataset.get_session_distribution()
    for session, count in sorted(session_dist.items()):
        print(f"  Session {session}: {count} images")
    
    # Test getting a sample
    print("\nTesting sample retrieval...")
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    sample_info = dataset.get_sample_info(0)
    print(f"Sample info: {sample_info}")
    
    # Create DataLoader
    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    # Test DataLoader
    print("Testing DataLoader...")
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}: Images shape: {images.shape}, Targets: {targets.shape}")
        if batch_idx == 2:  # Just test first 3 batches
            break
    
    print("\nDataset test complete!")
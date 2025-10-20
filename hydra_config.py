"""
Hydra Configuration Utilities
Helper functions for working with Hydra configs and creating dataloaders
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import torch
import torchvision.transforms as transforms
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def create_transforms(cfg: DictConfig, mode: str = 'train'):
    """
    Create transforms based on hydra configuration
    
    Args:
        cfg: Hydra config (full config, not just augmentation)
        mode: 'train', 'val', or 'test'
        
    Returns:
        transforms.Compose: Composed transforms
    """
    
    if mode == 'train':
        transform_list = []
        
        # Resize and crop
        if cfg.augmentation.random_crop:
            resize_size = int(cfg.data.image_size * 1.14)
            transform_list.extend([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(cfg.data.image_size)
            ])
        else:
            transform_list.append(
                transforms.Resize((cfg.data.image_size, cfg.data.image_size))
            )
        
        # Augmentations
        if cfg.augmentation.random_horizontal_flip:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=cfg.augmentation.flip_prob)
            )
        
        if cfg.augmentation.random_rotation:
            transform_list.append(
                transforms.RandomRotation(cfg.augmentation.rotation_degrees)
            )
        
        if cfg.augmentation.color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=cfg.augmentation.brightness,
                contrast=cfg.augmentation.contrast,
                saturation=cfg.augmentation.saturation,
                hue=cfg.augmentation.hue
            ))
        
        if cfg.augmentation.gaussian_blur:
            transform_list.append(
                transforms.GaussianBlur(kernel_size=cfg.augmentation.blur_kernel_size)
            )
        
        # Normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.augmentation.get('normalize_mean', [0.485, 0.456, 0.406]),
                std=cfg.augmentation.get('normalize_std', [0.229, 0.224, 0.225])
            )
        ])
        
    else:  # val/test - no augmentation
        transform_list = [
            transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.augmentation.get('normalize_mean', [0.485, 0.456, 0.406]),
                std=cfg.augmentation.get('normalize_std', [0.229, 0.224, 0.225])
            )
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders_from_hydra_config(cfg: DictConfig, dataset_class):
    """
    Create train/val/test dataloaders from hydra configuration
    
    Args:
        cfg: Hydra DictConfig (full config)
        dataset_class: The Dataset class to instantiate (e.g., CORe50Dataset)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, 
                train_dataset, val_dataset, test_dataset)
    """
    
    # Set seed for reproducibility
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
        log.info(f"Set random seed to {cfg.seed}")
    
    # Create transforms
    train_transform = create_transforms(cfg, 'train')
    val_transform = create_transforms(cfg, 'val')
    test_transform = create_transforms(cfg, 'test')
    
    # Convert OmegaConf lists to Python lists
    train_sessions = OmegaConf.to_container(cfg.data.train_sessions, resolve=True)
    val_sessions = OmegaConf.to_container(cfg.data.val_sessions, resolve=True)
    test_sessions = OmegaConf.to_container(cfg.data.test_sessions, resolve=True)
    objects = OmegaConf.to_container(cfg.data.objects, resolve=True) if cfg.data.objects else None
    
    log.info(f"Creating datasets:")
    log.info(f"  Train sessions: {train_sessions}")
    log.info(f"  Val sessions: {val_sessions}")
    log.info(f"  Test sessions: {test_sessions}")
    log.info(f"  Objects: {objects if objects else 'All (1-50)'}")
    
    # Create datasets
    train_dataset = dataset_class(
        root_dir=cfg.data.root_dir,
        sessions=train_sessions,
        objects=objects,
        transform=train_transform
    )
    
    val_dataset = dataset_class(
        root_dir=cfg.data.root_dir,
        sessions=val_sessions,
        objects=objects,
        transform=val_transform
    )
    
    test_dataset = dataset_class(
        root_dir=cfg.data.root_dir,
        sessions=test_sessions,
        objects=objects,
        transform=test_transform
    )
    
    log.info(f"Dataset sizes:")
    log.info(f"  Train: {len(train_dataset)} samples")
    log.info(f"  Val: {len(val_dataset)} samples")
    log.info(f"  Test: {len(test_dataset)} samples")
    log.info(f"  Classes: {len(train_dataset.class_to_idx)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle_train,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle_val,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle_test,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def validate_config(cfg: DictConfig):
    """
    Validate hydra configuration
    
    Args:
        cfg: Hydra DictConfig to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    
    # Check sessions don't overlap
    all_sessions = set(cfg.data.train_sessions + cfg.data.val_sessions + cfg.data.test_sessions)
    total_sessions = len(cfg.data.train_sessions) + len(cfg.data.val_sessions) + len(cfg.data.test_sessions)
    
    if len(all_sessions) != total_sessions:
        raise ValueError("Sessions cannot overlap between train/val/test splits")
    
    # Check data directory exists
    if not Path(cfg.data.root_dir).exists():
        log.warning(f"Data directory does not exist: {cfg.data.root_dir}")
    
    # Check session ranges
    for session_list, name in [(cfg.data.train_sessions, 'train'), 
                               (cfg.data.val_sessions, 'val'), 
                               (cfg.data.test_sessions, 'test')]:
        for session in session_list:
            if not (1 <= session <= 11):
                raise ValueError(f"Invalid session {session} in {name}_sessions. Must be 1-11")
    
    # Check object ranges if specified
    if cfg.data.objects:
        for obj in cfg.data.objects:
            if not (1 <= obj <= 50):
                raise ValueError(f"Invalid object {obj}. Must be 1-50")
    
    # Check batch size is positive
    if cfg.dataloader.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {cfg.dataloader.batch_size}")
    
    # Check image size is positive
    if cfg.data.image_size <= 0:
        raise ValueError(f"Image size must be positive, got {cfg.data.image_size}")
    
    log.info("Configuration validation passed ✓")


def print_config(cfg: DictConfig, resolve: bool = True):
    """
    Pretty print the configuration
    
    Args:
        cfg: Hydra DictConfig to print
        resolve: Whether to resolve interpolations
    """
    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
    print("=" * 80)


def save_config(cfg: DictConfig, path: str):
    """
    Save hydra config to file
    
    Args:
        cfg: Hydra DictConfig to save
        path: Output file path
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        OmegaConf.save(cfg, f)
    log.info(f"Configuration saved to {path}")


def load_config(path: str) -> DictConfig:
    """
    Load hydra config from file
    
    Args:
        path: Path to config file
        
    Returns:
        DictConfig: Loaded configuration
    """
    cfg = OmegaConf.load(path)
    log.info(f"Configuration loaded from {path}")
    return cfg


def get_config_diff(cfg1: DictConfig, cfg2: DictConfig) -> str:
    """
    Get the difference between two configs
    
    Args:
        cfg1: First config
        cfg2: Second config
        
    Returns:
        str: Human-readable diff
    """
    diff_lines = []
    
    def compare_dict(d1, d2, prefix=""):
        keys = set(list(d1.keys()) + list(d2.keys()))
        for key in sorted(keys):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in d1:
                diff_lines.append(f"+ {full_key}: {d2[key]}")
            elif key not in d2:
                diff_lines.append(f"- {full_key}: {d1[key]}")
            elif OmegaConf.is_dict(d1[key]) and OmegaConf.is_dict(d2[key]):
                compare_dict(d1[key], d2[key], full_key)
            elif d1[key] != d2[key]:
                diff_lines.append(f"  {full_key}: {d1[key]} -> {d2[key]}")
    
    compare_dict(cfg1, cfg2)
    return "\n".join(diff_lines) if diff_lines else "No differences"


def merge_configs(base_cfg: DictConfig, override_cfg: DictConfig) -> DictConfig:
    """
    Merge two configs, with override_cfg taking precedence
    
    Args:
        base_cfg: Base configuration
        override_cfg: Override configuration
        
    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_cfg, override_cfg)


# Example usage and testing
if __name__ == "__main__":
    """Test the hydra config utilities"""
    
    # Example: Create a config programmatically
    cfg = OmegaConf.create({
        "name": "test_experiment",
        "seed": 42,
        "data": {
            "root_dir": "/path/to/CORe50",
            "train_sessions": [1, 2, 3, 4, 5, 6, 7, 8],
            "val_sessions": [9],
            "test_sessions": [10, 11],
            "objects": None,
            "image_size": 224
        },
        "augmentation": {
            "random_crop": True,
            "random_horizontal_flip": True,
            "random_rotation": False,
            "color_jitter": True,
            "gaussian_blur": False,
            "flip_prob": 0.5,
            "rotation_degrees": 10,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
            "blur_kernel_size": 3,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "dataloader": {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": True,
            "shuffle_train": True,
            "shuffle_val": False,
            "shuffle_test": False
        }
    })
    
    print("Testing Hydra config utilities...\n")
    
    # Test print
    print("1. Print config:")
    print_config(cfg)
    
    # Test validation
    print("\n2. Validate config:")
    try:
        validate_config(cfg)
        print("✓ Validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test save/load
    print("\n3. Save and load config:")
    test_path = "/tmp/test_config.yaml"
    save_config(cfg, test_path)
    loaded_cfg = load_config(test_path)
    print(f"✓ Config saved to and loaded from {test_path}")
    
    # Test transforms creation
    print("\n4. Create transforms:")
    train_transform = create_transforms(cfg, 'train')
    val_transform = create_transforms(cfg, 'val')
    print(f"✓ Created train transform with {len(train_transform.transforms)} steps")
    print(f"✓ Created val transform with {len(val_transform.transforms)} steps")
    
    # Test config diff
    print("\n5. Test config diff:")
    cfg2 = cfg.copy()
    cfg2.dataloader.batch_size = 64
    cfg2.augmentation.random_rotation = True
    diff = get_config_diff(cfg, cfg2)
    print("Differences:")
    print(diff)
    
    # Test merge
    print("\n6. Test config merge:")
    override = OmegaConf.create({
        "dataloader": {"batch_size": 128},
        "seed": 123
    })
    merged = merge_configs(cfg, override)
    print(f"✓ Merged config - batch_size: {merged.dataloader.batch_size}, seed: {merged.seed}")
    
    print("\n✓ All tests passed!")
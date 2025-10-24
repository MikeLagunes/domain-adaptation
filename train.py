#!/usr/bin/env python3
"""
CORe50 training with Hydra configuration management
"""

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import logging
from pathlib import Path
from tqdm import tqdm
import wandb
import json
from datetime import datetime

# Import our modules
from core50_dataset import CORe50Dataset
from hydra_config import create_dataloaders_from_hydra_config, validate_config, print_config

log = logging.getLogger(__name__)


def create_model(cfg: DictConfig):
    """Create model from configuration"""
    
    if cfg.experiment.model.name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(pretrained=cfg.experiment.model.pretrained)
        model.fc = nn.Linear(model.fc.in_features, cfg.experiment.model.num_classes)
    elif cfg.experiment.model.name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=cfg.experiment.model.pretrained)
        model.fc = nn.Linear(model.fc.in_features, cfg.experiment.model.num_classes)
    elif cfg.experiment.model.name == "maxvit":
        from models import maxvit_t
        # TODO (mike): Add config option for weights path
        model = maxvit_t(weights="/path/to/maxvit_t-bc5ab103.pth")
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, cfg.experiment.model.num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.experiment.model.name}")
    
    return model


def create_optimizer(model, cfg: DictConfig):
    """Create optimizer from configuration"""
    
    if cfg.experiment.optimizer.name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.experiment.optimizer.lr,
            weight_decay=cfg.experiment.optimizer.weight_decay
        )
    elif cfg.experiment.optimizer.name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.experiment.optimizer.lr,
            momentum=cfg.experiment.optimizer.get('momentum', 0.9),
            weight_decay=cfg.experiment.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.experiment.optimizer.name}")
    
    return optimizer


def create_scheduler(optimizer, cfg: DictConfig):
    """Create learning rate scheduler from configuration"""
    
    if cfg.experiment.training.scheduler.name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.experiment.training.epochs,
            eta_min=cfg.experiment.training.scheduler.min_lr
        )
    elif cfg.experiment.training.scheduler.name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=cfg.experiment.training.scheduler.step_size,
            gamma=cfg.experiment.training.scheduler.gamma
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, split_name="Val"):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"{split_name}"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training function with Hydra"""
    
    # Print and validate configuration
    log.info("=== Configuration ===")
    print_config(cfg)
    validate_config(cfg)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    
    # Initialize wandb if available
    try:
        wandb.init(
            entity="mike_lf",
            project='domain-adaptation',
            name=cfg.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.experiment.name]
        )
        use_wandb = True
    except Exception as e:
        log.warning(f"Could not initialize wandb: {e}")
        use_wandb = False
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = \
        create_dataloaders_from_hydra_config(cfg, CORe50Dataset)
    
    # Create model, optimizer, scheduler
    model = create_model(cfg).to(device)
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()
    
    log.info(f"Model: {cfg.experiment.model.name}")
    log.info(f"Optimizer: {cfg.experiment.optimizer.name}")
    log.info(f"Scheduler: {cfg.experiment.training.scheduler.name}")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    results = []
    
    for epoch in range(1, cfg.experiment.training.epochs + 1):
        log.info(f"\nEpoch {epoch}/{cfg.experiment.training.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, "Val")
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.experiment.optimizer.lr
        
        # Log results
        epoch_results = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        }
        results.append(epoch_results)
        
        log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        log.info(f"Learning Rate: {current_lr:.6f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log(epoch_results)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, 'best_model.pth')
            log.info(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Final test evaluation
    log.info(f"\n=== Final Evaluation ===")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device, "Test")
    
    # Final results
    final_results = {
        'experiment_name': cfg.name,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_epochs': cfg.experiment.training.epochs,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    with open('training_history.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log final results
    log.info(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")
    log.info(f"Test Acc: {test_acc:.2f}%")
    log.info(f"Test Loss: {test_loss:.4f}")
    
    if use_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_loss': test_loss
        })
        wandb.finish()
    
    log.info("Training completed!")
    return test_acc


# Alternative: programmatic configuration for experimentation
def run_experiment(experiment_name: str, overrides: list = None):
    """Run experiment programmatically"""
    
    with initialize(version_base=None, config_path="conf"):
        # Compose configuration with overrides
        overrides = overrides or []
        overrides.append(f"experiment={experiment_name}")
        
        cfg = compose(config_name="config", overrides=overrides)
        
        # Run training
        return train(cfg)


# Multi-run experiments
@hydra.main(version_base=None, config_path="conf", config_name="config")
def multi_run(cfg: DictConfig) -> float:
    """Multi-run version for hyperparameter sweeps"""
    return train(cfg)


if __name__ == "__main__":
    # Command line training
    train()
    
    # Alternative: programmatic experiments
    # results = {}
    # experiments = ["baseline", "few_shot", "continual"]
    # 
    # for exp in experiments:
    #     print(f"\n{'='*50}")
    #     print(f"Running experiment: {exp}")
    #     print('='*50)
    #     
    #     test_acc = run_experiment(exp)
    #     results[exp] = test_acc
    #     
    #     print(f"Completed {exp}: Test Acc = {test_acc:.2f}%")
    # 
    # print(f"\nFinal Results: {results}")


# Example usage commands:
"""
# Basic training with default config
python train.py

# Override specific parameters
python train.py data=few_shot augmentation=heavy dataloader.batch_size=64

# Different experiments
python train.py experiment=baseline
python train.py experiment=few_shot  
python train.py experiment=continual

# Override data path
python train.py data.root_dir=/path/to/your/CORe50

# Multi-run with different seeds
python train.py --multirun seed=42,123,456

# Hyperparameter sweep
python train.py --multirun experiment.training.learning_rate=0.001,0.01,0.1 dataloader.batch_size=16,32,64

# Custom experiment
python train.py experiment=baseline data.train_sessions=[1,2,3] data.test_sessions=[4,5,6] name=custom_experiment
"""
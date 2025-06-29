#!/usr/bin/env python3
"""
Training script for Vision Transformer with different positional encodings.
Usage: python train.py --config configs/config.yaml
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from model.model_builder import build_model, get_model_summary
from dataset_utils import get_data_loaders


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, log_every_n_steps):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Logging
        if batch_idx % log_every_n_steps == 0:
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Step', loss.item(), step)
            writer.add_scalar('Accuracy/Train_Step', 100. * correct / total, step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc, checkpoint_path, config):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Vision Transformer')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['seed'])
    
    # Create output directories
    experiment_name = config['experiment']['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{experiment_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    # Copy config file to output directory for reproducibility
    shutil.copy(args.config, f"{output_dir}/config.yaml")
    
    # Create data loaders
    train_loader, test_loader = get_data_loaders(config)
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Print model summary
    model_summary = get_model_summary(model, config)
    print("\nModel Summary:")
    for key, value in model_summary.items():
        print(f"  {key}: {value}")
    
    # Save model summary
    with open(f"{output_dir}/model_summary.json", 'w') as f:
        json.dump(model_summary, f, indent=2)
    
    # Setup training
    training_config = config['training']
    criterion = nn.CrossEntropyLoss()
    
    if training_config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(training_config['learning_rate']),
            weight_decay=float(training_config['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
    
    if training_config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['num_epochs'])
    else:
        scheduler = None
    
    # Early stopping
    early_stopping = None
    if training_config['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=training_config['early_stopping']['patience'],
            min_delta=training_config['early_stopping']['min_delta']
        )
    
    # Setup logging
    writer = SummaryWriter(f"{output_dir}/logs")
    
    # Training state
    start_epoch = 0
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['val_acc']
    
    # Training loop
    print(f"\nStarting training for {training_config['num_epochs']} epochs...")
    
    for epoch in range(start_epoch, training_config['num_epochs']):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer, 
            training_config['log_every_n_steps']
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        if scheduler:
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        print(f'Epoch {epoch+1}/{training_config["num_epochs"]}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if training_config['save_best_model'] and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc,
                f"{output_dir}/checkpoints/best_model.pth", config
            )
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every n epochs
        if (epoch + 1) % training_config['save_every_n_epochs'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, val_acc,
                f"{output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pth", config
            )
        
        # Early stopping
        if early_stopping and early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch, val_loss, val_acc,
        f"{output_dir}/checkpoints/final_model.pth", config
    )
    
    # Plot and save training curves
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        f"{output_dir}/training_curves.png"
    )
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    with open(f"{output_dir}/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    writer.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
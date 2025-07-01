#!/usr/bin/env python3
"""
Testing script for evaluating Vision Transformer against adversarial attacks.
Usage: python test.py --config configs/config.yaml --checkpoint path/to/model.pth
"""

import argparse
import os
import yaml
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import shutil

from model.model_builder import build_model, get_model_summary
from dataset_utils import get_data_loaders
from attacks import create_attack, evaluate_attack


def display_adversarial_examples(clean_images, adv_images, predictions, labels, save_path, num_examples=5):
    """Display and save adversarial examples"""
    fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
    
    # CIFAR-10 class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(min(num_examples, len(clean_images))):
        # Original image
        clean_img = clean_images[i].permute(1, 2, 0).cpu().numpy()
        clean_img = np.clip(clean_img, 0, 1)
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f'Clean: {classes[labels[i]]}')
        axes[0, i].axis('off')
        
        # Adversarial image
        adv_img = adv_images[i].permute(1, 2, 0).cpu().numpy()
        adv_img = np.clip(adv_img, 0, 1)
        axes[1, i].imshow(adv_img)
        axes[1, i].set_title(f'Adv: {classes[predictions[i]]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_attack_results(results, save_path):
    """Plot attack results"""
    attack_names = list(results.keys())
    clean_accs = [results[name]['clean_accuracy'] for name in attack_names]
    adv_accs = [results[name]['adversarial_accuracy'] for name in attack_names]
    accuracy_drops = [results[name]['accuracy_drop'] for name in attack_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    x = np.arange(len(attack_names))
    width = 0.35
    
    ax1.bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
    ax1.bar(x + width/2, adv_accs, width, label='Adversarial Accuracy', alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Clean vs Adversarial Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(attack_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy drop
    ax2.bar(attack_names, accuracy_drops, alpha=0.8, color='red')
    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Accuracy Drop due to Attacks')
    ax2.set_xticklabels(attack_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Build model
    model = build_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='Test Vision Transformer against adversarial attacks')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', help='Output directory (optional)')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to test (optional)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), "../test_results")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/adversarial_examples", exist_ok=True)
    
    # Copy config file
    shutil.copy(args.config, f"{output_dir}/config.yaml")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, model_config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Print model summary
    model_summary = get_model_summary(model, model_config)
    print("\nModel Summary:")
    for key, value in model_summary.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    _, test_loader, _ = get_data_loaders(config)
    
    # Run attacks
    test_config = config['testing']
    attack_configs = test_config['attacks']
    
    print(f"\nRunning {len(attack_configs)} attacks...")
    
    results = {}
    all_clean_images = []
    all_adv_images = {}
    all_predictions = {}
    all_labels = []
    
    for attack_config in attack_configs:
        attack_name = attack_config['name']
        print(f"\nEvaluating attack: {attack_name}")
        
        # Create attack
        attack = create_attack(attack_config)
        
        # Evaluate attack
        attack_results = evaluate_attack(
            model, test_loader, attack, device, max_samples=args.max_samples
        )
        results[attack_name] = attack_results
        
        print(f"  Clean Accuracy: {attack_results['clean_accuracy']:.2f}%")
        print(f"  Adversarial Accuracy: {attack_results['adversarial_accuracy']:.2f}%")
        print(f"  Accuracy Drop: {attack_results['accuracy_drop']:.2f}%")
        
        # Collect examples for visualization (first attack only to save memory)
        if len(all_clean_images) == 0 and test_config.get('save_adversarial_examples', False):
            print("  Collecting adversarial examples...")
            model.eval()
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx >= 1:  # Only first batch
                        break
                        
                    images, labels = images.to(device), labels.to(device)
                    
                    # Generate adversarial examples
                    adv_images = attack(images)
                    
                    # Get predictions
                    adv_outputs = model(adv_images)
                    adv_preds = torch.argmax(adv_outputs, dim=1)
                    
                    # Store for visualization
                    all_clean_images = images.cpu()
                    all_labels = labels.cpu()
                    all_adv_images[attack_name] = adv_images.cpu()
                    all_predictions[attack_name] = adv_preds.cpu()
                    break
    
    # Save results
    print(f"\nSaving results to: {output_dir}")
    
    # Save numerical results
    with open(f"{output_dir}/attack_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    if test_config.get('save_plots', True):
        plot_attack_results(results, f"{output_dir}/attack_results.png")
    
    # Save adversarial examples
    if test_config.get('save_adversarial_examples', False) and (hasattr(all_clean_images, 'numel') and all_clean_images.numel() > 0):
        for attack_name in all_adv_images:
            display_adversarial_examples(
                all_clean_images, all_adv_images[attack_name],
                all_predictions[attack_name], all_labels,
                f"{output_dir}/adversarial_examples/{attack_name}_examples.png"
            )
    
    # Print summary
    print("\n" + "="*60)
    print("ATTACK EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Attack Name':<20} {'Clean Acc':<12} {'Adv Acc':<12} {'Drop':<12}")
    print("-"*60)
    
    for attack_name, result in results.items():
        print(f"{attack_name:<20} {result['clean_accuracy']:<12.2f} "
              f"{result['adversarial_accuracy']:<12.2f} {result['accuracy_drop']:<12.2f}")
    
    print("-"*60)
    
    # Calculate average metrics
    avg_clean_acc = np.mean([r['clean_accuracy'] for r in results.values()])
    avg_adv_acc = np.mean([r['adversarial_accuracy'] for r in results.values()])
    avg_drop = np.mean([r['accuracy_drop'] for r in results.values()])
    
    print(f"{'Average':<20} {avg_clean_acc:<12.2f} {avg_adv_acc:<12.2f} {avg_drop:<12.2f}")
    print("="*60)
    
    # Save summary
    summary = {
        'model_summary': model_summary,
        'attack_results': results,
        'average_metrics': {
            'clean_accuracy': avg_clean_acc,
            'adversarial_accuracy': avg_adv_acc,
            'accuracy_drop': avg_drop
        }
    }
    
    with open(f"{output_dir}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
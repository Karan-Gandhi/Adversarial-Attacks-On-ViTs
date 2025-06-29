#!/usr/bin/env python3
"""
Sample experiment script demonstrating how to run experiments with different positional encodings.
This script will train models with different PE configurations and compare their robustness.
"""

import os
import yaml
import subprocess
import json
from datetime import datetime


def create_config(base_config, pe_type, pe_mode="add", **kwargs):
    """Create a config with specific positional encoding settings"""
    config = base_config.copy()
    
    config['model']['positional_encoding']['type'] = pe_type
    config['model']['positional_encoding']['mode'] = pe_mode
    
    # Update specific PE parameters
    for key, value in kwargs.items():
        if '.' in key:
            keys = key.split('.')
            target = config['model']['positional_encoding']
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = value
        else:
            config['model']['positional_encoding'][key] = value
    
    return config


def save_config(config, filepath):
    """Save config to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def run_experiment(config_path, experiment_name):
    """Run training and testing for a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Run training
    print("Starting training...")
    train_cmd = f"python train.py --config {config_path}"
    result = subprocess.run(train_cmd.split(), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Training failed for {experiment_name}")
        print(result.stderr)
        return None
    
    # Find the output directory (most recent)
    output_dirs = [d for d in os.listdir('outputs') if d.startswith(experiment_name)]
    if not output_dirs:
        print(f"No output directory found for {experiment_name}")
        return None
    
    latest_output = max(output_dirs)
    checkpoint_path = f"outputs/{latest_output}/checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    
    # Run testing
    print("Starting testing...")
    test_output_dir = f"test_results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_cmd = f"python test.py --config {config_path} --checkpoint {checkpoint_path} --output-dir {test_output_dir}"
    result = subprocess.run(test_cmd.split(), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Testing failed for {experiment_name}")
        print(result.stderr)
        return None
    
    print(f"Experiment {experiment_name} completed successfully!")
    return test_output_dir


def compare_results(result_dirs):
    """Compare results from multiple experiments"""
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    
    all_results = {}
    
    for exp_name, result_dir in result_dirs.items():
        if result_dir is None:
            continue
            
        try:
            with open(f"{result_dir}/evaluation_summary.json", 'r') as f:
                summary = json.load(f)
                all_results[exp_name] = summary['average_metrics']
        except Exception as e:
            print(f"Failed to load results for {exp_name}: {e}")
    
    if not all_results:
        print("No results to compare")
        return
    
    # Print comparison table
    print(f"{'Experiment':<25} {'Clean Acc':<12} {'Adv Acc':<12} {'Avg Drop':<12}")
    print("-" * 65)
    
    for exp_name, metrics in all_results.items():
        print(f"{exp_name:<25} {metrics['clean_accuracy']:<12.2f} "
              f"{metrics['adversarial_accuracy']:<12.2f} {metrics['accuracy_drop']:<12.2f}")
    
    print("-" * 65)
    
    # Find best performing model
    best_robust = min(all_results.items(), key=lambda x: x[1]['accuracy_drop'])
    best_clean = max(all_results.items(), key=lambda x: x[1]['clean_accuracy'])
    
    print(f"\nMost robust model: {best_robust[0]} (avg drop: {best_robust[1]['accuracy_drop']:.2f}%)")
    print(f"Highest clean accuracy: {best_clean[0]} (clean acc: {best_clean[1]['clean_accuracy']:.2f}%)")


def main():
    # Create necessary directories
    os.makedirs('experiment_configs', exist_ok=True)
    os.makedirs('test_results', exist_ok=True)
    
    # Load base configuration
    with open('configs/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Reduce epochs for quick experiments
    base_config['training']['num_epochs'] = 20
    base_config['experiment']['name'] = 'pe_comparison'
    
    # Define experiments
    experiments = {
        'sincos_add': {
            'pe_type': '2d_sincos',
            'pe_mode': 'add'
        },
        'sincos_concat': {
            'pe_type': '2d_sincos', 
            'pe_mode': 'concat',
            'pos_dim': 64,
            'proj_dim': 192
        },
        'rope': {
            'pe_type': '2d_rope',
            'pe_mode': 'add'
        },
        'stft_add': {
            'pe_type': 'stft',
            'pe_mode': 'add'
        },
        'wavelets_add': {
            'pe_type': 'wavelets',
            'pe_mode': 'add'
        }
    }
    
    # Run experiments
    result_dirs = {}
    
    for exp_name, exp_config in experiments.items():
        # Create config
        config = create_config(base_config, **exp_config)
        config['experiment']['name'] = f"pe_comparison_{exp_name}"
        
        # Save config
        config_path = f"experiment_configs/{exp_name}_config.yaml"
        save_config(config, config_path)
        
        # Run experiment
        result_dir = run_experiment(config_path, f"pe_comparison_{exp_name}")
        result_dirs[exp_name] = result_dir
    
    # Compare results
    compare_results(result_dirs)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Check the test_results/ directory for detailed results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify the experimentation pipeline setup
"""

import sys
import os
import traceback

def test_imports():
    """Test all major imports"""
    tests = []
    
    # Test basic imports
    try:
        import torch
        import yaml
        import numpy as np
        tests.append(("Basic imports (torch, yaml, numpy)", True, ""))
    except Exception as e:
        tests.append(("Basic imports", False, str(e)))
    
    # Test model imports
    try:
        from model.vit import ViT
        from model.model_builder import build_model
        tests.append(("Model imports", True, ""))
    except Exception as e:
        tests.append(("Model imports", False, str(e)))
    
    # Test PE imports
    try:
        from model.PE.sincos_2d import SinCos2DPositionalEncoding
        from model.PE.rope_2d import RoPE2DPositionalEncoding
        from model.PE.stft import STFTPositionalEncoding
        from model.PE.wavelets import WaveletPositionalEncoding
        tests.append(("Positional Encoding imports", True, ""))
    except Exception as e:
        tests.append(("Positional Encoding imports", False, str(e)))
    
    # Test other imports
    try:
        from dataset_utils import get_data_loaders
        from attacks import create_attack
        tests.append(("Utility imports", True, ""))
    except Exception as e:
        tests.append(("Utility imports", False, str(e)))
    
    return tests


def test_config_loading():
    """Test config loading"""
    try:
        with open('configs/config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        return True, "Config loaded successfully"
    except Exception as e:
        return False, f"Config loading failed: {str(e)}"


def test_model_building():
    """Test model building"""
    try:
        with open('configs/config.yaml', 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        from model.model_builder import build_model
        model = build_model(config)
        return True, f"Model built successfully with {sum(p.numel() for p in model.parameters())} parameters"
    except Exception as e:
        return False, f"Model building failed: {str(e)}"


def main():
    print("="*60)
    print("EXPERIMENTATION PIPELINE SETUP TEST")
    print("="*60)
    
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Test imports
    print("Testing imports...")
    import_tests = test_imports()
    for test_name, success, error in import_tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            print(f"    Error: {error}")
    
    print()
    
    # Test config loading
    print("Testing config loading...")
    config_success, config_msg = test_config_loading()
    status = "✓ PASS" if config_success else "✗ FAIL"
    print(f"  Config loading: {status}")
    if not config_success:
        print(f"    Error: {config_msg}")
    else:
        print(f"    {config_msg}")
    
    print()
    
    # Test model building
    print("Testing model building...")
    model_success, model_msg = test_model_building()
    status = "✓ PASS" if model_success else "✗ FAIL"
    print(f"  Model building: {status}")
    if not model_success:
        print(f"    Error: {model_msg}")
    else:
        print(f"    {model_msg}")
    
    print()
    print("="*60)
    
    all_tests_passed = all([success for _, success, _ in import_tests]) and config_success and model_success
    
    if all_tests_passed:
        print("✓ ALL TESTS PASSED - Setup is ready!")
        print("\nYou can now run experiments using:")
        print("  python train.py --config configs/config.yaml")
        print("  python test.py --config configs/config.yaml --checkpoint path/to/model.pth")
        print("  python run_experiments.py  # For automated experiments")
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        
    print("="*60)


if __name__ == "__main__":
    main()

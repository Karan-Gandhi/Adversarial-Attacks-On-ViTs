# Vision Transformer Adversarial Attack Experimentation Pipeline

This pipeline provides a comprehensive framework for training Vision Transformers with different positional encodings and evaluating their robustness against various adversarial attacks.

## Features

- **Multiple Positional Encodings**: 2D SinCos, 2D RoPE, STFT-based, and Wavelet-based positional encodings
- **Flexible Integration Modes**: Add or concatenate positional encodings (except RoPE which only supports add mode)
- **Comprehensive Attack Suite**: Phase, magnitude, frequency-based, pixel-based, and noise-based attacks
- **Complete Training Pipeline**: Training with checkpointing, early stopping, and tensorboard logging
- **Evaluation Framework**: Systematic evaluation against multiple attacks with visualization

## Directory Structure

```
experimentation pipeline/
├── configs/
│   └── config.yaml              # Configuration file
├── model/
│   ├── vit.py                   # Base Vision Transformer model
│   ├── model_builder.py         # Model building utilities
│   ├── model_utils.py           # Additional model utilities
│   └── PE/                      # Positional Encoding modules
│       ├── sincos_2d.py         # 2D Sinusoidal encodings
│       ├── rope_2d.py           # 2D Rotary Position Embeddings
│       ├── stft.py              # STFT-based encodings
│       └── wavelets.py          # Wavelet-based encodings
├── attacks.py                   # Adversarial attack implementations
├── dataset_utils.py            # Data loading utilities
├── train.py                    # Training script
├── test.py                     # Testing/evaluation script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The `configs/config.yaml` file contains all experiment parameters:

### Model Configuration
- `image_size`: Input image size (default: 32 for CIFAR-10)
- `patch_size`: Patch size for tokenization (default: 4)
- `num_classes`: Number of output classes (default: 10 for CIFAR-10)
- `dim`: Model dimension (default: 256)
- `depth`: Number of transformer layers (default: 6)
- `heads`: Number of attention heads (default: 8)
- `mlp_dim`: MLP hidden dimension (default: 512)

### Positional Encoding Configuration
- `type`: Type of positional encoding (`2d_sincos`, `2d_rope`, `stft`, `wavelets`)
- `mode`: Integration mode (`add` or `concat`) - RoPE only supports `add`
- `pos_dim`: Dimension for positional encoding (only for concat mode)
- `proj_dim`: Dimension for patch projection (only for concat mode)

### Training Configuration
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `optimizer`: Optimizer type (currently supports `adamw`)
- `scheduler`: Learning rate scheduler (currently supports `cosine`)
- Early stopping, checkpointing, and logging parameters

### Attack Configuration
Multiple attacks can be configured with different parameters:
- `phase`: Phase-based frequency domain attacks
- `magnitude`: Magnitude-based frequency domain attacks
- `low_freq`/`high_freq`: Frequency-selective attacks
- `pixel`: Pixel-wise perturbations
- `normal`: Gaussian noise attacks

## Usage

### Training a Model

```bash
python train.py --config configs/config.yaml
```

Optional arguments:
- `--resume path/to/checkpoint.pth`: Resume training from a checkpoint

### Testing Against Attacks

```bash
python test.py --config configs/config.yaml --checkpoint path/to/model.pth
```

Optional arguments:
- `--output-dir results/`: Specify output directory
- `--max-samples 1000`: Limit number of test samples

## Positional Encoding Types

### 1. 2D Sinusoidal (2d_sincos)
- Traditional sinusoidal positional encoding adapted for 2D image patches
- Supports both add and concat modes
- Uses separate frequencies for height and width dimensions

### 2. 2D Rotary Position Embeddings (2d_rope)
- Rotary Position Embeddings adapted for 2D vision tasks
- Only supports add mode (applied during attention computation)
- Provides relative position information

### 3. STFT-based (stft)
- Positional encoding based on Short-Time Fourier Transform features
- Captures frequency domain characteristics of spatial positions
- Configurable window size, hop length, and FFT size

### 4. Wavelet-based (wavelets)
- Uses wavelet decomposition to generate positional features
- Configurable wavelet type (default: Daubechies-4) and decomposition levels
- Captures multi-scale spatial relationships

## Integration Modes

### Add Mode
- Positional encodings are added to patch embeddings
- Preserves original embedding dimension
- Standard approach used in most transformer architectures

### Concat Mode
- Positional encodings are concatenated with patch embeddings
- Requires splitting the model dimension between patch projection and positional encoding
- Allows model to learn separate representations for content and position

**Note**: RoPE doesn't support concat mode as it modifies the attention computation directly.

## Attack Types

1. **Phase Attack**: Perturbs the phase component in frequency domain
2. **Magnitude Attack**: Perturbs the magnitude component in frequency domain
3. **Low/High Frequency**: Targets specific frequency ranges
4. **Pixel Attack**: Direct pixel-wise perturbations
5. **Normal Noise**: Gaussian noise perturbations

## Output Structure

### Training Outputs
```
outputs/experiment_name_timestamp/
├── config.yaml                 # Copy of configuration
├── model_summary.json          # Model architecture summary
├── training_history.json       # Training metrics history
├── training_curves.png         # Loss and accuracy plots
├── checkpoints/
│   ├── best_model.pth          # Best model checkpoint
│   ├── final_model.pth         # Final model checkpoint
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
└── logs/                       # Tensorboard logs
```

### Testing Outputs
```
test_results_timestamp/
├── config.yaml                 # Copy of configuration
├── attack_results.json         # Numerical results
├── attack_results.png          # Results visualization
├── evaluation_summary.json     # Complete evaluation summary
└── adversarial_examples/       # Example images (if enabled)
    └── attack_name_examples.png
```

## Example Experiments

### Experiment 1: Compare Positional Encodings
Create different config files with different PE types and train models:

```yaml
# config_sincos.yaml
model:
  positional_encoding:
    type: "2d_sincos"
    mode: "add"

# config_rope.yaml  
model:
  positional_encoding:
    type: "2d_rope"
    mode: "add"

# config_stft.yaml
model:
  positional_encoding:
    type: "stft"
    mode: "add"
    stft:
      window_size: 16
      hop_length: 8
      n_fft: 32
```

### Experiment 2: Add vs Concat Mode
Compare different integration modes:

```yaml
# Add mode
model:
  dim: 256
  positional_encoding:
    type: "2d_sincos"
    mode: "add"

# Concat mode
model:
  dim: 256
  positional_encoding:
    type: "2d_sincos"
    mode: "concat"
    pos_dim: 64
    proj_dim: 192
```
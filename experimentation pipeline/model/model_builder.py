import torch
from .vit import ViT
from .PE.sincos_2d import SinCos2DPositionalEncoding, SinCos2DPositionalEncodingConcat
from .PE.rope_2d import RoPE2DPositionalEncoding
from .PE.stft import STFTPositionalEncoding, STFTPositionalEncodingConcat
from .PE.wavelets import WaveletPositionalEncoding, WaveletPositionalEncodingConcat


def build_model(config):
    """Build ViT model with specified positional encoding"""
    model_config = config['model']
    pe_config = model_config['positional_encoding']
    
    # Extract model parameters
    image_size = model_config['image_size']
    patch_size = model_config['patch_size']
    num_classes = model_config['num_classes']
    dim = model_config['dim']
    depth = model_config['depth']
    heads = model_config['heads']
    mlp_dim = model_config['mlp_dim']
    channels = model_config['channels']
    dropout = model_config['dropout']
    
    # Calculate patch dimensions
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    
    # Create model
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        channels=channels,
        dropout=dropout,
        positional_encoding=pe_config
    )
    
    # Create and set positional encoder based on type and mode
    pe_type = pe_config['type']
    pe_mode = pe_config['mode']
    
    if pe_type == '2d_sincos':
        if pe_mode == 'add':
            pos_encoder = SinCos2DPositionalEncoding(dim, h_patches, w_patches)
        elif pe_mode == 'concat':
            pos_dim = pe_config['pos_dim']
            pos_encoder = SinCos2DPositionalEncodingConcat(pos_dim, h_patches, w_patches)
        else:
            raise ValueError(f"Unsupported mode '{pe_mode}' for 2D SinCos encoding")
            
    elif pe_type == '2d_rope':
        if pe_mode != 'add':
            raise ValueError("RoPE only supports 'add' mode")
        pos_encoder = RoPE2DPositionalEncoding(dim, h_patches, w_patches)
        
    elif pe_type == 'stft':
        stft_config = pe_config['stft']
        if pe_mode == 'add':
            pos_encoder = STFTPositionalEncoding(
                dim=dim,
                h=h_patches,
                w=w_patches,
                window_size=stft_config['window_size'],
                hop_length=stft_config['hop_length'],
                n_fft=stft_config['n_fft'],
                mode='add'
            )
        elif pe_mode == 'concat':
            pos_dim = pe_config['pos_dim']
            pos_encoder = STFTPositionalEncodingConcat(
                pos_dim=pos_dim,
                h=h_patches,
                w=w_patches,
                window_size=stft_config['window_size'],
                hop_length=stft_config['hop_length'],
                n_fft=stft_config['n_fft']
            )
        else:
            raise ValueError(f"Unsupported mode '{pe_mode}' for STFT encoding")
            
    elif pe_type == 'wavelets':
        wavelet_config = pe_config['wavelets']
        if pe_mode == 'add':
            pos_encoder = WaveletPositionalEncoding(
                dim=dim,
                h=h_patches,
                w=w_patches,
                wavelet_type=wavelet_config['wavelet_type'],
                levels=wavelet_config['levels'],
                mode='add'
            )
        elif pe_mode == 'concat':
            pos_dim = pe_config['pos_dim']
            pos_encoder = WaveletPositionalEncodingConcat(
                pos_dim=pos_dim,
                h=h_patches,
                w=w_patches,
                wavelet_type=wavelet_config['wavelet_type'],
                levels=wavelet_config['levels']
            )
        else:
            raise ValueError(f"Unsupported mode '{pe_mode}' for wavelet encoding")
            
    else:
        raise ValueError(f"Unsupported positional encoding type: {pe_type}")
    
    # Set the positional encoder
    model.set_positional_encoder(pos_encoder)
    
    return model


def count_parameters(model):
    """Count the total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model, config):
    """Get a summary of the model"""
    total_params = count_parameters(model)
    pe_config = config['model']['positional_encoding']
    
    summary = {
        'model_type': 'Vision Transformer',
        'positional_encoding': f"{pe_config['type']} ({pe_config['mode']} mode)",
        'total_parameters': total_params,
        'image_size': config['model']['image_size'],
        'patch_size': config['model']['patch_size'],
        'dim': config['model']['dim'],
        'depth': config['model']['depth'],
        'heads': config['model']['heads']
    }
    
    return summary

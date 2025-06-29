import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class STFTPositionalEncoding(nn.Module):
    """STFT-based Positional Encoding for Vision Transformer"""
    
    def __init__(self, dim, h=8, w=8, window_size=16, hop_length=8, n_fft=32, mode='add', pos_dim=None):
        super(STFTPositionalEncoding, self).__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.mode = mode
        self.pos_dim = pos_dim if mode == 'concat' else dim
        
        # Precompute STFT-based positional encodings
        self._precompute_stft_encodings()
        
    def _precompute_stft_encodings(self):
        """Precompute STFT-based positional encodings"""
        # Create position signals for height and width
        y_signal = torch.sin(2 * math.pi * torch.arange(self.h).float() / self.h)
        x_signal = torch.sin(2 * math.pi * torch.arange(self.w).float() / self.w)
        
        # Apply STFT to position signals
        y_stft = torch.stft(
            y_signal, 
            n_fft=min(self.n_fft, len(y_signal)), 
            hop_length=min(self.hop_length, len(y_signal)//2),
            window=torch.hann_window(min(self.window_size, len(y_signal))),
            return_complex=True
        )
        
        x_stft = torch.stft(
            x_signal, 
            n_fft=min(self.n_fft, len(x_signal)), 
            hop_length=min(self.hop_length, len(x_signal)//2),
            window=torch.hann_window(min(self.window_size, len(x_signal))),
            return_complex=True
        )
        
        # Extract magnitude and phase features
        y_mag = torch.abs(y_stft).flatten()
        y_phase = torch.angle(y_stft).flatten()
        x_mag = torch.abs(x_stft).flatten()
        x_phase = torch.angle(x_stft).flatten()
        
        # Combine features
        stft_features = torch.cat([y_mag, y_phase, x_mag, x_phase])
        
        # Create positional encodings for each patch
        pos_encodings = []
        for i in range(self.h):
            for j in range(self.w):
                # Create unique encoding for each position based on STFT features
                pos_idx = i * self.w + j
                
                # Use position-dependent features from STFT
                pos_encoding = torch.zeros(self.pos_dim)
                
                # Fill encoding with STFT-derived features
                feature_idx = pos_idx % len(stft_features)
                for k in range(self.pos_dim):
                    if k % 4 == 0:  # Magnitude component
                        pos_encoding[k] = stft_features[feature_idx] * torch.sin(k * pos_idx / self.pos_dim)
                    elif k % 4 == 1:  # Phase component
                        pos_encoding[k] = stft_features[feature_idx] * torch.cos(k * pos_idx / self.pos_dim)
                    elif k % 4 == 2:  # Position-dependent sine
                        pos_encoding[k] = torch.sin(2 * math.pi * pos_idx / (self.h * self.w) + k / self.pos_dim)
                    else:  # Position-dependent cosine
                        pos_encoding[k] = torch.cos(2 * math.pi * pos_idx / (self.h * self.w) + k / self.pos_dim)
                        
                pos_encodings.append(pos_encoding)
        
        # Stack all positional encodings
        pos_enc = torch.stack(pos_encodings, dim=0)  # (h*w, pos_dim)
        
        if self.mode == 'add':
            # Add class token encoding
            cls_pos_enc = torch.zeros(1, self.pos_dim)
            pos_enc = torch.cat([cls_pos_enc, pos_enc], dim=0)
            self.register_buffer('pos_enc', pos_enc.unsqueeze(0))
        else:
            # For concat mode, don't include class token
            self.register_buffer('pos_enc', pos_enc)
            
    def forward(self, x):
        """Apply STFT positional encoding (add mode)"""
        if self.mode == 'add':
            return x + self.pos_enc[:, 1:, :]  # Skip class token position
        else:
            raise NotImplementedError("Use get_patch_encodings() for concat mode")
            
    def get_patch_encodings(self, batch_size):
        """Get positional encodings for patches (concat mode)"""
        if self.mode == 'concat':
            return self.pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            raise NotImplementedError("get_patch_encodings() only for concat mode")


class STFTPositionalEncodingConcat(STFTPositionalEncoding):
    """STFT Positional Encoding specifically for concatenation mode"""
    
    def __init__(self, pos_dim, h=8, w=8, window_size=16, hop_length=8, n_fft=32):
        super().__init__(
            dim=None, h=h, w=w, window_size=window_size, 
            hop_length=hop_length, n_fft=n_fft, mode='concat', pos_dim=pos_dim
        )

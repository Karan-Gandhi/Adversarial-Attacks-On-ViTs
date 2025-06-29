import torch
import torch.nn as nn
import pywt
import numpy as np
import math


class WaveletPositionalEncoding(nn.Module):
    """Wavelet-based Positional Encoding for Vision Transformer"""
    
    def __init__(self, dim, h=8, w=8, wavelet_type='db4', levels=3, mode='add', pos_dim=None):
        super(WaveletPositionalEncoding, self).__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        self.pos_dim = pos_dim if mode == 'concat' else dim
        
        # Precompute wavelet-based positional encodings
        self._precompute_wavelet_encodings()
        
    def _precompute_wavelet_encodings(self):
        """Precompute wavelet-based positional encodings"""
        # Create 2D position grid
        y_pos = np.arange(self.h)
        x_pos = np.arange(self.w)
        
        # Create 2D position signals
        pos_grid = np.zeros((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                pos_grid[i, j] = i * self.w + j
                
        # Normalize position grid
        pos_grid = pos_grid / (self.h * self.w - 1)
        
        # Apply wavelet decomposition
        try:
            coeffs = pywt.wavedec2(pos_grid, self.wavelet_type, level=self.levels)
            
            # Extract wavelet coefficients and flatten
            wavelet_features = []
            
            # Approximation coefficients
            if len(coeffs) > 0:
                approx = coeffs[0]
                wavelet_features.extend(approx.flatten())
            
            # Detail coefficients
            for i in range(1, len(coeffs)):
                if isinstance(coeffs[i], tuple) and len(coeffs[i]) == 3:
                    cH, cV, cD = coeffs[i]
                    wavelet_features.extend(cH.flatten())
                    wavelet_features.extend(cV.flatten())
                    wavelet_features.extend(cD.flatten())
                    
        except Exception as e:
            # Fallback: use simple sine/cosine based on position
            print(f"Wavelet decomposition failed: {e}. Using fallback encoding.")
            wavelet_features = []
            for i in range(self.h * self.w):
                wavelet_features.append(np.sin(2 * np.pi * i / (self.h * self.w)))
                wavelet_features.append(np.cos(2 * np.pi * i / (self.h * self.w)))
        
        wavelet_features = np.array(wavelet_features)
        
        # Create positional encodings for each patch
        pos_encodings = []
        for i in range(self.h):
            for j in range(self.w):
                pos_idx = i * self.w + j
                pos_encoding = np.zeros(self.pos_dim)
                
                # Fill encoding with wavelet-derived features
                for k in range(self.pos_dim):
                    if len(wavelet_features) > 0:
                        # Use wavelet features cyclically
                        feature_idx = (pos_idx + k) % len(wavelet_features)
                        base_feature = wavelet_features[feature_idx]
                        
                        # Apply different transformations based on position in encoding
                        if k % 4 == 0:
                            pos_encoding[k] = base_feature * np.sin(k * pos_idx / self.pos_dim)
                        elif k % 4 == 1:
                            pos_encoding[k] = base_feature * np.cos(k * pos_idx / self.pos_dim)
                        elif k % 4 == 2:
                            pos_encoding[k] = np.tanh(base_feature + k / self.pos_dim)
                        else:
                            pos_encoding[k] = base_feature * np.sin(2 * np.pi * pos_idx / (self.h * self.w))
                    else:
                        # Fallback to sinusoidal encoding
                        pos_encoding[k] = np.sin(2 * np.pi * pos_idx / (self.h * self.w) + k / self.pos_dim)
                        
                pos_encodings.append(pos_encoding)
        
        # Convert to torch tensor
        pos_enc = torch.tensor(np.stack(pos_encodings, axis=0), dtype=torch.float32)
        
        if self.mode == 'add':
            # Add class token encoding
            cls_pos_enc = torch.zeros(1, self.pos_dim)
            pos_enc = torch.cat([cls_pos_enc, pos_enc], dim=0)
            self.register_buffer('pos_enc', pos_enc.unsqueeze(0))
        else:
            # For concat mode, don't include class token
            self.register_buffer('pos_enc', pos_enc)
            
    def forward(self, x):
        """Apply wavelet positional encoding (add mode)"""
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


class WaveletPositionalEncodingConcat(WaveletPositionalEncoding):
    """Wavelet Positional Encoding specifically for concatenation mode"""
    
    def __init__(self, pos_dim, h=8, w=8, wavelet_type='db4', levels=3):
        super().__init__(
            dim=None, h=h, w=w, wavelet_type=wavelet_type, 
            levels=levels, mode='concat', pos_dim=pos_dim
        )

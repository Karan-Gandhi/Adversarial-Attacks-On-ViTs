import torch
import torch.nn as nn
import math


class SinCos2DPositionalEncoding(nn.Module):
    """2D Sinusoidal Positional Encoding for Vision Transformer (Add mode)"""
    
    def __init__(self, dim, h=8, w=8):
        super(SinCos2DPositionalEncoding, self).__init__()
        
        # Create 2D positional encoding
        pos_enc = torch.zeros(h, w, dim)
        
        # Separate channels for width and height dimensions
        dim_h = dim // 2
        dim_w = dim // 2
        
        # Position indices
        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w).reshape(h, w)
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1).reshape(h, w)
        
        # Create division term for computing positional encoding values
        div_term_h = torch.exp(torch.arange(0, dim_h, 2).float() * -(math.log(10000.0) / dim_h))
        div_term_w = torch.exp(torch.arange(0, dim_w, 2).float() * -(math.log(10000.0) / dim_w))
        
        # Apply sin and cos to odd and even indices
        for i in range(0, dim_h, 2):
            if i < dim_h:
                pos_enc[:, :, i] = torch.sin(y_pos.float() * div_term_h[i//2])
                pos_enc[:, :, i+1] = torch.cos(y_pos.float() * div_term_h[i//2])
            
        for i in range(0, dim_w, 2):
            if i + dim_h < dim:
                pos_enc[:, :, i+dim_h] = torch.sin(x_pos.float() * div_term_w[i//2])
                pos_enc[:, :, i+dim_h+1] = torch.cos(x_pos.float() * div_term_w[i//2])
        
        # Flatten the positional encoding to match the sequence format (h*w, dim)
        pos_enc = pos_enc.reshape(h * w, dim)
        
        # Add extra position for class token
        cls_pos_enc = torch.zeros(1, dim)
        pos_enc = torch.cat([cls_pos_enc, pos_enc], dim=0)
        
        # Register as buffer (persistent but not model parameter)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pos_enc[:, 1:, :]  # Skip class token position


class SinCos2DPositionalEncodingConcat(nn.Module):
    """2D Sinusoidal Positional Encoding for Vision Transformer (Concat mode)"""
    
    def __init__(self, pos_dim, h, w):
        super(SinCos2DPositionalEncodingConcat, self).__init__()
        self.pos_dim = pos_dim
        self.h = h
        self.w = w
        
        # Create position indices
        y_pos = torch.arange(h).unsqueeze(1).repeat(1, w).reshape(h, w)
        x_pos = torch.arange(w).unsqueeze(0).repeat(h, 1).reshape(h, w)
        
        # Split dimensions for height and width
        dim_h = pos_dim // 2
        dim_w = pos_dim - dim_h  # In case pos_dim is odd
        
        # Division terms for computing positional encoding
        div_term_h = torch.exp(torch.arange(0, dim_h, 2).float() * -(math.log(10000.0) / dim_h))
        div_term_w = torch.exp(torch.arange(0, dim_w, 2).float() * -(math.log(10000.0) / dim_w))
        
        # Create positional encoding tensor
        pos_enc = torch.zeros(h, w, pos_dim)
        
        # Apply sin and cos to encode height positions
        for i in range(0, dim_h, 2):
            if i < dim_h:
                pos_enc[:, :, i] = torch.sin(y_pos.float() * div_term_h[i//2])
                if i + 1 < dim_h:
                    pos_enc[:, :, i+1] = torch.cos(y_pos.float() * div_term_h[i//2])
        
        # Apply sin and cos to encode width positions
        for i in range(0, dim_w, 2):
            if i + dim_h < pos_dim:
                pos_enc[:, :, i+dim_h] = torch.sin(x_pos.float() * div_term_w[i//2])
                if i + dim_h + 1 < pos_dim:
                    pos_enc[:, :, i+dim_h+1] = torch.cos(x_pos.float() * div_term_w[i//2])
        
        # Reshape to (h*w, pos_dim)
        pos_enc = pos_enc.reshape(h * w, pos_dim)
        
        # Register as buffer
        self.register_buffer('pos_enc', pos_enc)
        
    def get_patch_encodings(self, batch_size):
        """Get positional encodings for patches (excluding class token)"""
        return self.pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        
    def forward(self, x):
        # This shouldn't be called for concat mode
        raise NotImplementedError("Use get_patch_encodings() for concat mode")

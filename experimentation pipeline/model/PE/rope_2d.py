import torch
import torch.nn as nn
import math


def rotate_half(x):
    """Helper function to rotate half the dimensions"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    """Apply rotary positional embedding to queries and keys"""
    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    return q_embed, k_embed


class RoPE2DPositionalEncoding(nn.Module):
    """2D Rotary Positional Encoding (RoPE) for Vision Transformer"""
    
    def __init__(self, dim, h=8, w=8, theta=10000.0):
        super(RoPE2DPositionalEncoding, self).__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.theta = theta
        
        # RoPE frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute position grids
        y_pos = torch.arange(h).float()
        x_pos = torch.arange(w).float()
        
        # Create 2D position grid
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Flatten to match patch sequence
        self.register_buffer('grid_y', grid_y.flatten())
        self.register_buffer('grid_x', grid_x.flatten())
        
    def get_freqs(self, seq_len, device):
        """Get frequency tensors for the given sequence length"""
        # Use half frequencies for each dimension (y and x)
        freqs_y = torch.outer(self.grid_y[:seq_len], self.freqs[:self.dim//4]).to(device)
        freqs_x = torch.outer(self.grid_x[:seq_len], self.freqs[:self.dim//4]).to(device)
        
        # Combine y and x frequencies
        freqs = torch.cat([freqs_y, freqs_x], dim=-1)
        
        # Compute cos and sin
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin
        
    def forward(self, x):
        """
        Note: RoPE is typically applied in the attention mechanism,
        not as a simple addition to embeddings. This method is for compatibility.
        """
        # For RoPE, we don't add anything to the embeddings
        # The rotary encoding is applied in the attention computation
        return x
        
    def apply_rope_to_attention(self, q, k, v):
        """Apply RoPE to query and key tensors during attention computation"""
        # q, k shape: (batch, heads, seq_len, head_dim)
        seq_len = q.size(2) - 1  # Exclude class token
        
        # Get frequencies for patches (excluding class token)
        freqs_cos, freqs_sin = self.get_freqs(seq_len, q.device)
        
        # Add batch and head dimensions to freqs
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)
        
        # Split q and k into class token and patches
        q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
        k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]
        
        # Apply RoPE to patches only
        q_patches_rope, k_patches_rope = apply_rotary_pos_emb(
            q_patches, k_patches, freqs_cos, freqs_sin
        )
        
        # Concatenate class token back
        q_rope = torch.cat([q_cls, q_patches_rope], dim=2)
        k_rope = torch.cat([k_cls, k_patches_rope], dim=2)
        
        return q_rope, k_rope, v

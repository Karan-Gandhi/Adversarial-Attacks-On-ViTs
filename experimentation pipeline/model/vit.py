import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Will be set by the model if using RoPE
        self.rope_encoder = None
        
    def set_rope_encoder(self, rope_encoder):
        """Set RoPE encoder for rotary position embeddings"""
        self.rope_encoder = rope_encoder
        
    def forward(self, x):
        b, n, c = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, self.head_dim).transpose(1, 2), qkv)
        
        # Apply RoPE if available
        if self.rope_encoder is not None:
            q, k, v = self.rope_encoder.apply_rope_to_attention(q, k, v)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to v
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.to_out(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self, 
        image_size=32, 
        patch_size=4, 
        num_classes=10, 
        dim=256, 
        depth=6, 
        heads=8, 
        mlp_dim=512, 
        channels=3, 
        dropout=0.1,
        positional_encoding=None
    ):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        
        self.patch_size = patch_size
        self.h_patches = image_size // patch_size
        self.w_patches = image_size // patch_size
        self.positional_encoding = positional_encoding
        
        # Determine patch embedding dimension based on PE mode
        if positional_encoding and positional_encoding.get('mode') == 'concat':
            proj_dim = positional_encoding.get('proj_dim', dim)
            pos_dim = positional_encoding.get('pos_dim', 0)
            assert dim == proj_dim + pos_dim, 'proj_dim + pos_dim must equal dim'
            self.to_patch_embedding = nn.Linear(patch_dim, proj_dim)
        else:
            self.to_patch_embedding = nn.Linear(patch_dim, dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Positional encoding will be set by the model builder
        self.pos_encoder = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def set_positional_encoder(self, pos_encoder):
        """Set the positional encoder after initialization"""
        self.pos_encoder = pos_encoder
        
        # If using RoPE, set it for all attention layers
        if hasattr(pos_encoder, 'apply_rope_to_attention'):
            for block in self.transformer_blocks:
                block.attn.set_rope_encoder(pos_encoder)
        
    def forward(self, img):
        # Get batch size and reshape image into patches
        b, c, h, w = img.shape
        
        # Split image into patches
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(b, -1, c * self.patch_size * self.patch_size)
        
        # Project patches to embedding dimension
        x = self.to_patch_embedding(patches)  # (b, num_patches, proj_dim or dim)
        
        # Apply positional encoding based on mode
        if self.pos_encoder is not None:
            if self.positional_encoding and self.positional_encoding.get('mode') == 'concat':
                # Concatenate mode: get positional encodings and concatenate
                pos_enc = self.pos_encoder.get_patch_encodings(b)
                x = torch.cat([x, pos_enc], dim=2)  # (b, num_patches, dim)
            else:
                # Add mode: add positional encodings to patch embeddings
                x = self.pos_encoder(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (b, 1+num_patches, dim)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get the class token representation
        x = x[:, 0]
        
        # MLP head
        return self.mlp_head(x)

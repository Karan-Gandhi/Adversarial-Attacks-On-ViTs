import torch
import torch.nn as nn
import pywt
import numpy as np
import math

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, pos_dim, h, w):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.h = h
        self.w = w
        
        # Initialize learnable positional encoding
        self.pos_enc = nn.Parameter(torch.zeros(1, h * w + 1, pos_dim))

    def forward(self, x):
        # Add learnable positional encoding to input
        return x + self.pos_enc[:, 1:, :]
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import Module, Conv2d, Linear, MaxPool2d, Sequential, ReLU, Embedding
import math 
from ape_local.nn.encoder import Encoder
from ape_local.nn.position import Grid, SinusoidalGrid, Sequential
from ape_local.nn.attention import multihead_atn_fn


class AlgebraicCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(AlgebraicCCT, self).__init__()
        self.patch_embed = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = Grid(num_axes=2, num_heads=num_heads, dim=dim//(num_heads*2))
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        h, w = patch_values.shape[-2], patch_values.shape[-2]
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        x_pos = torch.arange(0, h, device=patch_values.device).unsqueeze(-1).expand(h, w).flatten()[None]
        y_pos = torch.arange(0, w, device=patch_values.device).unsqueeze(0).expand(h, w).flatten()[None]
        self.positional_encoder.precompute(max(h, w))
        x_maps, y_maps = self.positional_encoder.forward(x_pos, y_pos)
        coords = torch.cat((x_pos, y_pos), dim=0)
        dists = ((coords[:, None] - coords[:, :, None]) ** 2).sum(0).sqrt().floor()[None, :, :, None, None]
        atn_fn = self.positional_encoder.adjust_attention((x_maps, y_maps), (x_maps, y_maps), (0.98 ** dists, True))
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)
    
class SinCos2DPositionalEncodingAppend(nn.Module):
    def __init__(self, pos_dim):
        super(SinCos2DPositionalEncodingAppend, self).__init__()
        self.pos_dim = pos_dim

    def forward(self, x):
        B, N, D = x.size()
        h = w = int(N ** 0.5)
        assert h * w == N, f"Cannot reshape sequence of length {N} into square ({h}x{w})"

        y_pos = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w).reshape(h, w)
        x_pos = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1).reshape(h, w)

        dim_h = self.pos_dim // 2
        dim_w = self.pos_dim - dim_h

        div_term_h = torch.exp(torch.arange(0, dim_h, 2, device=x.device).float() * -(math.log(10000.0) / dim_h))
        div_term_w = torch.exp(torch.arange(0, dim_w, 2, device=x.device).float() * -(math.log(10000.0) / dim_w))

        pos_enc = torch.zeros(h, w, self.pos_dim, device=x.device)

        for i in range(0, dim_h, 2):
            pos_enc[:, :, i] = torch.sin(y_pos.float() * div_term_h[i//2])
            if i + 1 < dim_h:
                pos_enc[:, :, i+1] = torch.cos(y_pos.float() * div_term_h[i//2])

        for i in range(0, dim_w, 2):
            j = i + dim_h
            if j < self.pos_dim:
                pos_enc[:, :, j] = torch.sin(x_pos.float() * div_term_w[i//2])
                if j + 1 < self.pos_dim:
                    pos_enc[:, :, j+1] = torch.cos(x_pos.float() * div_term_w[i//2])

        pos_enc = pos_enc.reshape(1, N, self.pos_dim).expand(B, -1, -1)
        return torch.cat((x, pos_enc), dim=2)

class ourCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(ourCCT, self).__init__()
        self.patch_embed = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=(dim*7)//8,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = SinCos2DPositionalEncodingAppend(pos_dim=dim//8)
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        
        patch_values = self.positional_encoder(patch_values)
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=multihead_atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)
class SinusoidalCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(SinusoidalCCT, self).__init__()
        self.patch_embed = nn.Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = SinusoidalGrid(dim=dim)
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        positions = torch.arange(0, patch_values.shape[1], device=patch_values.device)[None]
        pos_emb = self.positional_encoder.forward(positions)
        patch_values = patch_values + pos_emb
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=multihead_atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)


class AbsoluteCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int,
            num_embeddings: int):
        super(AbsoluteCCT, self).__init__()
        self.patch_embed = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = Embedding(embedding_dim=dim, num_embeddings=num_embeddings)
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        positions = torch.arange(0, patch_values.shape[1], device=patch_values.device)[None]
        pos_emb = self.positional_encoder.forward(positions)
        patch_values = patch_values + pos_emb
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=multihead_atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)


class AlgebraicSeqCCT(Module):
    def __init__(
            self,
            dim: int,
            num_layers: int,
            num_heads: int,
            in_channels: int,
            kernel_size: tuple[int, int],
            num_classes: int,
            mlp_ratio: int):
        super(AlgebraicSeqCCT, self).__init__()
        self.patch_embed = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=dim,
                kernel_size=kernel_size,
                stride=(1, 1),
                bias=False
            ),
            ReLU(),
            MaxPool2d(kernel_size=kernel_size))
        self.encoder = Encoder(
            num_heads=num_heads,
            num_layers=num_layers,
            dim=dim,
            dropout_rate=0.1,
            weight_dropout=0.1,
            activation='GELU',
            drop_path=True,
            mlp_ratio=mlp_ratio)
        self.positional_encoder = Sequential(dim=dim//num_heads, num_heads=num_heads)
        self.pooler = Linear(dim, 1)
        self.fc = Linear(dim, num_classes)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_values = self.patch_embed(pixel_values)
        patch_values = patch_values.permute(0, 2, 3, 1).flatten(1, 2)
        positions = torch.arange(0, patch_values.shape[1], device=patch_values.device)[None]
        distances = (positions[:, :, None] - positions[:, None]).unsqueeze(-1).unsqueeze(-1)
        mediator = (0.98 ** distances.abs())
        self.positional_encoder.precompute(positions.shape[1])
        maps = self.positional_encoder.forward(positions[:1, :patch_values.shape[1]])
        atn_fn = self.positional_encoder.adjust_attention(q_maps=maps, k_maps=maps, mediator=(mediator, True))
        patch_values = self.encoder.forward(
            encoder_input=patch_values,
            encoder_mask=None,  # type: ignore
            atn_fn=atn_fn)
        gate = self.pooler(patch_values).softmax(dim=1)
        aggr = (gate * patch_values).sum(dim=1)
        return self.fc(aggr)

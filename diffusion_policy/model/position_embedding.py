from typing import Union, Optional, Tuple
import math
import logging
import torch
import torch.nn as nn

class PositionEmbedding2D(nn.Module):
    """
    2D position embedding copied from detr directory in act
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        '''
        @param tensor: (B, H, W, D)
        @return pos: (1, H, W, num_pos_feats*2)
            1 for broadcasting
            num_pos_feats*2 == embedding_dim -> 2*d == D
        '''
        x = tensor
        height, width = x.shape[1], x.shape[2]

        not_mask = torch.ones((1, height, width), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # (1, H, W)
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # (1, H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # (d,)
        dim_t = torch.exp(
            2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats * math.log(self.temperature)
        ) # (d,) self.temperature ** (2 * i / d), i = 0, 0, 1, 1, 2, 2, ..., d//2

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4 # (1, H, W, d//2, 2)
        ).flatten(3) # (1, H, W, d)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4 # (1, H, W, d//2, 2)
        ).flatten(3) # (1, H, W, d)
        pos = torch.cat((pos_y, pos_x), dim=3)  # (1, H, W, d*2)
        return pos
import torch
import torch.nn as nn
from torch import Tensor

class MLP2D(nn.Module):
    """
    Naive MLP for 2D data conditioned on noise level.
    Apply positional encoding to the inputs.
    """
    def __init__(self, hidden_dim, num_layers):
        super(MLP2D, self).__init__()
        self.linpos = nn.Linear(2, 64)
        layers = [nn.Linear(2*64, hidden_dim), nn.ReLU()]
        for _ in range(0, num_layers-2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
        self.pe = PE(num_pos_feats=64)

    def forward(self, x, sigma):
        x = torch.cat([self.linpos(x), self.pe(sigma)], dim=1)
        return self.mlp(x)

class PE(nn.Module):
    """
    Positional encoding.
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        dim_t = torch.arange(num_pos_feats)
        self.register_buffer("dim_t", temperature ** (2 * (dim_t // 2) / num_pos_feats))

    def forward(self, x):
        pos_x = x[:, :, None] / self.dim_t
        pos = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(1)
        return pos

class FMnet(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim + 1, h),
                nn.ELU(),
                nn.Linear(h, h),
                nn.ELU(),
                nn.Linear(h, h),
                nn.ELU(),
                nn.Linear(h, dim))
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
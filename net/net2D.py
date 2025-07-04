import torch
import torch.nn as nn
from torch import Tensor
from TTF.basic import basicTTF
from .utils import PE, Swish

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


class HeavyT_MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.linpos = nn.Linear(2, 64)
        self.pe = PE(num_pos_feats=64)

        self.main = nn.Sequential(
            nn.Linear(2*64, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            basicTTF(dim=input_dim)
            )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([self.linpos(x), self.pe(t)], dim=1)
        # h = torch.cat([x, t], dim=1)

        output = self.main(h)

        return output.reshape(*sz)
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)

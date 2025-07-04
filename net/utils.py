import torch
from torch import Tensor

class PE(torch.nn.Module):
    """
    Positional encoding.
    """
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        dim_t = torch.arange(num_pos_feats)
        self.register_buffer("dim_t",
                             temperature ** (2 * (dim_t // 2) / num_pos_feats))

    def forward(self, x):
        pos_x = x[:, :, None] / self.dim_t
        pos = torch.stack([pos_x[:, :, 0::2].sin(),
                           pos_x[:, :, 1::2].cos()],
                          dim=3).flatten(1)
        return pos

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x
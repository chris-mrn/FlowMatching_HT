import torch
import torch.nn as nn
from torch import Tensor
from TTF.basic import basicTTF

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
        self.register_buffer("dim_t",
                             temperature ** (2 * (dim_t // 2) / num_pos_feats))

    def forward(self, x):
        pos_x = x[:, :, None] / self.dim_t
        pos = torch.stack([pos_x[:, :, 0::2].sin(),
                           pos_x[:, :, 1::2].cos()],
                          dim=3).flatten(1)
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


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


class HeavyT_MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            basicTTF(dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            basicTTF(dim=hidden_dim),
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


class MLP_TailParam(nn.Module):
    def __init__(self, time_dim: int = 1, hidden_dim: int = 128,output_dim: int =8):
        super().__init__()

        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.output_dim= output_dim
        self.change_input=nn.Linear(output_dim//4,output_dim//4)

        self.main = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim//2),
            Swish(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//4, hidden_dim//8),
            Swish(),
            nn.Linear(hidden_dim//8, output_dim),
            )


    def forward( self,t: Tensor) -> Tensor:
        # sz = x.size()
        # x = x.reshape(-1, self.input_dim)
        # t = t.reshape(-1, self.time_dim).float()

        # t = t.reshape(-1, 1).expand(x.shape[0], 1)
        # h = torch.cat([x, t], dim=1)
        # print("YES",t.shape)
        output = self.main(t.float())

        return output#.reshape(*sz)



class MLP2(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim//2),
            # Swish(),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//4, input_dim),
            )


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)
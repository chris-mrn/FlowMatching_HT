# Activation class
import torch
from torch import nn, Tensor

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x

# Model class
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



import torch
import torch.nn as nn
import torch.nn.functional as F
class BigTimeConditionalNet(nn.Module):
    def __init__(self, input_dim=20, time_dim=128, hidden_dim=512):
        super(BigTimeConditionalNet, self).__init__()

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Fully connected network conditioned on time embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        """
        x: Tensor of shape (B, 20)
        t: Tensor of shape (B,) - time
        """
        t = t.view(-1, 1)                     # (B, 1)
        t_emb = self.time_mlp(t)              # (B, time_dim)

        x_cat = torch.cat([x, t_emb], dim=-1) # (B, 20 + time_dim)

        out = self.net(x_cat)                 # (B, 20)
        return out

class TimeToVecNet(nn.Module):
    def __init__(self, time_dim=128, hidden_dim=512,output_dim=80 ):
        super(TimeToVecNet, self).__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            # nn.BatchNorm1d(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            # nn.BatchNorm1d(time_dim),
            nn.SiLU(),
        )

        # Main network to map time embedding to output vector
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t):
        """
        t: Tensor of shape (B,) - time
        returns: Tensor of shape (B, 80)
        """
        B = t.size(0)
        t = t.view(B, 1)            # (B, 1)
        t_emb = self.time_mlp(t)    # (B, time_dim)
        out = self.net(t_emb)       # (B, 80)
        return out

class FullConnectedScoreModel(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super(FullConnectedScoreModel, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(data_dim+1, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.hidden_layers.append(nn.Sequential(layer, batch_norm))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, data_dim)  # Assuming output is a single value

    def forward(self, x, t):
        x_conc_t = torch.concat([x,t.unsqueeze(1)],axis=1)
        x = F.relu(self.input_batch_norm(self.input_layer(x_conc_t)))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.output_layer(x)



class FullConnectedScoreModel_time(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super(FullConnectedScoreModel_time, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(1, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.hidden_layers.append(nn.Sequential(layer, batch_norm))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, data_dim*4)  # Assuming output is a single value

    def forward(self, t):
        x_conc_t =t
        x = F.relu(self.input_batch_norm(self.input_layer(x_conc_t)))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.output_layer(x)


class MLP_diffusion(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128,output_dims:int=128):
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
            nn.Linear(hidden_dim, output_dims),
            )


    def forward(self, x: Tensor) -> Tensor:



        output = self.main(x)

        return output

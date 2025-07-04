import torch
import numpy as np

class basicTTF(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.lambd_plus = torch.nn.Parameter(torch.randn((dim)))
        self.lambd_neg = torch.nn.Parameter(torch.randn((dim)))
        self.mu = torch.nn.Parameter(torch.randn((dim)))
        self.sigma = torch.nn.Parameter(torch.randn((dim)))

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def normalize(self, x):
        return (1+torch.tanh(x))/3 + 1e-1

    def forward(self, z):

        sigma = 1e-3 + self.softplus(self.sigma)
        lambd_plus = self.normalize(self.lambd_plus)
        lambd_neg = self.normalize(self.lambd_neg)

        sign = torch.sign(z)
        lambd_s = torch.where(z > 0, lambd_plus, lambd_neg)
        g = torch.erfc(torch.abs(z) / np.sqrt(2)) + 1e-6 # handle zero power negative in next line
        x = (torch.pow(g, - lambd_s) - 1) / lambd_s
        x =  sign * x * sigma + self.mu

        return x
"""Implementations of permutation-like transforms."""
import torch
import numpy as np
import nflows.utils.typechecks as check
from nflows.transforms.permutations import Permutation
from nflows.transforms.linear import Linear
from nflows.transforms.lu import LULinear
from torch.nn import functional as F
from torch.nn import init
from torch import nn

class TailRandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features_light, features_heavy, dim=1):
        if not check.is_positive_int(features_light):
            raise ValueError("Number of light-tailed features must be a positive integer.")
        if not check.is_positive_int(int(features_heavy)):
            print(int(features_heavy))
            raise ValueError("Number of heavy-tailed features must be a positive integer.")
        # features_light and features_heavy are the number of light and heavy-tailed marginals, respectively
        self.permutation = torch.cat([torch.randperm(features_light), torch.randperm(features_heavy) + features_light])
        super().__init__(self.permutation, dim)


class TailLU(Linear):
    """Creates 3 matrices W1,W3, which are parameterized by the LU-Decomposition
    and an arbitrary matrix W2.
       They define a new matrix
          [ W1   0 ]
       W =[ W2   W3] where W1 is a lxl matrix (l = number of light-tailed components),
       W2 is lxh, and W3 is hxh (h = number of heavy-tailed components).
       """
    def __init__(self, features, num_heavy, using_cache=False, device="cuda"):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        if not check.is_positive_int(num_heavy):
            raise ValueError("Number of heavy-tailed features must be a positive integer.")
        if features < num_heavy:
            raise ValueError("Number of features must be a larger than the number of heavy-tailed features.")
        super().__init__(features, using_cache)
        self.num_light = features - num_heavy
        self.num_heavy = num_heavy
        self.W1 = LULinear(self.num_light).to(device)
        self.W2 = nn.Linear(self.num_light, self.num_heavy)
        self.W2.weight.data.fill_(0.0001) # initialize W2 essentially as zero-matrix
        self.W3 = LULinear(self.num_heavy).to(device)
        init.zeros_(self.bias)


    def forward_no_cache(self, inputs, context=None):
        light_forward = (self.W1.weight() @ inputs[:, :self.num_light, None]).squeeze()
        heavy_forward = (torch.cat([self.W2.weight, self.W3.weight()], dim=1) @ inputs[:, :, None]).squeeze()
        if len(heavy_forward.shape)==1: # Only one heavy tailed marginal
            heavy_forward = heavy_forward[:, None]
        if len(light_forward.shape)==1: # Only one light tailed marginal
            light_forward = light_forward[:, None]
        outputs = torch.cat([light_forward, heavy_forward], dim=1) + self.bias

        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet


    def inverse_no_cache(self, inputs, context=None):
        """
        The inverse has also a block triangular form:
                  [ W_1^{-1}           ,    0     ]
        W^{-1} =  [-W_3^{-1}W_2W_1^{-1}, W_3^{-1} ]
        """
        inputs -= self.bias
        W1_inv = self.W1.weight_inverse()
        W3_inv = self.W3.weight_inverse()
        W_prod = -W3_inv @ self.W2.weight @ W1_inv

        upper_outputs = (W1_inv @ inputs[:, :self.num_light, None]).squeeze()
        lower_outputs = (torch.cat([W_prod, W3_inv], dim=1) @ inputs[:, :, None]).squeeze()

        if len(upper_outputs.shape)==1: # Only one light tailed marginal
            upper_outputs = upper_outputs[:, None]
        if len(lower_outputs.shape)==1: # Only one heavy tailed marginal
            lower_outputs = lower_outputs[:, None]

        outputs = torch.cat([upper_outputs, lower_outputs], dim=1)

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def logabsdet(self):
        return self.W1.logabsdet() + self.W3.logabsdet()
        
        
class LULinear(Linear): 
    """A linear transform where we parameterize the LU decomposition of the weights."""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        n_triangular_entries = ((features - 1) * features) // 2

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))

        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = F.linear(inputs, upper)
        outputs = F.linear(outputs, lower, self.bias)
        logabsdet = self.logabsdet() * inputs.new_ones(outputs.shape[0])
        return outputs, logabsdet

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(
            outputs.t(), lower, upper=False, unitriangular=True
        )
        outputs, _ = torch.triangular_solve(
            outputs, upper, upper=True, unitriangular=False
        )
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features).to("cuda")
        lower_inverse, _ = torch.triangular_solve(
            identity, lower, upper=False, unitriangular=True
        )
        weight_inverse, _ = torch.triangular_solve(
            lower_inverse, upper, upper=True, unitriangular=False
        )
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))

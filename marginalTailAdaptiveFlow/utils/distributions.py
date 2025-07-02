from nflows.distributions.base import Distribution
import torch
from nflows.utils import torchutils
from torch.distributions.studentT import StudentT
from torch.distributions.normal import Normal
from torch import nn
import numpy as np

if torch.cuda.is_available():
    torch.device("cuda")
    device = "cuda"
else:
    torch.device("cpu")
    device = "cpu"
print(device)

class tDist(Distribution):
    """A multivariate t-Distribution with zero mean, unit scale, and learnable degree of freedom."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._d = self._shape[0]
        self.df = nn.Parameter(torch.tensor(50, dtype=torch.float64)) # initialize the df at 50.

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        return torch.sum(StudentT(self.df).log_prob(inputs), dim=1)


    def _sample(self, num_samples, context):
        if context is None:
            dim_wise_sample = []
            for dim in range(self._d):
                samp = StudentT(self.df).rsample([num_samples]).type(torch.float)
                dim_wise_sample.append(samp)
            return torch.transpose(torch.stack(dim_wise_sample, 1).view(self._d, -1), 0, 1)
        else:
            context_size = context.shape[0]
            dim_wise_sample = []
            for dim in range(self._d):
                samp, = StudentT(self.df).rsample([context_size * num_samples])
                dim_wise_sample.append(samp)
            samples = torch.stack(dim_wise_sample, 1)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])


class norm_tDist(Distribution):
    """A multivariate Distribution composed of marginal Gaussian and t Distributions with zero mean, unit scale, and learnable degree of freedom.
        Important: tail_indexes must be sorted such that the first d indeces are 0, and the following are non-zero."""

    def __init__(self, shape, tail_indexes):
        for j in range(len(tail_indexes) - 1):
            if (tail_indexes[j]>0 and tail_indexes[j+1]==0):
                raise ValueError('First d tail indeces needs to be 0, the following must be larger than zero.')
        super().__init__()
        self._shape = torch.Size(shape)
        self._d = self._shape[0]
        self._tail_indexes = tail_indexes
        try:
            self.num_light = tail_indexes[np.where(tail_indexes==0)].size
        except:
            self.num_light = 0
            print("No light-tailed components.")
        self.num_heavy = self._d - self.num_light

        self.register_buffer("_log_z_normcomp",
                             torch.tensor(0.5 * self.num_light * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)
        
        tail_indexes = np.array(tail_indexes)

        self.df = nn.Parameter(torch.tensor(tail_indexes[np.nonzero(tail_indexes)], dtype=torch.float, device=device))

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        # normal-components:
        inputs_norm = inputs[:, :self.num_light]
        neg_energy = -0.5 * \
            torchutils.sum_except_batch(inputs_norm ** 2, num_batch_dims=1)
        log_prob_normal = neg_energy - self._log_z_normcomp

        # t-components:
        batch_size = inputs.shape[0]
        if self.num_light==0:
            inputs_t = inputs.reshape([batch_size, self.num_heavy, 1])
        else:
            inputs_t = inputs[:, self.num_light:].reshape([batch_size, self.num_heavy, 1])
        log_prob_compwise = torch.lgamma((self.df + 1) / 2) - 1 / 2 * torch.log(self.df * torch.tensor(np.pi)) - torch.lgamma(
            self.df / 2) - (self.df + 1) / 2 * torch.log(1 + torch.pow(torch.squeeze(inputs_t), 2) / self.df)
        try:
            log_prob_t = torch.sum(log_prob_compwise, axis=1)
        except:
            # only 1 heavy-tailed component
            log_prob_t = log_prob_compwise
        return torch.add(log_prob_normal, log_prob_t).to(device)


    def _sample(self, num_samples, context):
        counter_heavytail = 0
        if context is None:
            dim_wise_sample = []
            for dim in range(self._d):
                if self._tail_indexes[dim] == 0:
                    samp = Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device)).rsample([num_samples]).type(torch.float32).to(device)
                else:
                    samp = StudentT(torch.tensor([self.df[counter_heavytail]])).rsample([num_samples]).type(torch.float32).to(device)
                    counter_heavytail += 1
                dim_wise_sample.append(samp)
            return torch.stack(dim_wise_sample, 1).view(-1, self._d)
        else:
            context_size = context.shape[0]
            dim_wise_sample = []
            for dim in range(self._d):
                if self._tail_indexes == 0:
                    samp = torch.tensor(Normal(torch.tensor([0.0]), torch.tensor([1.0])).rsample([num_samples]), dtype=torch.float)
                else:
                    samp = torch.tensor(StudentT(self.df[counter_heavytail].data.to(device)).rsample([num_samples]), dtype=torch.float)
                    counter_heavytail += 1
                dim_wise_sample.append(samp)
            samples = torch.stack(dim_wise_sample, 1)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])


''' Test cases:
# testing: t distribution
base_dist = tDist([2])
import matplotlib.pyplot as plt

sample = base_dist.sample(1000).detach().numpy()



sample = sample[:10, :]
print("My log_prob:")
print(base_dist.log_prob(sample))
print("True log_prob:")
print(StudentT(10).log_prob(torch.tensor(sample[:, 0])) + StudentT(10).log_prob(torch.tensor(sample[:, 1])))

for parameter in base_dist.parameters():
    print(parameter)

# testing: norm-t distribution
base_dist = norm_tDist([2], np.array([0, 4])).to("cuda")

sample = base_dist.sample(1000)

sample = sample[:10, :]
print("My log_prob:")
print(base_dist.log_prob(sample))
print("True log_prob:")
print(Normal(torch.tensor([0.0]).to("cuda"), torch.tensor([1.0]).to("cuda")).log_prob(torch.tensor(sample[:, 0])) + StudentT(4).log_prob(torch.tensor(sample[:, 1]).to("cuda")))

for parameter in base_dist.parameters():
    print(parameter)
'''

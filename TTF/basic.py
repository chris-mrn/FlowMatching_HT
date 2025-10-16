import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class basicTTF(torch.nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        scale = 1e-2
        self.lambd_plus = torch.nn.Parameter(scale * torch.randn(dim))
        self.lambd_neg  = torch.nn.Parameter(scale * torch.randn(dim))
        self.mu         = torch.nn.Parameter(scale * torch.randn(dim))
        self.sigma      = torch.nn.Parameter(scale * torch.randn(dim))

        # Parameter tracking
        self.param_history = {
            'lambd_plus': [],
            'lambd_neg': [],
            'mu': [],
            'sigma': [],
            'steps': []
        }
        self.step_count = 0

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def normalize(self, x):
        return (1+torch.tanh(x))/4 + 1e-1

    def forward(self, z):

        sigma = 1e-3 + self.softplus(self.sigma)
        lambd_plus = self.normalize(self.lambd_plus)
        lambd_neg = self.normalize(self.lambd_neg)

        sign = torch.sign(z)
        lambd_s = torch.where(z > 0, lambd_plus, lambd_neg)
        g = torch.erfc(torch.abs(z) / np.sqrt(2)) + 1e-6 # handle zero power negative in next line
        x = (torch.pow(g, - lambd_s) - 1) / lambd_s
        x =  sign * x * sigma + self.mu

        return  x

    def log_parameters(self):
        """Log current parameter values for tracking evolution"""
        self.param_history['lambd_plus'].append(self.lambd_plus.detach().cpu().clone())
        self.param_history['lambd_neg'].append(self.lambd_neg.detach().cpu().clone())
        self.param_history['mu'].append(self.mu.detach().cpu().clone())
        self.param_history['sigma'].append(self.sigma.detach().cpu().clone())
        self.param_history['steps'].append(self.step_count)
        self.step_count += 1

    def plot_parameter_evolution(self, save_path: str = 'outputs/ttf_evolution.png',
                               figsize: tuple = (15, 10)):
        """Plot the evolution of TTF parameters over training steps"""
        if not self.param_history['steps']:
            print("No parameter history recorded. Call log_parameters() during training.")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        steps = np.array(self.param_history['steps'])
        param_names = ['lambd_plus', 'lambd_neg', 'mu', 'sigma']
        colors = ['red', 'blue', 'green', 'orange']

        for idx, (param_name, color) in enumerate(zip(param_names, colors)):
            ax = axes[idx]

            # Stack all parameter values across steps
            param_values = torch.stack(self.param_history[param_name])  # [n_steps, dim]

            # Plot each dimension
            for dim in range(param_values.shape[1]):
                ax.plot(steps, param_values[:, dim].numpy(),
                       label=f'dim_{dim}', alpha=0.8, linewidth=2)

            ax.set_title(f'{param_name.replace("_", " ").title()} Evolution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Parameter Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TTF parameter evolution plot saved to: {save_path}")
        plt.show()

    def get_parameter_statistics(self) -> Dict:
        """Get statistics about parameter evolution"""
        if not self.param_history['steps']:
            return {}

        stats = {}
        param_names = ['lambd_plus', 'lambd_neg', 'mu', 'sigma']

        for param_name in param_names:
            param_values = torch.stack(self.param_history[param_name])  # [n_steps, dim]

            stats[param_name] = {
                'initial': param_values[0].numpy(),
                'final': param_values[-1].numpy(),
                'mean': param_values.mean(dim=0).numpy(),
                'std': param_values.std(dim=0).numpy(),
                'min': param_values.min(dim=0)[0].numpy(),
                'max': param_values.max(dim=0)[0].numpy(),
                'change': (param_values[-1] - param_values[0]).numpy()
            }

        return stats
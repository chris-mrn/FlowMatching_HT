import torch
import torch.nn as nn
import numpy as np


class FlowMatchingX0HT:
    def __init__(self, net, TTF, L=10, device='cpu'):
        self.net = net.to(device)
        self.TTF = TTF
        self.L = L
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def train(self, optim, X1_loader, X0_loader, n_epochs=10):
        print("Training flow matching X0...")

        for epoch in range(n_epochs):
            for x1, x0 in zip(X1_loader, X0_loader):
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)

                x0 = self.TTF(x0)

                t = torch.rand(len(x1), 1)
                t = t.to(self.device)
                x_t = (1 - t) * x0 + t * x1

                dx_t = x1 - x0

                optim.zero_grad()

                loss = self.loss_fn(self.flow(x_t, t), dx_t)
                loss.backward()

                optim.step()

            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.6f}")

    def flow(self, x_t, t):
        # t should a of shape (batch_size, 1)
        # t should be extended to match the input shape of x_t

        return self.net(x_t, t)

    def step(self, x_t, t_start, t_end):
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + (t_end - t_start) * self.flow(x_t + self.flow(x_t, t_start) * (t_end - t_start) / 2,
        t_start + (t_end - t_start) / 2)

    def sample_from(self, X0, n_steps=10):
        time_steps = torch.linspace(0, 1.0, n_steps + 1, device=self.device)
        x = X0.to(self.device)
        x = self.TTF(x)
        hist = torch.zeros(n_steps + 1, *X0.shape, device=self.device)
        hist[0] = x

        for i in range(n_steps):
            x = self.step(x, time_steps[i], time_steps[i + 1])
            hist[i + 1] = x

        return x, hist

    def coupling(self):
        # Implement your custom coupling
        pass

import torch


class GaussFlowMatching_OT:
    def __init__(self, neural_net, L=10, device='cpu'):
        self.net = neural_net.to(device)
        self.L = L
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def train(self, optimizer, X1_loader, X0_loader, n_epochs=10):
        print("Training flow matching...")

        for epoch in range(n_epochs):
            for x1, x0 in zip(X1_loader, X0_loader):
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)

                batch_size = x0.size(0)
                t = torch.rand(batch_size, 1, device=self.device)

                t = t.view(-1, 1, 1, 1)

                x_t = (1 - t) * x0 + t * x1

                dx_t = x1 - x0

                optimizer.zero_grad()
                loss = self.loss_fn(self.flow(x_t, t), dx_t)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.6f}")

    def flow(self, x_t, t):
        # t should a of shape (batch_size, 1)
        # t should be extended to match the input shape of x_t

        return self.net(x_t, t)

    def step(self, x_t, t_start, t_end):
        t_start = t_start.view(-1, 1).to(self.device)
        t_end = t_end.view(-1, 1).to(self.device)
        t_start = t_start.view(-1, 1, 1, 1)
        t_end = t_end.view(-1, 1, 1, 1)
        mid_t = t_start + (t_end - t_start) / 2

        midpoint = x_t + self.flow(x_t, t_start) * (t_end - t_start) / 2
        return x_t + (t_end - t_start) * self.flow(midpoint, mid_t)

    def sample_from(self, X0, n_steps=10):
        time_steps = torch.linspace(0, 1.0, n_steps + 1, device=self.device)
        x = X0.to(self.device)
        hist = torch.zeros(n_steps + 1, *X0.shape, device=self.device)
        hist[0] = x

        for i in range(n_steps):
            x = self.step(x, time_steps[i], time_steps[i + 1])
            hist[i + 1] = x

        return x, hist

    def coupling(self):
        # You can implement a custom coupling strategy here if needed
        pass

import torch
import numpy as np

class NCSN:
    def __init__(self, model, L=10, sigma_low=0.01, sigma_high=1, device=None):
        self.model = model
        self.L = L
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Generate a sequence of noise levels
        sigma = [sigma_high]
        for _ in range(L - 1):
            sigma.append(sigma[-1] * (sigma_low / sigma_high) ** (1 / (L - 1)))
        self.sigma = torch.tensor(sigma, device=self.device)

    def sample_from(self, X0, T=100, eps=2e-5):
        x_step = X0.to(self.device)
        x_hist = torch.zeros(self.L + 1, *x_step.shape, device=self.device)
        x_hist[0] = x_step

        with torch.no_grad():
            for i in range(self.L):
                alpha_i = eps * self.sigma[i] ** 2 / self.sigma[-1] ** 2
                for _ in range(T):
                    noise_level = self.sigma[i].expand(x_step.shape[0], 1)
                    # adapt to x shape
                    noise_level = noise_level.view(-1, 1, 1, 1)
                    x_step = x_step + alpha_i / 2 * self.model(x_step, noise_level) / noise_level \
                             + torch.sqrt(alpha_i) * torch.randn_like(x_step)
                x_hist[i + 1] = x_step

        return x_step, x_hist

    def train(self, optimizer, epochs, dataloader, print_interval=1):
        print("Training NCSN...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                batch_size = x.size(0)
                sigma_level_idx = torch.randint(0, self.L, (batch_size,), device=self.device)
                sigma_level = self.sigma[sigma_level_idx].unsqueeze(1)

                noise = torch.randn_like(x)
                # adapt this to the size of x
                sigma_level = sigma_level.unsqueeze(1).unsqueeze(2)

                x_perturbed = x + sigma_level * noise

                optimizer.zero_grad()
                noise_predicted = -self.model(x_perturbed, sigma_level)
                loss = ((noise - noise_predicted) ** 2).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            if epoch % print_interval == 0:
                avg_loss = total_loss / total_batches
                print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}")



class CNCSN:
    def __init__(self, model, L=10, sigma_low=0.01, sigma_high=1, device=None):
        self.model = model
        self.L = L
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Generate a sequence of noise levels
        sigma = [sigma_high]
        for _ in range(L - 1):
            sigma.append(sigma[-1] * (sigma_low / sigma_high) ** (1 / (L - 1)))
        self.sigma = torch.tensor(sigma, device=self.device)

    def sample_from(self, X0, T=100, eps=2e-5):
        x_step = X0.to(self.device)
        x_hist = torch.zeros(self.L + 1, *x_step.shape, device=self.device)
        x_hist[0] = x_step

        with torch.no_grad():
            for i in range(self.L):
                noise_level = self.sigma[i].expand(x_step.shape[0], 1)
                # adapt to x shape
                noise_level = noise_level.view(-1, 1, 1, 1)
                z = torch.randn_like(x_step)
                x_step = x_step + noise_level * z
                x_step = self.model(x_step, noise_level)

                x_hist[i + 1] = x_step

        return x_step, x_hist

    def train(self, optimizer, epochs, dataloader, print_interval=1):
        print("Training CNCSN...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)

                batch_size = x.size(0)
                sigma_level_idx = torch.randint(1, self.L, (batch_size,), device=self.device)
                sigma_level1 = self.sigma[sigma_level_idx-1].view(-1,1,1,1)
                sigma_level2 = self.sigma[sigma_level_idx].view(-1,1,1,1)

                noise = torch.randn_like(x)

                x1 = x + sigma_level1 * noise
                x2 = x + sigma_level2 * noise

                optimizer.zero_grad()

                predict1 = self.model(x1, sigma_level1)
                predict2 = self.model(x2, sigma_level2)

                loss = ((predict1 - predict2) ** 2).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

            if epoch % print_interval == 0:
                avg_loss = total_loss / total_batches
                print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.6f}")
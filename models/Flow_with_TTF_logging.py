import torch
from .utils import NativeScalerWithGradNormCount as NativeScaler


class GaussFlowMatching_OT_TTF:
    """Enhanced Flow Matching with TTF parameter logging"""

    def __init__(self, neural_net, L=10, device='cpu'):
        self.net = neural_net
        self.device = device
        self.L = L
        self.loss_fn = torch.nn.MSELoss()

        # Find TTF layer in the network
        self.ttf_layer = self._find_ttf_layer()

    def _find_ttf_layer(self):
        """Find the TTF layer in the network"""
        from TTF.basic import basicTTF

        def find_ttf_recursive(module):
            for child in module.children():
                if isinstance(child, basicTTF):
                    return child
                result = find_ttf_recursive(child)
                if result is not None:
                    return result
            return None

        return find_ttf_recursive(self.net)

    def train(self, optimizer, X1_loader, X0_loader, n_epochs=10, log_interval=100):
        """Training with TTF parameter logging"""
        print("Training flow matching with TTF parameter logging...")
        loss_scaler = NativeScaler()

        step_count = 0

        for epoch in range(n_epochs):
            epoch_loss = 0
            batch_count = 0

            for x1, x0 in zip(X1_loader, X0_loader):
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)

                t = torch.rand(len(x1), 1)
                t = t.to(self.device)
                x_t = (1 - t) * x0 + t * x1

                dx_t = x1 - x0

                optimizer.zero_grad()
                loss = self.loss_fn(self.flow(x_t, t), dx_t)

                loss_scaler(
                    loss,
                    optimizer,
                    parameters=self.net.parameters(),
                    update_grad=True,
                )

                # Log TTF parameters at specified intervals
                if self.ttf_layer is not None and step_count % log_interval == 0:
                    self.ttf_layer.log_parameters()

                epoch_loss += loss.item()
                batch_count += 1
                step_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")

        # Log final parameters
        if self.ttf_layer is not None:
            self.ttf_layer.log_parameters()
            print(f"TTF parameter logging completed. Total steps logged: {len(self.ttf_layer.param_history['steps'])}")

    def flow(self, x_t, t):
        # t should be of shape (batch_size, 1)
        return self.net(x_t, t)

    def step(self, x_t, t_start, t_end):
        return x_t + (t_end - t_start) * self.flow(x_t + self.flow(x_t, t_start) * (t_end - t_start) / 2,
                                                   t_start + (t_end - t_start) / 2)

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
        # Implement your custom coupling
        pass

    def plot_ttf_evolution(self, save_path='outputs/ttf_evolution.png'):
        """Plot TTF parameter evolution"""
        if self.ttf_layer is not None:
            self.ttf_layer.plot_parameter_evolution(save_path)
        else:
            print("No TTF layer found in the network.")

    def get_ttf_statistics(self):
        """Get TTF parameter statistics"""
        if self.ttf_layer is not None:
            return self.ttf_layer.get_parameter_statistics()
        else:
            print("No TTF layer found in the network.")
            return {}
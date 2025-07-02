import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
import numpy as np
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A script to train generative models."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "mps", "cpu"],
        help="Device to use",
    )
    return parser.parse_args()


def show_images(tensor, title=None, nrow=5, save_path=None):
    """
    Plot and optionally save a batch of images: tensor of shape (B, C, H, W)

    Args:
        tensor: torch.Tensor
        title: str, optional
        nrow: int, number of images per row in the grid
        save_path: str or Path, full path where to save the image (e.g. "results/sample.png")
    """
    # If tensor is on GPU, bring it to CPU
    tensor = tensor.detach().cpu()

    # Convert to grid of images
    grid = vutils.make_grid(tensor, nrow=nrow, normalize=True)

    # Convert to numpy
    np_img = grid.permute(1, 2, 0).numpy()

    # Create figure
    plt.figure(figsize=(nrow * 2, 2.5))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(np_img)

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved image to: {save_path}")

    plt.close()  # Don't display inline if only saving


def plot_model_samples(sample_list, model_names, ground_truth, figsize=(20, 5), save_path=None):
    """
    Plots samples from different models and the ground truth side by side,
    with shared axis limits for better comparison. Can optionally save the plot.

    Args:
        sample_list (list of torch.Tensor): List of tensors, each of shape (n_samples, 2).
        model_names (list of str): Names of the corresponding models.
        ground_truth (torch.Tensor or np.ndarray): Ground truth data, shape (n_samples, 2).
        figsize (tuple): Size of the figure.
        save_path (str or Path, optional): If provided, saves the plot to this path.
    """
    num_models = len(sample_list)
    fig, axs = plt.subplots(1, num_models + 1, figsize=figsize)

    # Convert all tensors to numpy
    all_samples = sample_list + [ground_truth]
    all_data = [s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s for s in all_samples]
    concatenated = np.concatenate(all_data, axis=0)

    # Compute axis limits
    x_min, x_max = concatenated[:, 0].min(), concatenated[:, 0].max()
    y_min, y_max = concatenated[:, 1].min(), concatenated[:, 1].max()

    # Plot model samples
    for i, (samples, name) in enumerate(zip(all_data[:-1], model_names)):
        axs[i].scatter(samples[:, 0], samples[:, 1], s=1)
        axs[i].set_title(f'{name} Samples')
        axs[i].set_xlim(x_min, x_max)
        axs[i].set_ylim(y_min, y_max)

    # Plot ground truth
    gt = all_data[-1]
    axs[-1].scatter(gt[:, 0], gt[:, 1], s=1)
    axs[-1].set_title('Ground Truth Samples')
    axs[-1].set_xlim(x_min, x_max)
    axs[-1].set_ylim(y_min, y_max)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_particle_trajectories(histories, model_names, X1, figsize=(20, 5), step=1, max_particles=25):
    """
    Plots particle trajectories over time for multiple models, using a subset of particles.

    Args:
        histories (list of list of torch.Tensor): Each element is a list of tensors (timesteps), each of shape (n_particles, 2).
        model_names (list of str): Names of the models.
        X1 (torch.Tensor or np.ndarray): Ground truth samples to display.
        figsize (tuple): Figure size.
        step (int): Plot every `step`-th time step to reduce clutter.
        max_particles (int): Maximum number of particles to plot per model.

    Returns:
        None
    """
    num_models = len(histories)
    fig, axs = plt.subplots(1, num_models + 1, figsize=figsize)

    for i, (hist, name) in enumerate(zip(histories, model_names)):
        hist = [h.detach().cpu() for h in hist]
        n_particles = hist[0].shape[0]

        # Sample a subset of particles
        indices = np.random.choice(n_particles, size=min(max_particles, n_particles), replace=False)

        for j in indices:
            traj = [hist[t][j] for t in range(0, len(hist), step)]
            traj = torch.stack(traj)
            axs[i].plot(traj[:, 0], traj[:, 1], lw=0.5)

        axs[i].set_title(f'{name} Trajectories')
        axs[i].scatter(hist[0][indices, 0], hist[0][indices, 1], s=2, c='green', label='Start')
        axs[i].scatter(hist[-1][indices, 0], hist[-1][indices, 1], s=2, c='red', label='End')
        axs[i].legend()
    axs[-1].scatter(X1[:, 0], X1[:, 1], s=1)
    axs[-1].set_title('Ground Truth')

    plt.tight_layout()
    plt.show()
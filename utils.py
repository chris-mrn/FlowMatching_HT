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


def plot_model_samples(sample_list, model_names, ground_truth, figsize=(20, 5), save_path='outputs/samples.png',
                      show_metrics=True, metrics_path='outputs/metrics.txt'):
    """
    Plots samples from different models and the ground truth side by side,
    with shared axis limits for better comparison. Can optionally save the plot.
    Optionally computes and displays heavy-tail specific evaluation metrics.

    Args:
        sample_list (list of torch.Tensor): List of tensors, each of shape (n_samples, 2).
        model_names (list of str): Names of the corresponding models.
        ground_truth (torch.Tensor or np.ndarray): Ground truth data, shape (n_samples, 2).
        figsize (tuple): Size of the figure.
        save_path (str or Path, optional): If provided, saves the plot to this path.
        show_metrics (bool): Whether to compute and display evaluation metrics.
        metrics_path (str or Path, optional): Path to save metrics summary.
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

    # Compute and display metrics if requested
    if show_metrics:
        try:
            from evaluation import comprehensive_evaluation, print_evaluation_summary

            # Convert ground truth to tensor if needed
            if isinstance(ground_truth, np.ndarray):
                gt_tensor = torch.from_numpy(ground_truth).float()
            else:
                gt_tensor = ground_truth.float()

            # Ensure sample tensors are float
            sample_tensors = [s.float() if isinstance(s, torch.Tensor) else torch.from_numpy(s).float()
                            for s in sample_list]

            # Compute metrics
            evaluation_results = comprehensive_evaluation(gt_tensor, sample_tensors, model_names)

            # Print summary
            print_evaluation_summary(evaluation_results, model_names)

            # Save metrics to file if path provided
            if metrics_path:
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, 'w') as f:
                    f.write("HEAVY-TAIL GENERATIVE MODEL EVALUATION METRICS\n")
                    f.write("=" * 60 + "\n\n")
                    for name in model_names:
                        f.write(f"{name.upper()}:\n")
                        f.write("-" * 30 + "\n")
                        for metric, value in evaluation_results[name].items():
                            if isinstance(value, float):
                                f.write(f"  {metric:<30}: {value:.6f}\n")
                            else:
                                f.write(f"  {metric:<30}: {value}\n")
                        f.write("\n")
                print(f"Saved metrics to: {metrics_path}")

        except ImportError:
            print("Warning: Could not import evaluation module. Metrics will not be displayed.")
        except Exception as e:
            print(f"Warning: Error computing metrics: {e}")

    # Plot model samples with metrics in titles if available
    for i, (samples, name) in enumerate(zip(all_data[:-1], model_names)):
        axs[i].scatter(samples[:, 0], samples[:, 1], s=1)

        # Add basic statistics to title if metrics are enabled
        title = f'{name} Samples'
        if show_metrics:
            try:
                # Compute basic tail statistics for display
                norms = np.linalg.norm(samples, axis=1)
                q99 = np.quantile(norms, 0.99)
                from evaluation import tail_index_hill
                tail_idx = tail_index_hill(torch.from_numpy(norms))
                title += f'\nQ99: {q99:.2f}, TI: {tail_idx:.3f}'
            except:
                pass

        axs[i].set_title(title)
        axs[i].set_xlim(-90, 90)
        axs[i].set_ylim(-90, 90)

    # Plot ground truth
    gt = all_data[-1]
    axs[-1].scatter(gt[:, 0], gt[:, 1], s=1)

    # Add ground truth statistics to title
    title = 'Ground Truth Samples'
    if show_metrics:
        try:
            gt_norms = np.linalg.norm(gt, axis=1)
            gt_q99 = np.quantile(gt_norms, 0.99)
            from evaluation import tail_index_hill
            gt_tail_idx = tail_index_hill(torch.from_numpy(gt_norms))
            title += f'\nQ99: {gt_q99:.2f}, TI: {gt_tail_idx:.3f}'
        except:
            pass

    axs[-1].set_title(title)
    axs[-1].set_xlim(-90, 90)
    axs[-1].set_ylim(-90, 90)

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
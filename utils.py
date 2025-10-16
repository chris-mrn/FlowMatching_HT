import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
import numpy as np
import yaml
from types import SimpleNamespace

def load_config(config_name):
    """Load configuration from YAML config file"""
    config_path = f"configs/{config_name}.yaml"

    if not os.path.exists(config_path):
        available_configs = [f.replace('.yaml', '') for f in os.listdir('configs') if f.endswith('.yaml')]
        raise FileNotFoundError(f"Config '{config_name}' not found. Available configs: {available_configs}")

    # Load the YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Flatten nested dictionaries for easier access
    flattened = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened[subkey] = subvalue
        else:
            flattened[key] = value

    # Convert to namespace for attribute access
    return SimpleNamespace(**flattened)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Flow Matching models for heavy-tailed distributions using config files."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="heavy_t_mlp",
        help="Name of the config file to use (without .yaml extension). Available: heavy_t_mlp, standard_mlp, fm_net, fm_x0_ht",
    )

    # Optional overrides
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Override device from config",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate from config",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="Override train/test split ratio from config (default: 0.8)",
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

    # Compute metrics silently if requested (for title enhancement)
    evaluation_results = None
    if show_metrics:
        try:
            from evaluation import comprehensive_evaluation

            # Convert ground truth to tensor if needed
            if isinstance(ground_truth, np.ndarray):
                gt_tensor = torch.from_numpy(ground_truth).float()
            else:
                gt_tensor = ground_truth.float()

            # Ensure sample tensors are float
            sample_tensors = [s.float() if isinstance(s, torch.Tensor) else torch.from_numpy(s).float()
                            for s in sample_list]

            # Compute metrics (but don't print them here - leave that to main evaluation)
            evaluation_results = comprehensive_evaluation(gt_tensor, sample_tensors, model_names)

            # Only save metrics to file if path provided (no console printing)
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

        except ImportError:
            pass  # Silently skip if evaluation module not available
        except Exception as e:
            pass  # Silently skip on error

    # Plot model samples with metrics in titles if available
    for i, (samples, name) in enumerate(zip(all_data[:-1], model_names)):
        axs[i].scatter(samples[:, 0], samples[:, 1], s=1)

        # Add basic statistics to title if metrics are enabled
        title = f'{name} Samples'
        if show_metrics and evaluation_results:
            try:
                # Use computed metrics for enhanced titles
                if name in evaluation_results:
                    w_dist = evaluation_results[name].get('wasserstein_distance', 0)
                    tail_diff = evaluation_results[name].get('tail_index_diff', 0)
                    title += f'\nW: {w_dist:.3f}, TI-diff: {tail_diff:.3f}'
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


def create_network(config):
    """Create network based on config"""
    from net.net2D import HeavyT_MLP, MLP, FMnet

    if config.network_type == "HeavyT_MLP":
        return HeavyT_MLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim)
    elif config.network_type == "MLP":
        return MLP(input_dim=config.input_dim, hidden_dim=config.hidden_dim)
    elif config.network_type == "FMnet":
        return FMnet(dim=config.input_dim, h=config.hidden_dim)
    else:
        raise ValueError(f"Unknown network type: {config.network_type}")


def create_model(network, config, device):
    """Create flow model based on config"""
    from models.Flow import GaussFlowMatching_OT
    from models.Flow_with_TTF_logging import GaussFlowMatching_OT_TTF
    from models.Flow_X0HT import FlowMatchingX0HT
    from TTF.basic import basicTTF

    if config.flow_type == "GaussFlowMatching_OT":
        return GaussFlowMatching_OT(network, device=device)
    elif config.flow_type == "GaussFlowMatching_OT_TTF":
        return GaussFlowMatching_OT_TTF(network, device=device)
    elif config.flow_type == "FlowMatchingX0HT":
        # For X0HT, we need both network and TTF
        ttf = basicTTF(dim=config.input_dim).to(device)
        return FlowMatchingX0HT(network, ttf, config.input_dim, device), ttf
    else:
        raise ValueError(f"Unknown flow type: {config.flow_type}")


def setup_config_and_device(args):
    """Setup configuration and device"""
    config = load_config(args.config)

    # Apply command-line overrides
    if args.device is not None:
        config.device = args.device
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.train_ratio is not None:
        config.train_ratio = args.train_ratio

    # Print configuration
    print("="*60)
    print("FLOW MATCHING FOR HEAVY-TAIL DISTRIBUTIONS")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Model: {config.model_name}")
    print(f"Network: {config.network_type}")
    print(f"Flow: {config.flow_type}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Train ratio: {getattr(config, 'train_ratio', 0.8):.1%}")
    print(f"TTF logging: {config.use_ttf_logging}")
    print("="*60)

    # Setup device
    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = 'cpu'

    return config, device


def prepare_data(config):
    """Load and split data into train/test sets"""
    # Load data
    data = torch.tensor(np.load(config.data_file))
    indices = torch.randperm(data.size(0))
    total_data = data[indices][:config.num_samples]

    # Split data into train and test sets
    train_ratio = getattr(config, 'train_ratio', 0.8)
    n_total = total_data.size(0)
    n_train = int(n_total * train_ratio)

    X1_train = total_data[:n_train]
    X1_test = total_data[n_train:]

    X0_train = torch.randn_like(X1_train)
    X0_test = torch.randn_like(X1_test)

    print(f"Data split: {n_train} train samples, {n_total - n_train} test samples")

    # Create dataloaders (only training data used for training)
    dataloader1 = torch.utils.data.DataLoader(X1_train, batch_size=config.batch_size, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0_train, batch_size=config.batch_size, shuffle=True)

    return X1_train, X1_test, X0_train, X0_test, dataloader1, dataloader0


def setup_model_and_optimizer(config, device):
    """Create network, model, and optimizer"""
    # Create network and model
    network = create_network(config).to(device)

    if config.flow_type == "FlowMatchingX0HT":
        model, ttf = create_model(network, config, device)
        # Special optimizer for X0HT (includes TTF parameters)
        optimizer = torch.optim.Adam(
            list(network.parameters()) + list(ttf.parameters()),
            lr=config.lr,
            weight_decay=getattr(config, 'weight_decay', 0.0)
        )
    else:
        model = create_model(network, config, device)
        optimizer = torch.optim.Adam(network.parameters(), lr=config.lr)

    return model, optimizer


def run_training(model, optimizer, dataloader1, dataloader0, config):
    """Run model training"""
    print(f"\nStarting training...")
    if config.use_ttf_logging and hasattr(model, 'train') and 'log_interval' in model.train.__code__.co_varnames:
        model.train(optimizer, dataloader1, dataloader0, n_epochs=config.epochs, log_interval=config.log_interval)
    else:
        model.train(optimizer, dataloader1, dataloader0, n_epochs=config.epochs)


def run_evaluation(model, X1_train, X1_test, X0_train, X0_test, config, device):
    """Run complete evaluation including TTF analysis, plotting, and metrics"""
    # Generate samples
    print(f"\nGenerating samples...")
    gen_samples_train, _ = model.sample_from(X0_train.to(device))
    gen_samples_test, _ = model.sample_from(X0_test.to(device))

    # TTF parameter analysis (if enabled)
    if config.use_ttf_logging and hasattr(model, 'plot_ttf_evolution'):
        print("\nAnalyzing TTF parameter evolution...")
        os.makedirs(config.output_dir, exist_ok=True)
        model.plot_ttf_evolution(os.path.join(config.output_dir, 'ttf_evolution.png'))

        # Print summary
        stats = model.get_ttf_statistics()
        if stats:
            print("TTF Parameter Summary:")
            for param_name, param_stats in stats.items():
                change = np.linalg.norm(param_stats['change'])
                print(f"  {param_name}: change magnitude = {change:.4f}")

    # Create plots and evaluation
    print(f"\nGenerating plots and evaluation...")
    os.makedirs(config.output_dir, exist_ok=True)

    # Sample plots for both train and test
    plot_model_samples(
        [gen_samples_train],
        [f"{config.model_name.upper()}_TRAIN"],
        X1_train,
        show_metrics=False,
        save_path=os.path.join(config.output_dir, 'samples_train.png')
    )

    plot_model_samples(
        [gen_samples_test],
        [f"{config.model_name.upper()}_TEST"],
        X1_test,
        show_metrics=False,
        save_path=os.path.join(config.output_dir, 'samples_test.png')
    )

    # Comprehensive evaluation on TRAINING data
    print("\n" + "="*60)
    print("EVALUATION ON TRAINING DATA")
    print("="*60)
    try:
        from evaluation import run_full_evaluation
        train_results = run_full_evaluation(
            real_data=X1_train,
            generated_samples=[gen_samples_train],
            model_names=[f"{config.model_name.upper()}_TRAIN"],
            output_dir=os.path.join(config.output_dir, 'evaluation_train'),
            create_plots=True
        )
    except Exception as e:
        print(f"Warning: Training evaluation failed: {e}")

    # Comprehensive evaluation on TEST data
    print("\n" + "="*60)
    print("EVALUATION ON TEST DATA")
    print("="*60)
    try:
        test_results = run_full_evaluation(
            real_data=X1_test,
            generated_samples=[gen_samples_test],
            model_names=[f"{config.model_name.upper()}_TEST"],
            output_dir=os.path.join(config.output_dir, 'evaluation_test'),
            create_plots=True
        )
        print(f"✅ Evaluation completed! Results saved to: {config.output_dir}")
    except Exception as e:
        print(f"Warning: Test evaluation failed: {e}")

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Outputs saved to: {config.output_dir}")
    print("="*60)
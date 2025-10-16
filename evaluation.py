"""
Evaluation metrics for heavy-tail generative modeling.

This module provides 5 key evaluation metrics specifically designed
for assessing the quality of generative models on heavy-tailed distributions:

1. wasserstein_distance: Overall distribution similarity
2. tail_index_diff: Direct comparison of tail heaviness (Hill estimator)
3. quantile_0.99_ratio: Extreme quantile behavior (99th percentile)
4. quantile_0.999_ratio: Ultra-extreme quantile behavior (99.9th percentile)
5. moment_4_ratio: Fourth moment captures tail behavior through higher-order statistics
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance
import warnings
from typing import List, Tuple, Dict, Optional


def tail_index_hill(data: torch.Tensor, k: Optional[int] = None) -> float:
    """
    Estimate the tail index using Hill's estimator.

    Args:
        data: Input data tensor of shape (n_samples, n_dims) or (n_samples,)
        k: Number of largest order statistics to use. If None, uses sqrt(n_samples)

    Returns:
        Estimated tail index (higher values indicate heavier tails)
    """
    if data.dim() > 1:
        # For multivariate data, use the norm
        data = torch.norm(data, dim=1)

    data = data.detach().cpu().numpy()
    data = np.abs(data)  # Take absolute values
    n = len(data)

    if k is None:
        k = max(10, int(np.sqrt(n)))

    # Sort in descending order
    sorted_data = np.sort(data)[::-1]

    # Hill estimator
    if k >= n:
        k = n - 1

    log_ratios = np.log(sorted_data[:k] / sorted_data[k])
    hill_estimate = np.mean(log_ratios)

    return hill_estimate


def tail_behavior_comparison(real_data: torch.Tensor, generated_data: torch.Tensor,
                           quantiles: List[float] = [0.95, 0.99, 0.999]) -> Dict[str, float]:
    """
    Compare tail behavior between real and generated data.

    Args:
        real_data: Real data tensor
        generated_data: Generated data tensor
        quantiles: List of quantiles to compare

    Returns:
        Dictionary containing tail comparison metrics
    """
    if real_data.dim() > 1:
        real_norms = torch.norm(real_data, dim=1)
        gen_norms = torch.norm(generated_data, dim=1)
    else:
        real_norms = torch.abs(real_data)
        gen_norms = torch.abs(generated_data)

    real_norms = real_norms.detach().cpu().numpy()
    gen_norms = gen_norms.detach().cpu().numpy()

    results = {}

    # Compare quantiles
    for q in quantiles:
        real_q = np.quantile(real_norms, q)
        gen_q = np.quantile(gen_norms, q)
        results[f'quantile_{q}_ratio'] = gen_q / real_q if real_q != 0 else float('inf')
        results[f'quantile_{q}_diff'] = abs(gen_q - real_q)

    # Compare tail indices
    real_tail_idx = tail_index_hill(torch.from_numpy(real_norms))
    gen_tail_idx = tail_index_hill(torch.from_numpy(gen_norms))
    results['tail_index_real'] = real_tail_idx
    results['tail_index_generated'] = gen_tail_idx
    results['tail_index_diff'] = abs(real_tail_idx - gen_tail_idx)

    return results


def wasserstein_distance_nd(real_data: torch.Tensor, generated_data: torch.Tensor) -> float:
    """
    Compute 1-Wasserstein distance for multidimensional data using projections.

    Args:
        real_data: Real data tensor of shape (n_samples, n_dims)
        generated_data: Generated data tensor of shape (n_samples, n_dims)

    Returns:
        Approximate Wasserstein distance
    """
    real_data = real_data.detach().cpu().numpy()
    generated_data = generated_data.detach().cpu().numpy()

    if real_data.shape[1] == 1:
        return wasserstein_distance(real_data.flatten(), generated_data.flatten())

    # For higher dimensions, use random projections
    n_projections = min(100, real_data.shape[1] * 10)
    distances = []

    for _ in range(n_projections):
        # Random unit vector
        direction = np.random.randn(real_data.shape[1])
        direction = direction / np.linalg.norm(direction)

        # Project data
        real_proj = real_data @ direction
        gen_proj = generated_data @ direction

        # Compute 1D Wasserstein distance
        distances.append(wasserstein_distance(real_proj, gen_proj))

    return np.mean(distances)


def moment_comparison(real_data: torch.Tensor, generated_data: torch.Tensor,
                     max_moment: int = 4) -> Dict[str, float]:
    """
    Compare statistical moments between real and generated data.

    Args:
        real_data: Real data tensor
        generated_data: Generated data tensor
        max_moment: Maximum moment order to compute

    Returns:
        Dictionary containing moment comparison metrics
    """
    real_data = real_data.detach().cpu().numpy()
    generated_data = generated_data.detach().cpu().numpy()

    if real_data.ndim > 1:
        real_data = np.linalg.norm(real_data, axis=1)
        generated_data = np.linalg.norm(generated_data, axis=1)

    results = {}

    for moment in range(1, max_moment + 1):
        real_moment = np.mean(real_data ** moment)
        gen_moment = np.mean(generated_data ** moment)

        results[f'moment_{moment}_real'] = real_moment
        results[f'moment_{moment}_generated'] = gen_moment
        results[f'moment_{moment}_ratio'] = gen_moment / real_moment if real_moment != 0 else float('inf')
        results[f'moment_{moment}_diff'] = abs(real_moment - gen_moment)

    return results


def comprehensive_evaluation(real_data: torch.Tensor, generated_samples: List[torch.Tensor],
                           model_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Perform focused evaluation using the 5 most relevant metrics for heavy-tail modeling.

    The 5 key metrics are:
    1. wasserstein_distance: Measures overall distribution similarity (lower = better)
    2. tail_index_diff: Compares tail heaviness using Hill estimator (lower = better)
    3. quantile_0.99_ratio: How well extreme values (99th percentile) are captured (closer to 1.0 = better)
    4. quantile_0.999_ratio: How well ultra-extreme values (99.9th percentile) are captured (closer to 1.0 = better)
    5. moment_4_ratio: Fourth moment comparison captures tail behavior (closer to 1.0 = better)

    Args:
        real_data: Real data tensor
        generated_samples: List of generated sample tensors from different models
        model_names: List of model names corresponding to generated_samples

    Returns:
        Dictionary containing the 5 key evaluation metrics for each model
    """
    results = {}

    for i, (samples, name) in enumerate(zip(generated_samples, model_names)):
        model_results = {}

        # 1. Wasserstein distance - Overall distribution similarity
        model_results['wasserstein_distance'] = wasserstein_distance_nd(real_data, samples)

        # 2. Tail behavior comparison - Get tail index difference and extreme quantiles
        tail_metrics = tail_behavior_comparison(real_data, samples, quantiles=[0.99, 0.999])
        model_results['tail_index_diff'] = tail_metrics['tail_index_diff']
        model_results['quantile_0.99_ratio'] = tail_metrics['quantile_0.99_ratio']
        model_results['quantile_0.999_ratio'] = tail_metrics['quantile_0.999_ratio']

        # 3. Fourth moment ratio - Captures tail behavior through higher-order statistics
        moment_metrics = moment_comparison(real_data, samples, max_moment=4)
        model_results['moment_4_ratio'] = moment_metrics['moment_4_ratio']

        results[name] = model_results

    return results


def print_evaluation_summary(evaluation_results: Dict[str, Dict[str, float]],
                           model_names: List[str]) -> None:
    """
    Print a formatted summary of evaluation results.

    Args:
        evaluation_results: Results from comprehensive_evaluation
        model_names: List of model names
    """
    print("=" * 80)
    print("HEAVY-TAIL GENERATIVE MODEL EVALUATION SUMMARY")
    print("=" * 80)

    # Key metrics to display - 5 most relevant for heavy-tail modeling
    key_metrics = [
        'wasserstein_distance',     # Overall distribution similarity
        'tail_index_diff',          # Direct tail heaviness comparison
        'quantile_0.99_ratio',      # Extreme quantile behavior
        'quantile_0.999_ratio',     # Ultra-extreme quantile behavior
        'moment_4_ratio'            # Fourth moment captures tail behavior
    ]

    print(f"{'Metric':<25}", end="")
    for name in model_names:
        print(f"{name:<15}", end="")
    print()
    print("-" * (25 + 15 * len(model_names)))

    for metric in key_metrics:
        print(f"{metric:<25}", end="")
        for name in model_names:
            if metric in evaluation_results[name]:
                value = evaluation_results[name][metric]
                if isinstance(value, float):
                    print(f"{value:<15.4f}", end="")
                else:
                    print(f"{str(value):<15}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()


def plot_tail_comparison(real_data: torch.Tensor, generated_samples: List[torch.Tensor],
                        model_names: List[str], figsize: Tuple[int, int] = (15, 10),
                        save_path: Optional[str] = None) -> None:
    """
    Create comprehensive plots comparing tail behavior.

    Args:
        real_data: Real data tensor
        generated_samples: List of generated sample tensors
        model_names: List of model names
        figsize: Figure size
        save_path: Optional path to save the plot
    """
    n_models = len(generated_samples)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Convert to numpy and compute norms
    real_norms = torch.norm(real_data, dim=1).detach().cpu().numpy()
    gen_norms = [torch.norm(samples, dim=1).detach().cpu().numpy() for samples in generated_samples]

    # 1. Distribution comparison (log scale)
    ax = axes[0]
    bins = np.logspace(np.log10(0.1), np.log10(np.max(real_norms)), 50)
    ax.hist(real_norms, bins=bins, alpha=0.7, density=True, label='Real', color='black')
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    for i, (norms, name) in enumerate(zip(gen_norms, model_names)):
        ax.hist(norms, bins=bins, alpha=0.7, density=True, label=name, color=colors[i])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('||x||')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-Q plots
    ax = axes[1]
    quantiles = np.linspace(0.01, 0.99, 100)
    real_quantiles = np.quantile(real_norms, quantiles)
    ax.plot(real_quantiles, real_quantiles, 'k--', label='Perfect match', alpha=0.5)
    for i, (norms, name) in enumerate(zip(gen_norms, model_names)):
        gen_quantiles = np.quantile(norms, quantiles)
        ax.plot(real_quantiles, gen_quantiles, label=name, color=colors[i])
    ax.set_xlabel('Real Data Quantiles')
    ax.set_ylabel('Generated Data Quantiles')
    ax.set_title('Q-Q Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Tail quantiles comparison
    ax = axes[2]
    tail_quantiles = [0.9, 0.95, 0.99, 0.995, 0.999]
    real_tail_vals = [np.quantile(real_norms, q) for q in tail_quantiles]

    x_pos = np.arange(len(tail_quantiles))
    width = 0.8 / (n_models + 1)

    ax.bar(x_pos - width * n_models / 2, real_tail_vals, width, label='Real', color='black', alpha=0.7)
    for i, (norms, name) in enumerate(zip(gen_norms, model_names)):
        gen_tail_vals = [np.quantile(norms, q) for q in tail_quantiles]
        ax.bar(x_pos - width * n_models / 2 + width * (i + 1), gen_tail_vals,
               width, label=name, color=colors[i], alpha=0.7)

    ax.set_xlabel('Quantile')
    ax.set_ylabel('Value')
    ax.set_title('Tail Quantiles Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{q:.3f}' for q in tail_quantiles])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 4. Complementary CDF (survival function)
    ax = axes[3]
    sorted_real = np.sort(real_norms)
    ccdf_real = 1 - np.arange(1, len(sorted_real) + 1) / len(sorted_real)
    ax.loglog(sorted_real, ccdf_real, 'k-', label='Real', linewidth=2)

    for i, (norms, name) in enumerate(zip(gen_norms, model_names)):
        sorted_gen = np.sort(norms)
        ccdf_gen = 1 - np.arange(1, len(sorted_gen) + 1) / len(sorted_gen)
        ax.loglog(sorted_gen, ccdf_gen, label=name, color=colors[i], linewidth=2)

    ax.set_xlabel('||x||')
    ax.set_ylabel('P(||X|| > x)')
    ax.set_title('Complementary CDF (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Wasserstein distances
    ax = axes[4]
    w_distances = []
    for norms in gen_norms:
        w_dist = wasserstein_distance(real_norms, norms)
        w_distances.append(w_dist)

    bars = ax.bar(model_names, w_distances, color=colors[:n_models], alpha=0.7)
    ax.set_ylabel('Wasserstein Distance')
    ax.set_title('Wasserstein Distance from Real Data')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, dist in zip(bars, w_distances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.3f}', ha='center', va='bottom')

    # 6. Tail indices comparison
    ax = axes[5]
    real_tail_idx = tail_index_hill(torch.from_numpy(real_norms))
    gen_tail_indices = [tail_index_hill(torch.from_numpy(norms)) for norms in gen_norms]

    all_indices = [real_tail_idx] + gen_tail_indices
    all_names = ['Real'] + model_names
    colors_with_real = ['black'] + list(colors[:n_models])

    bars = ax.bar(all_names, all_indices, color=colors_with_real, alpha=0.7)
    ax.set_ylabel('Tail Index (Hill Estimator)')
    ax.set_title('Tail Index Comparison')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, idx in zip(bars, all_indices):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{idx:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved tail comparison plot to: {save_path}")

    plt.show()


def run_full_evaluation(real_data: torch.Tensor, generated_samples: List[torch.Tensor],
                       model_names: List[str], output_dir: str = 'outputs/evaluation',
                       create_plots: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Run complete evaluation suite for heavy-tail generative models.

    This function performs comprehensive evaluation including:
    - All quantitative metrics (Wasserstein, KL divergence, tail indices, etc.)
    - Detailed comparison plots
    - Summary reports

    Args:
        real_data: Real/ground truth data tensor
        generated_samples: List of generated sample tensors from different models
        model_names: List of model names
        output_dir: Directory to save all outputs
        create_plots: Whether to create and save plots

    Returns:
        Dictionary containing all evaluation metrics for each model
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Compute all metrics
    evaluation_results = comprehensive_evaluation(real_data, generated_samples, model_names)

    # 2. Print summary to console
    print_evaluation_summary(evaluation_results, model_names)

    # 3. Save detailed metrics to file
    metrics_file = os.path.join(output_dir, 'detailed_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("COMPREHENSIVE HEAVY-TAIL GENERATIVE MODEL EVALUATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Real data shape: {real_data.shape}\n")
        f.write(f"Number of models evaluated: {len(model_names)}\n")
        f.write(f"Model names: {', '.join(model_names)}\n\n")

        # Summary table
        f.write("SUMMARY TABLE (5 Key Heavy-Tail Metrics):\n")
        f.write("-" * 50 + "\n")
        key_metrics = ['wasserstein_distance', 'tail_index_diff', 'quantile_0.99_ratio', 'quantile_0.999_ratio', 'moment_4_ratio']
        f.write(f"{'Metric':<25}")
        for name in model_names:
            f.write(f"{name:<15}")
        f.write("\n")
        f.write("-" * (25 + 15 * len(model_names)) + "\n")

        for metric in key_metrics:
            f.write(f"{metric:<25}")
            for name in model_names:
                if metric in evaluation_results[name]:
                    value = evaluation_results[name][metric]
                    f.write(f"{value:<15.4f}")
                else:
                    f.write(f"{'N/A':<15}")
            f.write("\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("DETAILED METRICS:\n")
        f.write("=" * 70 + "\n")

        for name in model_names:
            f.write(f"\n{name.upper()}:\n")
            f.write("-" * 40 + "\n")
            for metric, value in evaluation_results[name].items():
                if isinstance(value, float):
                    f.write(f"  {metric:<35}: {value:.8f}\n")
                else:
                    f.write(f"  {metric:<35}: {value}\n")

    # 4. Create and save plots
    if create_plots:
        # Tail comparison plot
        tail_plot_path = os.path.join(output_dir, 'tail_comparison.png')
        plot_tail_comparison(real_data, generated_samples, model_names,
                           figsize=(18, 12), save_path=tail_plot_path)

        # Enhanced sample plot with metrics
        from utils import plot_model_samples
        sample_plot_path = os.path.join(output_dir, 'samples_with_metrics.png')
        metrics_summary_path = os.path.join(output_dir, 'metrics_summary.txt')
        plot_model_samples(generated_samples, model_names, real_data,
                         figsize=(20, 6), save_path=sample_plot_path,
                         show_metrics=True, metrics_path=metrics_summary_path)

    # 5. Create ranking summary
    ranking_file = os.path.join(output_dir, 'model_ranking.txt')
    with open(ranking_file, 'w') as f:
        f.write("MODEL RANKING SUMMARY\n")
        f.write("=" * 40 + "\n\n")

        # Rank by key metrics (lower is better for most)
        metrics_for_ranking = ['wasserstein_distance', 'tail_index_diff', 'moment_4_ratio']

        for metric in metrics_for_ranking:
            f.write(f"Ranking by {metric} (lower is better):\n")
            f.write("-" * 35 + "\n")

            # Get values and sort
            metric_values = [(name, evaluation_results[name][metric])
                           for name in model_names if metric in evaluation_results[name]]
            metric_values.sort(key=lambda x: x[1])

            for i, (name, value) in enumerate(metric_values, 1):
                f.write(f"  {i}. {name:<20}: {value:.6f}\n")
            f.write("\n")

        # Best overall (composite score)
        f.write("COMPOSITE RANKING:\n")
        f.write("-" * 25 + "\n")
        f.write("(Average normalized rank across key metrics)\n\n")

        # Compute composite scores
        composite_scores = {}
        for name in model_names:
            ranks = []
            for metric in metrics_for_ranking:
                if metric in evaluation_results[name]:
                    values = [evaluation_results[other_name][metric]
                            for other_name in model_names if metric in evaluation_results[other_name]]
                    rank = sorted(values).index(evaluation_results[name][metric]) + 1
                    ranks.append(rank)
            composite_scores[name] = np.mean(ranks) if ranks else float('inf')

        sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1])
        for i, (name, score) in enumerate(sorted_composite, 1):
            f.write(f"  {i}. {name:<20}: {score:.2f}\n")

    return evaluation_results


if __name__ == "__main__":
    """
    Example usage of the evaluation module.
    """
    import torch
    import numpy as np

    # Example: Generate synthetic heavy-tail data and test samples
    print("Running evaluation module example...")

    # Create synthetic Student-t data (heavy tails)
    np.random.seed(42)
    torch.manual_seed(42)

    # Real data: Student-t with 2 degrees of freedom (heavy tails)
    n_samples = 10000
    real_data = torch.tensor(np.random.standard_t(df=2, size=(n_samples, 2))).float()

    # Generated samples:
    # Model 1: Normal distribution (light tails)
    gen_samples_1 = torch.randn(n_samples, 2)

    # Model 2: Student-t with 5 df (medium tails)
    gen_samples_2 = torch.tensor(np.random.standard_t(df=5, size=(n_samples, 2))).float()

    # Model 3: Student-t with 2 df (matching tails)
    gen_samples_3 = torch.tensor(np.random.standard_t(df=2, size=(n_samples, 2))).float()

    generated_samples = [gen_samples_1, gen_samples_2, gen_samples_3]
    model_names = ['Normal (Light Tails)', 'Student-t (df=5)', 'Student-t (df=2)']

    # Run full evaluation
    results = run_full_evaluation(real_data, generated_samples, model_names,
                                 output_dir='outputs/example_evaluation')

    print("\nExample evaluation completed!")

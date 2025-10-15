# Flow Matching for Heavy Tail Distributions

PyTorch implementation of Flow Matching models for heavy-tailed distributions, featuring Tail-to-Tail Flows (TTF) and specialized neural architectures.

## Models

- **Standard FM**: Basic flow matching with optimal transport
- **FM-HT**: Flow matching with heavy-tail specialized networks
- **FM-X0-HT**: Flow matching with TTF transformations on source distribution

## New: Comprehensive Evaluation Metrics

This implementation includes specialized evaluation metrics for heavy-tail generative modeling:
- **Tail Index Estimation** (Hill estimator)
- **Tail Behavior Comparison** (extreme quantiles)
- **Wasserstein Distance** for robust distribution comparison
- **KL Divergence** and Kolmogorov-Smirnov tests
- **Comprehensive visualizations** including Q-Q plots, complementary CDFs, and tail analysis

Metrics are automatically computed after training. See `EVALUATION_GUIDE.md` for details.

## Installation

```bash
pip install torch torchvision numpy matplotlib flow_matching scipy
git clone https://github.com/chris-mrn/FlowMatching_HT.git
cd FlowMatching_HT
```

## Usage

```bash
python main.py --device cpu  # or cuda for GPU
```

After training, comprehensive evaluation metrics will be automatically computed and saved to `outputs/evaluation/`.

## Structure

```
├── main.py                   # Training script with automatic evaluation
├── evaluation.py             # Heavy-tail evaluation metrics
├── models/                   # Flow matching implementations
│   ├── Flow.py              # Standard Gaussian Flow Matching
│   └── Flow_X0HT.py         # Flow Matching with TTF
├── net/                     # Neural architectures
├── TTF/                     # Tail-to-Tail Flow transformations
├── data/                    # Dataset handling (Student-t, Pareto, Funnel)
└── outputs/                 # Results and evaluation reports
    └── evaluation/          # Detailed metrics and plots
```

## Key Features

- **TTF (Tail-to-Tail Flow)**: Learnable transformations for heavy-tail modeling
- **Heavy-tail networks**: Specialized MLPs for heavy-tailed distributions
- **Multiple datasets**: Student-t (2D), Pareto (20D), Funnel (2D)
- **Comprehensive evaluation**: Automated heavy-tail specific metrics
- **Visual analysis**: Q-Q plots, tail comparison, complementary CDFs

## Parameters

- Learning rate: 1e-4
- Batch size: 2048
- Epochs: 1000

## Evaluation Outputs

After training, check:
- `outputs/evaluation/detailed_metrics.txt` - Complete numerical results
- `outputs/evaluation/tail_comparison.png` - Visual tail analysis
- `outputs/evaluation/model_ranking.txt` - Model performance ranking


```

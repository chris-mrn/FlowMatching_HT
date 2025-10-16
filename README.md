# Flow Matching for Heavy Tail Distributions

PyTorch implementation of Flow Matching models for heavy-tailed distributions, featuring Tail-to-Tail Flows (TTF) and specialized neural architectures.

## Models

- **Heavy-T MLP**: Heavy-tail MLP with TTF logging
- **Standard MLP**: Standard MLP without TTF
- **FM Net**: Flow Matching Network
- **FM X0-HT**: Flow Matching with X0 Heavy-Tail transformation

## Features

- **YAML Configuration System**: Easy model selection via config files
- **TTF Parameter Tracking**: Monitor tail transformation evolution
- **Heavy-tail Evaluation**: 5 specialized metrics (Wasserstein, tail index, quantile ratios)
- **Utility Functions**: Clean model creation with `create_network()` and `create_model()`

## Installation

```bash
pip install torch torchvision numpy matplotlib flow_matching scipy pyyaml
git clone https://github.com/chris-mrn/FlowMatching_HT.git
cd FlowMatching_HT
```

## Usage

```bash
# Use default config (heavy_t_mlp)
python main.py

# Use specific config
python main.py --config standard_mlp
python main.py --config fm_net
python main.py --config fm_x0_ht

# Override parameters
python main.py --config heavy_t_mlp --epochs 1000 --batch-size 4096

# Custom train/test split (default: 80% train, 20% test)
python main.py --config fm_net --train-ratio 0.9  # 90% train, 10% test
```

## Structure

```
├── main.py                   # Training script with config system
├── utils.py                  # Utility functions (create_network, create_model)
├── configs/                  # YAML configuration files
│   ├── heavy_t_mlp.yaml     # Heavy-tail MLP with TTF
│   ├── standard_mlp.yaml    # Standard MLP
│   ├── fm_net.yaml          # Flow Matching Network
│   └── fm_x0_ht.yaml        # X0 Heavy-tail transformation
├── evaluation.py             # Heavy-tail evaluation metrics (5 key metrics)
├── models/                   # Flow matching implementations
├── net/                      # Neural architectures
├── TTF/                      # Tail-to-Tail Flow transformations
└── outputs/                  # Results and evaluation reports
```

## Available Configs

- **heavy_t_mlp.yaml**: HeavyT_MLP + GaussFlowMatching_OT_TTF (with TTF logging)
- **standard_mlp.yaml**: MLP + GaussFlowMatching_OT
- **fm_net.yaml**: FMnet + GaussFlowMatching_OT
- **fm_x0_ht.yaml**: FMnet + FlowMatchingX0HT + basicTTF

## Evaluation

**Automatic Train/Test Split Evaluation:**
- Data automatically split into train (80%) and test (20%) sets
- Training performed only on training data
- Evaluation performed on both training and test sets separately
- Separate metrics, plots, and reports for train vs test performance

**5 Heavy-Tail Specific Metrics:**
- Wasserstein distance
- Tail index difference
- Quantile ratios (0.99, 0.999)
- 4th moment ratio

**Output Structure:**
```
outputs/
├── samples_train.png          # Training set samples
├── samples_test.png           # Test set samples
├── evaluation_train/          # Training evaluation
│   ├── detailed_metrics.txt
│   └── tail_comparison.png
└── evaluation_test/           # Test evaluation
    ├── detailed_metrics.txt
    └── tail_comparison.png
```
```

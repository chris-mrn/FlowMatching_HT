import torch
import numpy as np
import os
from utils import parse_arguments, load_config, plot_model_samples, create_network, create_model


def main():
    # Parse arguments and load config
    args = parse_arguments()
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

    # Load data
    data = torch.tensor(np.load(config.data_file))
    indices = torch.randperm(data.size(0))
    total_data = data[indices][:config.num_samples]

    # Split data into train and test sets
    train_ratio = getattr(config, 'train_ratio', 0.8)  # Default 80% train, 20% test
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

    # Training
    print(f"\nStarting training...")
    if config.use_ttf_logging and hasattr(model, 'train') and 'log_interval' in model.train.__code__.co_varnames:
        model.train(optimizer, dataloader1, dataloader0, n_epochs=config.epochs, log_interval=config.log_interval)
    else:
        model.train(optimizer, dataloader1, dataloader0, n_epochs=config.epochs)

    # Generate samples
    print(f"\nGenerating samples...")
    gen_samples_train, _ = model.sample_from(X0_train.to(device))
    gen_samples_test, _ = model.sample_from(X0_test.to(device))

    # TTF parameter analysis (if enabled)
    if config.use_ttf_logging and hasattr(model, 'plot_ttf_evolution'):
        print("\nAnalyzing TTF parameter evolution...")
        os.makedirs(config.output_dir, exist_ok=True)
        model.plot_ttf_evolution(os.path.join(config.output_dir, 'ttf_evolution.png'))
        model.save_ttf_history(os.path.join(config.output_dir, 'ttf_history.pt'))

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
        show_metrics=False,  # Metrics will be shown in full evaluation
        save_path=os.path.join(config.output_dir, 'samples_train.png')
    )

    plot_model_samples(
        [gen_samples_test],
        [f"{config.model_name.upper()}_TEST"],
        X1_test,
        show_metrics=False,  # Metrics will be shown in full evaluation
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


if __name__ == "__main__":
    main()
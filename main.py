from utils import (
    parse_arguments,
    setup_config_and_device,
    prepare_data,
    setup_model_and_optimizer,
    run_training,
    run_evaluation
)


def main():

    # Parse arguments and setup configuration
    args = parse_arguments()
    config, device = setup_config_and_device(args)

    # BLOCK 1: DATA PREPARATION
    X1_train, X1_test, X0_train, X0_test, dataloader1, dataloader0 = prepare_data(config)

    # BLOCK 2: MODEL SETUP
    model, optimizer = setup_model_and_optimizer(config, device)

    # BLOCK 3: TRAINING
    run_training(model, optimizer, dataloader1, dataloader0, config)

    # BLOCK 4: EVALUATION
    run_evaluation(model, X1_train, X1_test, X0_train, X0_test, config, device)


if __name__ == "__main__":
    main()
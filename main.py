import torch
import torch.nn as nn
import torchvision
from net.unet import Unet
from net.net2D import FMnet, MLP2D
from models.Flow import GaussFlowMatching_OT
from models.Score import NCSN
import torchvision.transforms as transforms
from utils import parse_arguments, show_images
import numpy as np
from utils import plot_model_samples
import matplotlib.pyplot as plt
import flow_matching
from models.Flow_X0HT import FlowMatchingX0HT
from TTF.basic import basicTTF
from net.net2D import HeavyT_MLP, MLP

#from models.utils.extreme_transforms import TTF



def main():
    # Parse arguments

    args = parse_arguments()
    data = torch.tensor(np.load("data/ST2.npy"))

    indices = torch.randperm(data.size(0))  # Get random indices
    X1 = data[indices][:100000]  # Apply the random permutation
    X0 = torch.randn_like(torch.Tensor(X1))

    batch_size = 2*4096

    # Creating dataloader
    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=batch_size, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=batch_size, shuffle=True)

    device = 'cuda'

    # Setting the parameters of the model
    dim = 2
    lr = 1e-4
    epochs = 1000

    """""""""
    net_fm = FMnet().to(device)
    model_FM = GaussFlowMatching_OT(net_fm, device=device)
    optimizer_fm = torch.optim.Adam(net_fm.parameters(), lr)

    model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=epochs)
    gen_FM_samples, hist_FM = model_FM.sample_from(X0.to(device))


    net = FMnet().to(device)
    ttf = basicTTF(dim=dim).to(device)

    optimizer = torch.optim.Adam(list(net.parameters()) + list(ttf.parameters()),
                                 lr=lr,
                                 weight_decay=1e-3)

    model_FMX0_HT = FlowMatchingX0HT(net, ttf, dim, device)
    model_FMX0_HT.train(optimizer, dataloader1, dataloader0, epochs)
    gen_samples_X0, hist = model_FMX0_HT.sample_from(X0.to(device))
    """""""""

    net_HT = HeavyT_MLP().to(device)
    #net = MLP().to(device)
    #net_HT = net
    model_FM_HT = GaussFlowMatching_OT(net_HT, device=device)
    optimizer = torch.optim.Adam(net_HT.parameters(), lr)

    model_FM_HT.train(optimizer, dataloader1 , dataloader0 , n_epochs=epochs)

    gen_samples_FM_HT, hist_FMHT = model_FM_HT.sample_from(X0.to(device))

    # Collect all generated samples and model names for evaluation
    generated_samples = [gen_samples_FM_HT]
    model_names = ['FM_HT']

    # If other models are trained (uncomment the sections above), add them:
    # generated_samples.extend([gen_FM_samples, gen_samples_X0])
    # model_names.extend(['FM_Standard', 'FM_X0_HT'])

    # Plots with basic metrics
    plot_model_samples(
        generated_samples,
        model_names,
        X1,
        show_metrics=True)

    # Comprehensive evaluation with detailed metrics
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE HEAVY-TAIL EVALUATION")
    print("="*60)

    try:
        from evaluation import run_full_evaluation

        # Run complete evaluation suite
        evaluation_results = run_full_evaluation(
            real_data=X1,
            generated_samples=generated_samples,
            model_names=model_names,
            output_dir='outputs/evaluation',
            create_plots=True
        )

        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("Check the 'outputs/evaluation' directory for detailed results.")
        print("="*60)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Continuing without detailed evaluation...")


if __name__ == "__main__":
    main()

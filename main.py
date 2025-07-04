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
from models.NewX0Heavy import FlowMatchingHTX0, TTF

#from models.utils.extreme_transforms import TTF



def main():
    # Parse arguments

    args = parse_arguments()
    data = torch.tensor(np.load("data/ST2.npy"))

    indices = torch.randperm(data.size(0))  # Get random indices
    X1 = data[indices][:100000]  # Apply the random permutation
    X0 = torch.randn_like(torch.Tensor(X1))

    # Creating dataloader
    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=2, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=2, shuffle=True)

    device = 'cuda'

    # Setting the parameters of the model
    dim = 2
    hidden_dim = 512
    lr = 1e-4
    epochs = 150


    net_fm = FMnet()
    model_FM = GaussFlowMatching_OT(net_fm, device=args.device)
    optimizer_fm = torch.optim.Adam(net_fm.parameters(), lr)

    model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=epochs)
    gen_FM_samples, hist_FM = model_FM.sample_from(X0.to(device))

    net = FMnet()
    ttf = TTF(dim=dim).to(device)

    optimizer = torch.optim.Adam(list(net.parameters()) + list(ttf.parameters()),
                                 lr=lr,
                                 weight_decay=1e-3)

    model_ht_fm = FlowMatchingHTX0(net, ttf, dim, device)
    model_ht_fm.train(optimizer, dataloader1, dataloader0, epochs)
    gen_samples_FMHT, hist = model_ht_fm.sample_from(X0.to(device))


    # Plots
    plot_model_samples(
        [gen_samples_FMHT, gen_FM_samples],
        ['FM_HT', 'FM'],
        X1)


if __name__ == "__main__":
    main()

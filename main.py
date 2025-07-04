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
from net.net2D import HeavyT_MLP

#from models.utils.extreme_transforms import TTF



def main():
    # Parse arguments

    args = parse_arguments()
    data = torch.tensor(np.load("data/ST2.npy"))

    indices = torch.randperm(data.size(0))  # Get random indices
    X1 = data[indices][:200000]  # Apply the random permutation
    X0 = torch.randn_like(torch.Tensor(X1))

    # Creating dataloader
    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=2048, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=2048, shuffle=True)

    device = 'cuda'

    # Setting the parameters of the model
    dim = 2
    lr = 1e-4
    epochs = 100

    """"""""""""""""
    net_fm = FMnet().to(device)
    model_FM = GaussFlowMatching_OT(net_fm, device=device)
    optimizer_fm = torch.optim.Adam(net_fm.parameters(), lr)

    model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=1)
    gen_FM_samples, hist_FM = model_FM.sample_from(X0.to(device))

    net = FMnet().to(device)
    ttf = basicTTF(dim=dim).to(device)

    optimizer = torch.optim.Adam(list(net.parameters()) + list(ttf.parameters()),
                                 lr=lr,
                                 weight_decay=1e-3)

    model_FMX0_HT = FlowMatchingX0HT(net, ttf, dim, device)
    model_FMX0_HT.train(optimizer, dataloader1, dataloader0, epochs)
    gen_samples_X0, hist = model_FMX0_HT.sample_from(X0.to(device))
    """""""""""

    net_HT = HeavyT_MLP().to(device)
    model_FM_HT = GaussFlowMatching_OT(net_HT, device=device)
    optimizer = torch.optim.Adam(net_HT.parameters(), lr)

    model_FM_HT.train(optimizer, dataloader1 , dataloader0 , n_epochs=epochs)

    gen_samples_FM_HT, hist_FMHT = model_FM_HT.sample_from(X0.to(device))


    # Plots
    plot_model_samples(
        [ gen_samples_FM_HT],
        [ 'FM_HT'],
        X1)


if __name__ == "__main__":
    main()

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



def main():
    # Parse arguments
    args = parse_arguments()

    X1 = torch.tensor(np.load("data/ST2.npy"))
    X0 = torch.rand_like(torch.Tensor(X1))
    # Creating dataloader
    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=512, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=512, shuffle=True)


    net_fm = FMnet()
    model_FM = GaussFlowMatching_OT(net_fm, device=args.device)
    optimizer_fm = torch.optim.Adam(net_fm.parameters(), 5e-4)

    model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=50)
    gen_FM_samples, hist_FM = model_FM.sample_from(X0[:4000])

    # Plots
    plot_model_samples(
        [gen_FM_samples],
        ['FM'],
        X1)


if __name__ == "__main__":
    main()

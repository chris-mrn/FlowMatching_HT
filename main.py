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

from models.HT_Flow import HT_FlowMatching_X0
from net.network import MLP, MLP_TailParam
from models.utils.extreme_transforms import TTF



def main():
    # Parse arguments

    args = parse_arguments()

    X1 = torch.tensor(np.load("data/ST2.npy"))
    X0 = torch.randn_like(torch.Tensor(X1))
    # Creating dataloader
    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=4096, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=4096, shuffle=True)


    #net_fm = FMnet()
    #model_FM = GaussFlowMatching_OT(net_fm, device=args.device)
    #optimizer_fm = torch.optim.Adam(net_fm.parameters(), 5e-4)

    #model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=120)
    #gen_FM_samples, hist_FM = model_FM.sample_from(X0[:4000])

    dim = 2
    hidden_dim = 512
    lr = 1e-4
    epochs = 100

    device = 'cpu'
    flow_net = MLP(input_dim=dim, time_dim=1, hidden_dim=hidden_dim).to(device)
    tail_net = MLP_TailParam(time_dim=1, hidden_dim=hidden_dim//2, output_dim=4*dim).to(device)

    optim_net = torch.optim.Adam(flow_net.parameters(), lr=lr, weight_decay=1e-3)
    optim_tail = torch.optim.Adam(tail_net.parameters(), lr=lr, weight_decay=1e-3)
    ttf = TTF(dimz=dim).to(device)

    model_ht_fm = HT_FlowMatching_X0(tail_net, flow_net, ttf, dim, device)
    model_ht_fm.train(optim_net, optim_tail, dataloader1, dataloader0, epochs)
    gen_samples_FMHT = model_ht_fm.generate(X0[0:10000])

    # Plots
    plot_model_samples(
        [gen_samples_FMHT],
        ['FM_HT'],
        X1)


if __name__ == "__main__":
    main()

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



def main():
    # Parse arguments
    args = parse_arguments()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # make sure the data have the format (Batch, Channel, Height, Width)

    # root_cifar = '/Users/christophermarouani/Desktop/cifar-10-batches-py'
    # cifar_data = torchvision.datasets.CIFAR10(root_cifar, download=True, transform=transform)

    root_mnist = '/Users/christophermarouani/Desktop/mnist'
    mnist_data = torchvision.datasets.MNIST(root_mnist, download=True, transform=transform)

    #X1 = mnist_data.data.unsqueeze(1)
    # transform X1 to normalize it and put it to float
    #X1 = X1 / 255.0
    #X1 = X1.float()[:1000]


    # ST1 dataset
    X1 = torch.tensor(np.load("data/ST1.npy"))

    print(X1.shape)

    X0 = torch.rand_like(torch.Tensor(X1))



    dataloader1 = torch.utils.data.DataLoader(X1, batch_size=64, shuffle=True)
    dataloader0 = torch.utils.data.DataLoader(X0, batch_size=64, shuffle=True)



    # net_score = Unet()
    h = 64
    net_score = MLP2D(hidden_dim=h, num_layers=4)
    model_score = NCSN(net_score, L=10, device=args.device)
    optimizer_score = torch.optim.Adam(net_score.parameters(), 1e-3)

    model_score.train(optimizer_score, epochs=150, dataloader=dataloader1, print_interval=10)
    gen_score_samples, hist_score = model_score.sample_from(X0[:10])


    net_fm = FMnet()
    model_FM = GaussFlowMatching_OT(net_fm, device=args.device)
    optimizer_fm = torch.optim.Adam(net_fm.parameters(), 1e-3)

    model_FM.train(optimizer_fm, dataloader1 , dataloader0 , n_epochs=150)
    gen_FM_samples, hist_FM = model_FM.sample_from(X0[:10])

    # Show and save FM samples
    show_images(gen_score_samples, title="Score Matching Samples", save_path="outputs/gen_NCSN_samples.png")
    show_images(gen_FM_samples, title="FM Samples", save_path="outputs/gen_FM_samples.png")

if __name__ == "__main__":
    main()

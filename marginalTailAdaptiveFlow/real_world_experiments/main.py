import os
import xarray as xr
import numpy as np
import wandb

import argparse
import torch

import seaborn as sns

sns.set_context('paper', font_scale=1.8)

import sys
sys.path.append('../utils')

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.weather_helper import to_unnormalized_dataset, plot_weather, to_stacked_array
from utils.flows import NSF


###################
# Parse Arguments #
###################
parser = argparse.ArgumentParser()
# architecture
parser.add_argument("--marginals",
                    type=str,
                    default="mTAF",
                    help="Choose the marginals of the base distribution:" \
                         "vanilla, TAF, mTAF, or mTAF(fix)."
                    )
parser.add_argument("--num_layers",
                    type=int,
                    default=10,
                    help="Number of flow layers.")
parser.add_argument("--num_blocks",
                    type=int,
                    default=2,
                    help="Number of hidden layers of the networks in each flow layer (backbone networks).")
parser.add_argument("--num_hidden",
                    type=int,
                    default=100,
                    help="Number of hidden nodes per layer in each backbone network.")
# optimization
parser.add_argument("--train_steps",
                    type=int,
                    default=20000,
                    help="Number of training steps.")
parser.add_argument("--lr_wd",
                    type=float,
                    default=0.0,
                    help="Weight decay learning rate.")
parser.add_argument("--lr_df",
                    type=float,
                    default=0.01,
                    help="Learning rate for the parameters corresponding to the degree of freedom.")

args = parser.parse_args()

# configure wandb
wandb.init(anonymous="allow", project="mTAF_ICML")
wandb.config.update(args)
wandb.config.update({"comment": "weather"})

if __name__=="__main__":
    # define the model
    model = NSF(args.marginals)
    model.load_data("weather", estimate_tails=False)
    model.config(linear_layer="LU",
                 num_layers=args.num_layers,
                 train_steps=args.train_steps,
                 num_hidden=args.num_hidden,
                 lr_df=args.lr_df,
                 lr_wd=args.lr_wd,
                 cosine_annealing=True,
                 tail_bound=2.5,
                 num_bins=3,
                 num_blocks=args.num_blocks,
                 lr=0.0001,
                 batch_norm_layer=True,
                 track_results=True)

    # train the model
    tst_loss = model.train(grad_clip=False)

    # generate new samples
    model.flow.eval()
    torch.cuda.empty_cache()
    num_samps = 100
    # due to memory issues: Compute 10 samples at once
    for j in range(int(num_samps/10)):
        print(f"Sampling from the model...{j*10}/{num_samps} samples generated.")
        if j==0:
            all_samps = model.flow.sample(10).detach().cpu().numpy()
            if args.marginals in ["mTAF", "gTAF"]:
                all_samps = all_samps[:, model.inv_perm]
        else:
            samps = model.flow.sample(10).detach().cpu().numpy()
            if args.marginals in ["mTAF", "gTAF"]:
                samps = samps[:, model.inv_perm]
            all_samps = np.vstack([all_samps, samps])

    samps_ds = xr.Dataset({"temperature_fl": xr.DataArray(all_samps[:, 275:], dims=["column", "level"]),
                       "pressure_hl": xr.DataArray(all_samps[:, 137:275], dims=["column", "halflevel"]),
                       "layer_cloud_optical_depth": xr.DataArray(all_samps[:, :137], dims=["column", "level"])
                })

    PATH_DATA = f"results/nsf/{args.marginals}/weather/{args.num_layers}layers/samples.csv"
    ds_synth, _ = to_stacked_array(samps_ds)
    ds_synth.to_pandas().to_csv(PATH_DATA)

    samps_ds = to_unnormalized_dataset(samps_ds, model.stats_info)

    # clip optical depth for visualization:
    samps_ds["layer_cloud_optical_depth"] = samps_ds["layer_cloud_optical_depth"].clip(min=0)

    # plot data
    plot_weather(samps_ds, args)

    # log data
    wandb.log({"test loss": tst_loss})

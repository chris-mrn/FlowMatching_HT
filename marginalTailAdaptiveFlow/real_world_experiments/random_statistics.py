import os
from pathlib import Path
import argparse
import torch
import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.weather_helper import load_ds_inputs, to_normalized_dataset, to_stacked_array
from utils.flows import NSF

sns.set_context('paper', font_scale=1.8)
alpha = 0.5# alpha used in plots

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
                    default=5,
                    help="Number of flow layers.")
parser.add_argument("--num_blocks",
                    type=int,
                    default=2,
                    help="Number of hidden layers of the networks in each flow layer (backbone networks).")
parser.add_argument("--num_hidden",
                    type=int,
                    default=100,
                    help="Number of hidden nodes per layer in each backbone network.")
# evaluation
parser.add_argument("--sample_from_model",
                    type=bool,
                    default=False,
                    help="Generate new samples from the model? Otherwise take locally saved samples.")
args = parser.parse_args()


if __name__=="__main__":
    # 1. load true data:
    PROJ_PATH = Path.cwd().parent
    ds_true_in = load_ds_inputs(PROJ_PATH)
    ds_true, stats_info = to_normalized_dataset(ds_true_in)

    ls_ds_synth = []
    for marginals in ["vanilla", "TAF", "gTAF", "mTAF"]:
        model = NSF(marginals)
        model.load_data("weather",
                        estimate_tails=True)
        model.config(linear_layer="LU",
                     num_layers=args.num_layers,
                     num_hidden=args.num_hidden,
                     tail_bound=2.5,
                     num_bins=3,
                     num_blocks=args.num_blocks,
                     batch_norm_layer=True)
        model.flow.load_state_dict(
            torch.load(f"trained_models/weather/nsf/{marginals}/{args.num_layers}layers/model_LU",
                       map_location="cuda"))
        model.flow.eval()

        # Sample from the model:
        PATH_DATA = f"results/nsf/{marginals}/weather/{args.num_layers}layers/samples.csv"
        if args.sample_from_model:
            num_samps = 200
            for j in range(int(num_samps / 10)):
                print(f"Sampling from the model...{j * 10}/{num_samps} samples generated.")
                if j == 0:
                    all_samps = model.flow.sample(10).detach().cpu().numpy()
                    print(args.marginals)
                    if args.marginals in ["mTAF", "mTAF(fix)"]:
                        all_samps = all_samps[:, model.inv_perm]
                else:
                    samps = model.flow.sample(10).detach().cpu().numpy()
                    if args.marginals in ["mTAF", "mTAF(fix)"]:
                        samps = samps[:, model.inv_perm]
                    all_samps = np.vstack([all_samps, samps])
                samps_ds = xr.Dataset({"temperature_fl": xr.DataArray(all_samps[:, 275:], dims=["column", "level"]),
                                       "pressure_hl": xr.DataArray(all_samps[:, 137:275],
                                                                   dims=["column", "halflevel"]),
                                       "layer_cloud_optical_depth": xr.DataArray(all_samps[:, :137],
                                                                                 dims=["column", "level"])
                                       })

                ds_synth, _ = to_stacked_array(samps_ds)
                ds_synth.to_pandas().to_csv(PATH_DATA)
        else:
            ds_synth = xr.DataArray(pd.read_csv(PATH_DATA, index_col=0))
            num_samps = len(ds_synth)

        ls_ds_synth.append(ds_synth)

    ds_true, _ = to_stacked_array(ds_true_in)

    # Randomly project to 1-D space
    np.random.seed(120)
    mean_true = []
    mean_vanilla = []
    mean_TAF = []
    mean_gTAF = []
    mean_mTAF = []
    std_true = []
    std_vanilla = []
    std_TAF = []
    std_gTAF = []
    std_mTAF = []
    q5_true = []
    q5_vanilla = []
    q5_TAF = []
    q5_gTAF = []
    q5_mTAF = []
    q1_true = []
    q1_vanilla = []
    q1_TAF = []
    q1_gTAF = []
    q1_mTAF = []
    q10_true = []
    q10_vanilla = []
    q10_TAF = []
    q10_gTAF = []
    q10_mTAF = []
    q95_true = []
    q95_vanilla = []
    q95_TAF = []
    q95_gTAF = []
    q95_mTAF = []
    q90_true = []
    q90_vanilla = []
    q90_TAF = []
    q90_gTAF = []
    q90_mTAF = []
    q99_true = []
    q99_vanilla = []
    q99_TAF = []
    q99_gTAF = []
    q99_mTAF = []

    ds_vanilla = ls_ds_synth[0]
    ds_TAF = ls_ds_synth[1]
    ds_gTAF = ls_ds_synth[2]
    ds_mTAF = ls_ds_synth[3]

    for j in range(100):
        # 1. Compute a random projection
        weights = np.random.rand(ds_true.shape[1], 1)

        # 2. Project your data onto this space
        proj_true = np.dot(ds_true[np.random.choice(range(len(ds_true)), num_samps)], weights)
        proj_vanilla = np.dot(ds_vanilla, weights)
        proj_TAF = np.dot(ds_TAF, weights)
        proj_gTAF = np.dot(ds_gTAF, weights)
        proj_mTAF = np.dot(ds_mTAF, weights)

        # 3. Compute Statistics
        # 3.1. Mean
        mean_true.append(np.mean(proj_true))
        mean_vanilla.append(np.mean(proj_vanilla))
        mean_TAF.append(np.mean(proj_TAF))
        mean_gTAF.append(np.mean(proj_gTAF))
        mean_mTAF.append(np.mean(proj_mTAF))

        # 3.2. Std
        std_true.append(np.std(proj_true))
        std_vanilla.append(np.std(proj_vanilla))
        std_TAF.append(np.std(proj_TAF))
        std_gTAF.append(np.std(proj_gTAF))
        std_mTAF.append(np.std(proj_mTAF))

        # 3.3 1%-Quantile
        q1_true.append(np.quantile(proj_true, 0.01))
        q1_vanilla.append(np.quantile(proj_vanilla, 0.01))
        q1_TAF.append(np.quantile(proj_TAF, 0.01))
        q1_gTAF.append(np.quantile(proj_gTAF, 0.01))
        q1_mTAF.append(np.quantile(proj_mTAF, 0.01))
        # 3.4 5%-Quantile
        q5_true.append(np.quantile(proj_true, 0.05))
        q5_vanilla.append(np.quantile(proj_vanilla, 0.05))
        q5_TAF.append(np.quantile(proj_TAF, 0.05))
        q5_gTAF.append(np.quantile(proj_gTAF, 0.05))
        q5_mTAF.append(np.quantile(proj_mTAF, 0.05))
        # 3.5 10%-Quantile
        q10_true.append(np.quantile(proj_true, 0.1))
        q10_vanilla.append(np.quantile(proj_vanilla, 0.1))
        q10_TAF.append(np.quantile(proj_TAF, 0.1))
        q10_gTAF.append(np.quantile(proj_gTAF, 0.1))
        q10_mTAF.append(np.quantile(proj_mTAF, 0.1))
        # 3.6 90%-Quantile
        q90_true.append(np.quantile(proj_true, 0.9))
        q90_vanilla.append(np.quantile(proj_vanilla, 0.9))
        q90_TAF.append(np.quantile(proj_TAF, 0.9))
        q90_gTAF.append(np.quantile(proj_gTAF, 0.9))
        q90_mTAF.append(np.quantile(proj_mTAF, 0.9))
        # 3.7 95%-Quantile
        q95_true.append(np.quantile(proj_true, 0.95))
        q95_vanilla.append(np.quantile(proj_vanilla, 0.95))
        q95_TAF.append(np.quantile(proj_TAF, 0.95))
        q95_gTAF.append(np.quantile(proj_gTAF, 0.95))
        q95_mTAF.append(np.quantile(proj_mTAF, 0.95))
        # 3.8 99%-Quantile
        q99_true.append(np.quantile(proj_true, 0.99))
        q99_vanilla.append(np.quantile(proj_vanilla, 0.99))
        q99_TAF.append(np.quantile(proj_TAF, 0.99))
        q99_gTAF.append(np.quantile(proj_gTAF, 0.99))
        q99_mTAF.append(np.quantile(proj_mTAF, 0.99))

    # Generate plots
    plt.scatter(mean_true, mean_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(mean_true, mean_TAF, label="TAF", alpha=alpha)
    plt.scatter(mean_true, mean_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(mean_true, mean_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(mean_true), np.min(mean_vanilla)]), np.max([np.max(mean_true), np.max(mean_vanilla)]))
    plt.legend()
    plt.plot(diag, diag, "--")
    #plt.title("Mean")
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_mean.pdf")
    plt.clf()

    plt.scatter(std_true, std_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(std_true, std_TAF, label="TAF", alpha=alpha)
    plt.scatter(std_true, std_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(std_true, std_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(std_true), np.min(std_vanilla)]), np.max([np.max(std_true), np.max(std_vanilla)]))
    plt.plot(diag, diag, "--")
    plt.legend()
    #plt.title("Std")
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_std.pdf")
    plt.clf()

    plt.scatter(q1_true, q1_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q1_true, q1_TAF, label="TAF", alpha=alpha)
    plt.scatter(q1_true, q1_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q1_true, q1_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q1_true), np.min(q1_vanilla)]), np.max([np.max(q1_true), np.max(q1_vanilla)]))
    plt.plot(diag, diag, "--")
    #plt.title("1%-quantile")
    plt.legend()
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_1q.pdf")
    plt.clf()

    plt.scatter(q10_true, q10_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q10_true, q10_TAF, label="TAF", alpha=alpha)
    plt.scatter(q10_true, q10_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q10_true, q10_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q10_true), np.min(q10_vanilla)]), np.max([np.max(q10_true), np.max(q10_vanilla)]))
    plt.plot(diag, diag, "--")
    #plt.title("10%-quantile")
    plt.legend()
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_10q.pdf")
    plt.clf()

    plt.scatter(q5_true, q5_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q5_true, q5_TAF, label="TAF", alpha=alpha)
    plt.scatter(q5_true, q5_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q5_true, q5_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q5_true), np.min(q5_vanilla)]), np.max([np.max(q5_true), np.max(q5_vanilla)]))
    plt.plot(diag, diag, "--")
    #plt.title("5%-quantile")
    plt.legend()
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_5q.pdf")
    plt.clf()

    plt.scatter(q90_true, q90_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q90_true, q90_TAF, label="TAF", alpha=alpha)
    plt.scatter(q90_true, q90_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q90_true, q90_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q90_true), np.min(q90_vanilla)]), np.max([np.max(q90_true), np.max(q90_vanilla)]))
    plt.plot(diag, diag, "--")
    plt.legend()
    #plt.title("90%-quantile")
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_90q.pdf")
    plt.clf()

    plt.scatter(q95_true, q95_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q95_true, q95_TAF, label="TAF", alpha=alpha)
    plt.scatter(q95_true, q95_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q95_true, q95_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q95_true), np.min(q95_vanilla)]), np.max([np.max(q95_true), np.max(q95_vanilla)]))
    plt.plot(diag, diag, "--")
    plt.legend()
    #plt.title("95%-quantile")
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_95q.pdf")
    plt.clf()

    plt.scatter(q99_true, q99_vanilla, label="vanilla", alpha=alpha)
    plt.scatter(q99_true, q99_TAF, label="TAF", alpha=alpha)
    plt.scatter(q99_true, q99_gTAF, label="gTAF", alpha=alpha)
    plt.scatter(q99_true, q99_mTAF, label="mTAF", alpha=alpha)
    diag = np.linspace(np.min([np.min(q99_true), np.min(q99_vanilla)]), np.max([np.max(q99_true), np.max(q99_vanilla)]))
    plt.plot(diag, diag, "--")
    plt.legend()
    #plt.title("99%-quantile")
    plt.ticklabel_format(style="sci", scilimits=(0, 0))
    plt.xlabel("data")
    plt.ylabel("flow")
    plt.tight_layout()
    plt.savefig("plots/random_proj_99q.pdf")
    plt.clf()

import torch
import argparse
import time
import os.path
import sys
import inspect
import numpy as np
import copulas
from copulas.multivariate import GaussianMultivariate
from scipy.stats import kstest
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn import functional as F
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.flows import MAF, NSF, compute_arealoglog, experiment


import wandb
wandb.init(anonymous="allow", project="mTAF_ICML")



parser = argparse.ArgumentParser()
parser.add_argument("--df", type=int, default=2)
parser.add_argument("--num_heavy", type=int, default=4)
parser.add_argument("--dim", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

wandb.config.update(args)


D = args.dim
wandb.config.update({"comment": "synth"})

if __name__=="__main__":
    # get the data
    exp = experiment("copula")
    exp.load_data(D, num_heavy=args.num_heavy, df=args.df, seed=args.seed)
    data_train = exp.data_train
    data_test = exp.data_test
    data_val = exp.data_val
    # concat train and val data:
    data_train = np.vstack([data_train, data_val])

    # fit a Gaussian Copula model:
    copula = GaussianMultivariate()
    copula.fit(data_train)

    # compute test loss:
    pdf = copula.probability_density(data_test)
    test_loss = - np.log(pdf.mean())

    # sample synth. data:
    num_samples = 10000
    synth_data = copula.sample(num_samples).to_numpy()

    # compute area
    area = []
    for j in range(D):
        marginal_area = compute_arealoglog(data_test[:num_samples, j], synth_data[:, j])
        area.append(np.round(marginal_area, 5))

    area_light = np.mean(area[:-args.num_heavy])
    area_heavy = np.mean(area[-args.num_heavy:])
    print(f"area_light {area_light}")
    print(f"area_heavy {area_heavy}")

    # tvar:
    tvar_dif = []
    tvar99_dif = []
    for component in range(D):
        sorted_abs_samps_synth = np.sort(np.abs(synth_data[:, component]))
        sorted_abs_data_test = np.sort(np.abs(data_test[:, component]))
        alpha = 0.95
        tvar_copula = 1 / (1 - alpha) * np.mean(sorted_abs_samps_synth[int(alpha * num_samples):])
        tvar_test = 1 / (1 - alpha) * np.mean(sorted_abs_data_test[int(alpha * len(sorted_abs_data_test)):])
        tvar_dif.append(np.abs(tvar_test - tvar_copula))

    tvar_heavy = np.mean(tvar_dif[-args.num_heavy:])
    tvar_light = np.mean(tvar_dif[:-args.num_heavy])


    # qq-plots
    setting = f"df{args.df}h{args.num_heavy}"
    for component in range(D):
        PATH = f"{setting}/qq/copula{component + 1}.png"
        sorted_copula_samps = np.sort(synth_data[:num_samples, component])
        sorted_test_samps = np.sort(data_test[:num_samples, component])
        pp_x = sm.ProbPlot(sorted_test_samps)
        pp_y = sm.ProbPlot(sorted_copula_samps)

        if component == (D - 1):
            fig = qqplot_2samples(pp_x, pp_y, line="45")
            title = f"Gaussian Copula"
            plt.xlabel("True Samples")
            plt.ylabel("Flow Samples")
            plt.title(title)
            wandb.log({"qq-plot": wandb.Image(fig)})
        qqplot_2samples(pp_x, pp_y, line="45")
        plt.xlabel("True Samples")
        plt.ylabel("Flow Samples")
        plt.savefig(PATH)
        plt.clf()

    # synth. tail estimation
    tail_estimators = []
    for j in range(D):
        marginal_data = np.abs(synth_data[:, j])
        df = pd.DataFrame()
        df["data"] = marginal_data
        df["helper"] = np.repeat(1, len(marginal_data))
        PATH_marg = "data/" + setting + "/synth_marginal" + str(j + 1)
        PATH_tailest = "data/" + setting + "/synth_tail_estimator" + str(j + 1) + ".txt"
        np.savetxt(PATH_marg + ".dat", df.values, fmt=["%10.5f", "%d"])

        script = "python ../utils/tail_estimation.py " + PATH_marg + ".dat " + PATH_marg + "_results.pdf --noise 0 --path_estimator " + PATH_tailest

        os.system(script)

        f = open(PATH_tailest)
        est = f.readline()
        est = float(est.strip())  # delete the /n
        tail_estimators.append(est)

    for j in range(D):
        print(f"Tail-estimator in component {j + 1}: {tail_estimators[j]}.")

    # log everything
    wandb.log({"test_loss": test_loss,
                "area_light": area_light,
                "area_heavy": area_heavy,
                "tvar_dif_light": tvar_light,
                "tvar_dif_heavy": tvar_heavy,
                "synth_tailest": tail_estimators
           })
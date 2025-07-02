# torch
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
# standard modules
import os, os.path
import sys
import inspect
from pathlib import Path
# data processing/logging/plotting
import wandb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot_2samples
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# nflows
import nflows.distributions.normal
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation

# local modules
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.weather_helper import load_ds_inputs
from utils.weather_helper import train_test_split_dataset
from utils.weather_helper import to_normalized_dataset
from utils.weather_helper import to_stacked_array
from utils.distributions import tDist, norm_tDist
from utils.tail_permutation import TailRandomPermutation, TailLU, LULinear
from synthetic_experiments.data_generators import copula_generator

# check device
if torch.cuda.is_available():
    torch.device("cuda")
    device = "cuda"
else:
    torch.device("cpu")
    device = "cpu"


class experiment:
    def __init__(self, model, comment=""):
        """
        Implements the whole empirical evaluation: 
        - load_data
        - estimate_tails
        - config: configuring the model, including number of layers, flow architecture, etc.
        - train 
        - compute_area, compute_tvar, synth_tailest, generate_qqplots  
        
        :param model: String describing the base distribution: "vanilla", "TAF", "mTAF", or "gTAF" 
        :param comment: Optional string, comment is added to the saved model. 
        """
        self.model = model
        self.heavy_tailed = []
        self.list_tailindexes = []
        self.comment = comment

    def load_data(self, data, num_heavy=0, df=0, estimate_tails=False, seed=1):
        """
        data: either ’weather’ or an integer that defines the dimensionality $D$ in the synthetic experiment.
        num_heavy: number of heavy-tailed components of the synthetic data set.
        df: degree of freedom of the heavy-tailed marginals in the synthetic data set.
        estimate_tails: True or False. If False, then read locally saved tail estimates.
        seed: for reproducibility.
        """
        self.data = data
        if self.data == "weather":
            PROJ_PATH = Path.cwd().parent
            ds_true_in = load_ds_inputs(PROJ_PATH)
            ds_true, stats_info = to_normalized_dataset(ds_true_in)
            self.stats_info = stats_info
            ds_train, ds_test = train_test_split_dataset(ds_true, test_size=0.6, dim='column', shuffle=True, seed=42)
            ds_test, ds_validation = train_test_split_dataset(ds_test, test_size=0.33334, dim='column', shuffle=True, seed=42)
            ds_train, _ = train_test_split_dataset(ds_train, train_size=1, dim='column', shuffle=False)

            ds_true_in['pressure_hl'] /= 100  # Convert Pa to hPa

            col_name = "pressure_hl"
            # impute a tiny bit of noise for
            # atmospheric pressure below 80:
            noise_sd = 0.001
            impute_noise = True
            ds_pressure_train = ds_train[col_name].to_numpy()
            ds_pressure_train[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_pressure_train), 80])
            ds_train[col_name].data = ds_pressure_train
            ds_pressure_test = ds_test[col_name].to_numpy()
            ds_pressure_test[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_pressure_test), 80])
            ds_test[col_name].data = ds_pressure_test
            ds_pressure_val = ds_validation[col_name].to_numpy()
            ds_pressure_val[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_pressure_val), 80])
            ds_validation[col_name].data = ds_pressure_val

            col_name = "layer_cloud_optical_depth"
            ds_opt_train = ds_train[col_name].to_numpy()
            ds_opt_train[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_opt_train), 80])
            ds_train[col_name].data = ds_opt_train
            ds_opt_test = ds_test[col_name].to_numpy()
            ds_opt_test[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_opt_test), 80])
            ds_test[col_name].data = ds_opt_test
            ds_opt_val = ds_validation[col_name].to_numpy()
            ds_opt_val[:, :80] += impute_noise * np.random.normal(0, noise_sd, [len(ds_opt_val), 80])
            ds_validation[col_name].data = ds_opt_val

            self.data_train = to_stacked_array(ds_train)[0].to_numpy()
            self.data_test = to_stacked_array(ds_test)[0].to_numpy()
            self.data_val = to_stacked_array(ds_validation)[0].to_numpy()

            self.D = self.data_train.shape[1]
        else:
            self.df = df
            self.num_heavy = num_heavy
            self.setting = "df" + str(self.df) + "h" + str(num_heavy)
            self.D = int(self.data)
            self.data = ""
            if self.D==50:
                num_samps = 135000
            else:
                num_samps = 100000
            generator = copula_generator(self.D, self.num_heavy, self.df, seed)
            data = generator.get_data(num_samps)
            self.marginals = generator.get_marginals()
            self.dist = generator.get_dist()
            if self.D==50:
                # 75.000 test set, 10.000 val set, 50.000 train set
                self.data_train, self.data_test = train_test_split(data, test_size=75000)
                self.data_train, self.data_val = train_test_split(self.data_train, test_size=10000)
            else:
                # 75.000 test set, 10.000 val set, 15.000 train set
                self.data_train, self.data_test = train_test_split(data, test_size=3/4)
                self.data_train, self.data_val = train_test_split(self.data_train, test_size=2/5)

            # normalize data:
            mu = data.mean(axis=0)
            s = data.std(axis=0)
            self.data_train = (self.data_train - mu)/s
            self.data_test = (self.data_test - mu)/s
            self.data_val = (self.data_val - mu)/s


        if self.model in ["mTAF", "gTAF", "mTAF(fix)"]:
            if self.data == "weather":
                self.get_weather_tails()
            else:
                self.get_tails(estimate_tails=estimate_tails)


    def get_weather_tails(self):
        """
        Set tails as explained in the Paper. We initialize the degree of freedom by 5, which was arbitrary chosen.
        This choice does not have a significant effect since a suited degree of freedom is learned during training.
        """
        ls_tailindices = [0] * 80
        ls_tailindices += [5] * (137 - 80)
        ls_tailindices += [0] * 100
        ls_tailindices += [5] * 38
        ls_tailindices += [0] * 58
        ls_tailindices += [5] * (137 - 58)


        self.list_tailindexes = ls_tailindices
        for tail_index in ls_tailindices:
            if tail_index > 0:
                self.heavy_tailed.append(True)
            else:
                self.heavy_tailed.append(False)

        self.num_heavy = np.sum(np.array(self.heavy_tailed))
        self.num_light = int(self.D - self.num_heavy)
        # 2. Reorder the Marginals
        # here: in a block-like fashion
        self.permutation = list(range(80)) # light-tailed temperature
        # light-tailed pressure
        pressure_light = list(range(137, 237))
        pressure_light.reverse()
        self.permutation += pressure_light
        self.permutation += list(range(137+100+38, 137 + 100 + 38 + 58)) # light-tailed depth
        self.permutation += list(range(80, 137)) # heavy-tailed temperature
        self.permutation += list(range(137+100, 137+100+38)) # heavy-tailed pressure
        self.permutation += list(range(137 + 100 + 38 + 58, self.D)) # heavy-tailed depth

        # define the inverse of the permutation for reordering
        self.inv_perm = np.zeros(self.D, dtype=np.int32)
        for j in range(self.D):
            self.inv_perm[self.permutation[j]] = j
        self.tail_index_permuted = np.array(self.list_tailindexes)[self.permutation]
        # permute data
        self.data_train = self.data_train[:, self.permutation]
        self.data_val = self.data_val[:, self.permutation]
        self.data_test = self.data_test[:, self.permutation]

    def get_tails(self, PATH="", estimate_tails=False):
        if estimate_tails:
            self.estimate_tails()
        # 1. Get the Marginals
        for j in range(self.D):
            if PATH == "":
                if self.data != "":  # real-world data
                    PATH_tailest = "data/marginals/" + self.data + "/tail_estimator" + str(
                        j + 1) + ".txt"
                else:  # synthetic data
                    PATH_tailest = "data/" + self.setting + "/tail_estimator" + str(j + 1) + ".txt"
            else:
                PATH_tailest = PATH + str(j + 1) + ".txt"
            if self.model in ["mTAF", "mTAF(fix)"]:
                assert Path(PATH_tailest).exists(), f"File {PATH_tailest} does not exists. Try rerunning the code with the flag --estimate_tails True"
                tail_index = np.loadtxt(PATH_tailest)
            else:
                print(f"Marginal Nr. {j+1} is set to a light-tailed marginal.")
                tail_index = 0

            # set tail_index > 10 to light-tailed distribution since t_v -> N(0,1) in distribution when v->\infty
            if tail_index > 10:
                tail_index = 0

            # for gTAF: initialize all tail indeces by 10.
            if self.model=="gTAF":
                tail_index = 10

            # in restricted gTAF: set all light-tailed tail indeces to 10
            if self.model=="res_gTAF":
                if tail_index==0:
                    tail_index = 10

            if tail_index == 0:
                self.heavy_tailed.append(False)
                print("{}th Marginal is detected as light-tailed.".format(j + 1))
            else:
                print("{}th Marginal is detected as heavy-tailed with tail-index {}.".format(j + 1, tail_index))

                if self.model!="res_gTAF":
                    self.heavy_tailed.append(True)
                else: # in res_gTAF: if tail_index is equal to 10, than it is a light-tailed marginal
                    if tail_index==10:
                        self.heavy_tailed.append(False)
                    else:
                        self.heavy_tailed.append(True)

            self.list_tailindexes.append(tail_index)

        self.num_heavy = np.sum(np.array(self.heavy_tailed))
        self.num_light = int(self.D - self.num_heavy)
        # 2. Reorder the Marginals
        self.permutation = np.argsort(np.array(self.heavy_tailed))
        self.inv_perm = np.zeros(self.D, dtype=np.int32)  # for reordering
        for j in range(self.D):
            self.inv_perm[self.permutation[j]] = j

        self.tail_index_permuted = np.array(self.list_tailindexes)[self.permutation]

        self.data_train = self.data_train[:, self.permutation]
        self.data_val = self.data_val[:, self.permutation]
        self.data_test = self.data_test[:, self.permutation]

    def estimate_tails(self):
        """
        Estimates the tails of each marginal of self.data_val i.e. the validation data set.
        Based on the Implementation in https://github.com/ivanvoitalov/tail-estimation
        """
        for j in range(self.D):
            marginal_data = np.abs(self.data_val[:, j])
            df = pd.DataFrame()
            df["data"] = marginal_data
            df["helper"] = np.repeat(1, len(marginal_data))
            if self.data=="":
                Path(f"data/{self.setting}/").mkdir(parents=True, exist_ok=True)
                PATH_marg = f"data/{self.setting}/marginal{str(j + 1)}"
                PATH_tailest = f"data/{self.setting}/tail_estimator{str(j + 1)}.txt"
            else:
                Path(f"data/marginals/{self.data}/").mkdir(parents=True, exist_ok=True)
                PATH_marg = f"data/marginals/{self.data}/marginal{j+1}"
                PATH_tailest = f"data/marginals/{self.data}/tail_estimator{j+1}.txt"
            np.savetxt(PATH_marg + ".dat", df.values, fmt=["%10.5f", "%d"])

            script = "python3 ../utils/tail_estimation.py " + PATH_marg + ".dat " + PATH_marg + "_results.pdf --noise 0 --path_estimator " + PATH_tailest

            os.system(script)

    def config(self, *args):
        """
        1. Define the flow
        2. Define the optimizer
        3. Define the scheduler
        """
        raise NotImplementedError()

    def train(self, batch_size=512, grad_clip=False):
        train_dataloader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(self.data_test, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(self.data_val, batch_size=batch_size, shuffle=True)

        # set best validation loss to infinity before training and update it during training
        best_val_loss = np.inf

        # track the learned dfs
        self.ls_dfs = []

        # training loop:
        tbar = tqdm(range(self.train_steps))
        for step in tbar:
            self.flow.train()
            batch = next(iter(train_dataloader)).type(torch.float32).to(device)
            self.optimizer.zero_grad()
            loss = -self.flow.log_prob(inputs=batch)
            loss = loss.mean()
            loss.backward()
            if grad_clip:
                clip_grad_norm_(self.flow.parameters(), 5)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            if (step + 1) % 250 == 0:
                trn_loss = np.mean(loss.detach().cpu().numpy())
                print(f"Training loss in batch {step +1 }/{self.train_steps}: {trn_loss}")

                # validate performance and save best model on validation set
                val_loss = []
                self.flow.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        val_batch = batch.type(torch.float32).to(device)
                        loss = -self.flow.log_prob(inputs=val_batch.detach())
                        loss = loss.mean()
                        val_loss.append(loss.detach().cpu().numpy())
                    val_loss_avg = np.mean(val_loss)
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    torch.save(self.flow.state_dict(), self.PATH_model)
                    best_step = step + 1
                print(f"Validation loss after {step + 1}/{self.train_steps} training batches: {val_loss_avg}")
                if self.model != "vanilla":
                    print("Learned degree of Freedom:")
                    for parameter in self.base_dist.parameters():
                        df = parameter.detach().cpu().numpy()
                        print(df)
                        self.ls_dfs.append(df)

                with open(self.PATH_results + "_train.txt", "a") as f:
                    f.write(str(trn_loss) + " " + str(self.model_nr) + "\n")
                with open(self.PATH_results + "_val.txt", "a") as f:
                    f.write(str(val_loss_avg) + " " + str(self.model_nr) + "\n")

                wandb.log({"trn_loss": trn_loss,
                          "val_loss": val_loss_avg})

        # load the best model:
        self.flow.load_state_dict(torch.load(self.PATH_model, map_location=device))
        # compute test loss:
        self.flow.eval()
        with torch.no_grad():
            loss_test = []
            for batch in test_dataloader:
                test_batch = batch.type(torch.float32).to(device)
                loss = -self.flow.log_prob(inputs=test_batch.detach()).mean()
                loss_test.append(loss.cpu().detach().numpy())
        average_testloss = np.mean(loss_test)
        print("{}: Final Test loss after {} Training steps: {}".format(self.model, self.train_steps, average_testloss))
        print(f"Best Model was learned after {best_step} steps.")
        if self.track_results:
            with open(self.PATH_results + "test.txt", "a") as f:
                    f.write(str(average_testloss) + " " + str(self.model_nr) + "\n")

        # after training: permute the data back:
        if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
            self.data_train = self.data_train[:, self.inv_perm]
            self.data_val = self.data_val[:, self.inv_perm]
            self.data_test = self.data_test[:, self.inv_perm]
        return np.round(average_testloss, 2)

    def compute_area(self, sampling=True, num_samples=10000):
        self.flow.eval()
        with torch.no_grad():
            if sampling:
                self.samps_flow = self.flow.to(device).sample(num_samples).detach().cpu().numpy()
                if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
                    self.samps_flow = self.samps_flow[:, self.inv_perm]


        # compute marginal areas
        area = []
        for j in range(self.D):
            marginal_area = compute_arealoglog(self.data_test[:num_samples, j], self.samps_flow[:, j])
            area.append(np.round(marginal_area, 5))

        if self.data=="":
            Path(f"{self.setting}/area/").mkdir(parents=True, exist_ok=True)
            PATH = f"{self.setting}/area/{self.flowtype}{self.model}.txt"
        else:
            PATH = f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/area.txt"

        if self.track_results:
            with open(PATH, "a") as f:
                f.write(" ".join(str(e) for e in area) + "\n")
        else:
            print(f"({self.model}) Average area under curve: {np.round(np.mean(area), 2)}")

        return area

    def compute_tvar(self, alpha=0.95, sampling=False, num_samples=10000):
        self.flow.eval()
        with torch.no_grad():
            if sampling:
                self.samps_flow = self.flow.to(device).sample(num_samples).detach().cpu().numpy()
                if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
                    self.samps_flow[: ,self.inv_perm]

        tvar_differences = []
        for component in range(self.D):
            sorted_abs_samps_flow = np.sort(np.abs(self.samps_flow[:, component]))
            sorted_abs_data_test = np.sort(np.abs(self.data_test[:, component]))
            tvar_flow = 1/(1-alpha) * np.mean(sorted_abs_samps_flow[int(alpha*num_samples): ])
            tvar_test = 1/(1-alpha) * np.mean(sorted_abs_data_test[int(alpha*len(self.data_test)): ])

            print(f"Component {component + 1}: tVaR in test data: {np.round(tvar_test, 2)}, tVaR in flow samples: {np.round(tvar_flow, 2)}")
            tvar_differences.append(np.abs(tvar_test - tvar_flow))
        if self.data == "":
            Path(f"{self.setting}/tvar/").mkdir(parents=True, exist_ok=True)
            PATH = f"{self.setting}/tvar/{self.flowtype}{self.model}.txt"
        else: # weather data
            PATH = f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/tvar.txt"

        with open(PATH, "a") as f:
            f.write(" ".join(str(e) for e in tvar_differences) + "\n")

        return tvar_differences

    def generate_qqplots(self, sampling=False, num_samples=1000):
        self.flow.eval()
        with torch.no_grad():
            if sampling:
                self.samps_flow = self.flow.to(device).sample(num_samples).detach().cpu().numpy()
                if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
                    self.samps_flow[:, self.inv_perm]

        for component in range(self.D):
            if self.data == "":
                Path(f"{self.setting}/qq/").mkdir(parents=True, exist_ok=True)
                PATH = f"{self.setting}/qq/{self.flowtype}{self.model}{component+1}.pdf"
            else:
                Path(f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/qq/").mkdir(parents=True, exist_ok=True)
                PATH = f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/qq/comp{component+1}.pdf"
            sorted_flow_samps = np.sort(self.samps_flow[:num_samples, component])
            sorted_test_samps = np.sort(self.data_test[:num_samples, component])
            pp_x = sm.ProbPlot(sorted_test_samps)
            pp_y = sm.ProbPlot(sorted_flow_samps)
            if self.data=="":
                if component==(self.D - 1):
                    fig = qqplot_2samples(pp_x, pp_y, line="45")
                    title = f"{self.flowtype}, {self.model}"
                    plt.xlabel("True Samples")
                    plt.ylabel("Flow Samples")
                    plt.title(title)
                    wandb.log({"qq-plot": wandb.Image(fig)})
            qqplot_2samples(pp_x, pp_y, line="45")
            plt.xlabel("True Samples")
            plt.ylabel("Flow Samples")
            plt.tight_layout()
            plt.savefig(PATH)
            plt.clf()
       
    def synth_tailest(self, sampling=False, num_samples=10000):
        """
        Estimates the tail estimators of each marginal based on synthetic model samples.
        """
        # 1. get data
        self.flow.eval()
        with torch.no_grad():
            if sampling:
                self.samps_flow = self.flow.to(device).sample(num_samples).detach().cpu().numpy()
                if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
                    self.samps_flow = self.samps_flow[:, self.inv_perm]
        # 2. estimate tails based on synth. data
        tail_estimators = []
        for j in range(self.D):
            marginal_data = np.abs(self.samps_flow[:, j])
            df = pd.DataFrame()
            df["data"] = marginal_data
            df["helper"] = np.repeat(1, len(marginal_data))
            if self.data=="":
                PATH_marg = "data/" + self.setting + "/synth_marginal" + str(j + 1)
                PATH_tailest = "data/" + self.setting + "/synth_tail_estimator" + str(j + 1) + ".txt"
            else:
                PATH_marg = f"data/marginals/{self.data}/synth_marginal{j+1}"
                PATH_tailest = f"data/marginals/{self.data}/synth_tail_estimator{j+1}.txt"
            np.savetxt(PATH_marg + ".dat", df.values, fmt=["%10.5f", "%d"])

            script = "python ../utils/tail_estimation.py " + PATH_marg + ".dat " + PATH_marg + "_results.pdf --noise 0 --path_estimator " + PATH_tailest

            os.system(script)

            f = open(PATH_tailest)
            est = f.readline()
            est = float(est.strip())
            tail_estimators.append(est)


        for j in range(self.D):
            print(f"Tail-estimator in component {j+1}: {tail_estimators[j]}.")

        wandb.log({"synth_tailest": tail_estimators})
        if self.data=="":
            Path(f"{self.setting}/synth_tailest/").mkdir(parents=True, exist_ok=True)
            PATH_results = f"{self.setting}/synth_tailest/{self.flowtype}{self.model}.txt"
        else: # i.e. weather data
            Path(f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/synth_tailest/").mkdir(parents=True, exist_ok=True)
            PATH_results = f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/synth_tailest/tail_est.txt"

        with open(PATH_results, "a") as f:
            f.write(" ".join(str(e) for e in tail_estimators) + "\n")

class NSF(experiment):
    def __init__(self, model):
        """
        model: vanilla, TAF, or mTAF
        """
        super().__init__(model)

    def config(self, num_layers=5, num_hidden=100, num_blocks=1, lr=1e-4, lr_wd=0, lr_df=0.0, batch_norm=False, batch_norm_layer=False, cosine_annealing=False,
               activation=F.relu, train_steps=250000, num_bins=8, tail_bound=3, dropout=0.0, linear_layer="LU", model_nr=0, track_results=False,
               PATH=""):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.activation = activation
        self.train_steps = train_steps

        self.flowtype = "nsf"
        self.linear_layer = linear_layer
        self.model_nr = model_nr
        self.track_results = track_results

        if self.data == "":
            if self.flowtype == "maf":
                self.PATH_model = "trained_models/" + self.model + "_df" + str(self.df) + "h" + str(self.num_heavy)
            else:
                self.PATH_model = "trained_models/" + self.flowtype + self.model + "_df" + str(self.df) + "h" + str(
                    self.num_heavy)
            Path("trained_models/").mkdir(parents=True, exist_ok=True)

        else:
            self.PATH_model = "trained_models/" + self.data + "/" + self.flowtype + "/" + self.model + "/" + str(
                self.num_layers) + "layers/model_" + self.linear_layer + self.comment
            Path(f"trained_models/{self.data}/{self.flowtype}/{self.model}/{self.num_layers}layers/").mkdir(parents=True, exist_ok=True)

        if self.data == "":
            self.PATH_results = f"{self.setting}/likelihood/{self.flowtype}{self.model}{str(self.num_layers)}"
            Path(f"{self.setting}/likelihood/").mkdir(parents=True, exist_ok=True)
        else:
            self.PATH_results = f"results/{self.flowtype}/{self.model}/{self.data}/{str(self.num_layers)}layers/"
            Path(f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/").mkdir(parents=True, exist_ok=True)

        if self.model == "vanilla":
            self.base_dist = nflows.distributions.normal.StandardNormal([self.D])
        elif self.model == "TAF":
            self.base_dist = tDist([self.D])
        elif self.model in ["mTAF", "mTAF(fix)", "gTAF"]:
            self.base_dist = norm_tDist([self.D], self.tail_index_permuted)

        transforms = []
        for _ in range(self.num_layers):
            if self.data!="weather":
                if self.model in ["vanilla", "TAF", "gTAF"]:
                    transforms.append(RandomPermutation(self.D))
                    if linear_layer=="LU":
                        transforms.append(LULinear(self.D))
                else:
                    transforms.append(TailRandomPermutation(self.num_light, self.num_heavy))
                    if linear_layer=="LU":
                        transforms.append(TailLU(self.D, int(self.num_heavy)))
            else:
                if _!=0: # skip the first linear layer, i.e. let the input remain unchanged, to exploit the autoregressive nature of the weather data
                    if self.model in ["vanilla", "TAF", "gTAF"]:
                        transforms.append(RandomPermutation(self.D))
                        if linear_layer == "LU":
                            transforms.append(LULinear(self.D))
                    else:
                        transforms.append(TailRandomPermutation(self.num_light, self.num_heavy))
                        if linear_layer == "LU":
                            transforms.append(TailLU(self.D, int(self.num_heavy)))
            transforms.append(MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=self.D,
                                                                  hidden_features=self.num_hidden,
                                                                  num_blocks=self.num_blocks,
                                                                  num_bins=self.num_bins,
                                                                  tail_bound = self.tail_bound,
                                                                  use_batch_norm=self.batch_norm,
                                                                  tails="linear",
                                                                  use_residual_blocks=True,
                                                                  dropout_probability=self.dropout,
                                                                  activation=self.activation))
            if batch_norm_layer:
                transforms.append(BatchNorm(features=self.D))
        if self.model in ["vanilla", "TAF", "gTAF"]:
            transforms.append(RandomPermutation(self.D))
            if linear_layer=="LU":
                transforms.append(LULinear(self.D))
        else:
            transforms.append(TailRandomPermutation(self.num_light, self.num_heavy))
            if linear_layer == "LU":
                transforms.append(TailLU(self.D, int(self.num_heavy)))

        self.transform = CompositeTransform(transforms)

        self.flow = Flow(self.transform, self.base_dist).to(device)

        if self.model in ["vanilla"]:
            self.optimizer = optim.Adam(self.flow.parameters(), lr=lr)
        elif self.model in ["mTAF", "gTAF", "TAF", "res_gTAF"]:
            self.optimizer = optim.Adam([{"params": self.flow._distribution.parameters(), "lr": lr_df},
                                         {"params": self.flow._transform.parameters(), "weight_decay": lr_wd},
                                         {"params": self.flow._embedding_net.parameters(), "weight_decay": lr_wd}
                                         ], lr=lr)
        elif self.model == "mTAF(fix)":
            self.optimizer = optim.Adam([{"params": self.flow._distribution.parameters(), "lr": 0},
                                         {"params": self.flow._transform.parameters()},
                                         {"params": self.flow._embedding_net.parameters()}
                                         ], lr=lr)
        if cosine_annealing:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.train_steps, 0)
        else:
            self.scheduler = None

class MAF(experiment):
    def __init__(self, model):
        super().__init__(model)


    def config(self, num_layers=5, num_hidden=100, num_blocks=1, train_steps=5000, lr=1e-4, lr_wd=1e-6, lr_df=0.0, batch_norm=False, activation=F.relu, dropout=0.0, linear_layer="LU", model_nr=0, track_results=False, PATH="", res_blocks=True):
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_blocks = num_blocks
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation

        self.flowtype = "maf"
        self.linear_layer = linear_layer
        self.model_nr = model_nr
        self.track_results = track_results
        self.train_steps = train_steps
        self.res_blocks = res_blocks

        if self.data == "":
            if self.flowtype == "maf":
                self.PATH_model = "trained_models/" + self.model + "_df" + str(self.df) + "h" + str(self.num_heavy)
            else:
                self.PATH_model = "trained_models/" + self.flowtype + self.model + "_df" + str(self.df) + "h" + str(
                    self.num_heavy)
            Path("trained_models/").mkdir(parents=True, exist_ok=True)
        else:
            self.PATH_model = "trained_models/" + self.data + "/" + self.flowtype + "/" + self.model + "/" + str(
                self.num_layers) + "layers/model_" + self.linear_layer + self.comment
            Path(f"trained_models/{self.data}/{self.flowtype}/{self.model}/{self.num_layers}layers/").mkdir(parents=True, exist_ok=True)

        if self.data == "":
            if self.flowtype == "maf":
                self.PATH_results = f"{self.setting}/likelihood/{self.model}{str(self.num_layers)}"
            else:
                self.PATH_results = f"{self.setting}/likelihood/{self.flowtype}{self.model}{str(self.num_layers)}"
        else:
            self.PATH_results = f"results/{self.flowtype}/{self.model}/{self.data}/{str(self.num_layers)}layers/"


        if self.model == "vanilla":
            self.base_dist = nflows.distributions.normal.StandardNormal([self.D])
        elif self.model == "TAF":
            self.base_dist = tDist([self.D])
        elif self.model in ["mTAF", "gTAF", "mTAF(fix)"]:
            self.base_dist = norm_tDist([self.D], self.tail_index_permuted)

        transforms = []
        for _ in range(self.num_layers):
            if self.model in ["vanilla", "TAF", "gTAF"]:
                transforms.append(RandomPermutation(self.D))
                if self.linear_layer=="LU":
                    transforms.append(LULinear(self.D))
            else:
                transforms.append(TailRandomPermutation(self.num_light, self.num_heavy))
                if self.linear_layer=="LU":
                    transforms.append(TailLU(self.D, int(self.num_heavy)))

            transforms.append(MaskedAffineAutoregressiveTransform(features=self.D,
                                                                   hidden_features=self.num_hidden,
                                                                   num_blocks=self.num_blocks,
                                                                   use_batch_norm=self.batch_norm,
                                                                   use_residual_blocks=self.res_blocks,
                                                                   dropout_probability=self.dropout,
                                                                   activation=self.activation))
            transforms.append(BatchNorm(features=self.D))

        self.transform = CompositeTransform(transforms)

        self.flow = Flow(self.transform, self.base_dist).to(device)

        if self.model == "vanilla":
            self.optimizer = optim.Adam(self.flow.parameters(), lr=lr, weight_decay=lr_wd)
        elif self.model in ["mTAF", "gTAF", "TAF", "res_gTAF"]:
            self.optimizer = optim.Adam([{"params": self.flow._distribution.parameters(), "lr": lr_df},
                        {"params": self.flow._transform.parameters()},
                        {"params": self.flow._embedding_net.parameters()}
                        ], lr=lr, weight_decay=lr_wd)
        elif self.model == "mTAF(fix)":
            self.optimizer = optim.Adam([{"params": self.flow._distribution.parameters(), "lr": 0},
                                         {"params": self.flow._transform.parameters()},
                                         {"params": self.flow._embedding_net.parameters()}
                                         ], lr=lr, weight_decay=lr_wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.train_steps, 0)
        
        
def compute_arealoglog(data_true, data_synth):
    """
    Computes the Area under the log-log plot.
    data_true: 2D-Array containing the true data
    data_synth: 2D-Array containing synthetic data
    """
    n = len(data_true)
    area = 0
    for j in range(n):
        i = j + 1
        area += np.abs( np.log(np.quantile(np.abs(data_true), 1 - i/n)) - np.log(np.quantile(np.abs(data_synth), 1 - i/n))) * np.log((i + 1)/i)
    return area

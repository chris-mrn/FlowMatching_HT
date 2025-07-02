###########
# Imports #
###########
import argparse
import time
import os.path
import sys
import inspect
import numpy as np
import seaborn as sns
sns.set_context('paper', font_scale=2)
import wandb
# login to wandb:
wandb.init(anonymous="allow", project="mTAF_ICML")

from torch.nn import functional as F

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.flows import MAF, NSF


###################
# Parse Arguments #
###################
parser = argparse.ArgumentParser()
# architectural arguments
parser.add_argument("--flow",
                    type=str,
                    default="maf",
                    help="Choose the normalizing flow model:"\
                        "maf or nsf")
parser.add_argument("--marginals",
                    type=str,
                    default="mTAF",
                    help="Choose the marginals of the base distribution:"\
                        "vanilla, TAF, mTAF, or mTAF(fix).")
parser.add_argument("--linearity",
                    type=str,
                    default="LU",
                    help="Pick the linearity inbetween flow-layers: LU or permutation")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="Number of flow layers.")
parser.add_argument("--num_blocks",
                    type=int,
                    default=1,
                    help="Number of hidden layers of the networks in each flow layer (backbone networks).")
parser.add_argument("--res_blocks",
                    type=str,
                    default="False",
                    help="Residual connection in the backbone networks?")
parser.add_argument("--tail_bound",
                    type=float,
                    default=3.0,
                    help="NSF layers are defined on a bounded interval [-tail_bound, tail_bound] and linearily extended outside the interval.")

# data
parser.add_argument("--dim",
                    type=int,
                    default=8,
                    help="Pick the dimensionality of the synthetic distribution. In the paper, we investigated dim in {8, 50}")
parser.add_argument("--df",
                    type=int,
                    default=2,
                    help="Pick a Degree of Freedom. In the paper, we have used df in {2,3}.")
parser.add_argument("--num_heavy",
                    type=int,
                    default=8,
                    help="Pick the number of heavy-tailed components.")
parser.add_argument("--model_nr",
                    type=int,
                    default=0,
                    )
parser.add_argument("--estimate_tails",
                    type=bool,
                    default=False,
                    help="Estimate the tail index of each marginal before training if set to true."\
                        "Otherwise, read the tails from memory. ")

# optimization
parser.add_argument("--grad_clip",
                    type=str,
                    default="False",
                    help="Employ gradient clipping?")

parser.add_argument("--seed",
                    type=int,
                    default=1)

parser.add_argument("--wandb_comment",
                    type=str,
                    default="",
                    help="Specify wandb comment, which allows for better tracking of the results.")
args = parser.parse_args()

# login to wandb:
wandb.config.update(args)
if args.wandb_comment=="":
    wandb.config.update({"comment": "2022:ICML"})
else:
    wandb.config.update({"comment": args.wandb_comment})

#########
# Setup #
#########
setting = "df" + str(args.df) + "h" + str(args.num_heavy) # saving destination

if args.dim==50:
    num_hidden = 200 # number of hidden nodes in each backbone network
    train_steps = 20000
else:
    num_hidden = 30 # number of hidden nodes in each backbone network
    train_steps = 10000
lr = 1e-5 # learning rate
lr_wd = 1e-6 # weight decay learning rate
lr_df = 0.005 # learning rate for the degree of freedom
if args.grad_clip == "True":
    grad_clip = True
else:
    grad_clip = False
if args.res_blocks == "True":
    res_blocks = True
else:
    res_blocks = False
activation = F.relu

########################
# Initialize the model #
########################
if args.flow=="maf":
    model = MAF(args.marginals)
elif args.flow=="nsf":
    model = NSF(args.marginals)

#################
# Load the data #
#################
model.load_data(args.dim, num_heavy=args.num_heavy, df=args.df, estimate_tails=args.estimate_tails, seed=args.seed)

###################
# Train the model #
###################
if args.flow=="maf":
    model.config(num_layers=args.num_layers, num_hidden=num_hidden, num_blocks=args.num_blocks, linear_layer=args.linearity , lr=lr, lr_wd=lr_wd, lr_df=lr_df,
                 activation=activation, model_nr=args.model_nr, track_results=True, train_steps=train_steps, res_blocks=res_blocks)
    test_loss = model.train(grad_clip=grad_clip)
elif args.flow=="nsf":
    model.config(num_layers=args.num_layers, tail_bound=args.tail_bound, train_steps=train_steps, num_bins=3, linear_layer=args.linearity, track_results=True, lr_df=lr_df,
                 model_nr=args.model_nr, num_hidden=num_hidden, activation=activation, num_blocks=args.num_blocks)
    test_loss = model.train(grad_clip=grad_clip)

##################################
# Generate additional statistics #
##################################

# Area under log-log-plot
start = time.time()
area = model.compute_area()
end = time.time()
print(f"Time required to compute area (sampling is the expensive part): {end-start} seconds.")
area_light = np.mean(area[:-args.num_heavy])
area_heavy = np.mean(area[-args.num_heavy:])

# tVaR
tvar_differences = model.compute_tvar()
tvar_heavy = np.mean(tvar_differences[-args.num_heavy:])
tvar_light = np.mean(tvar_differences[:-args.num_heavy])

# QQ-plots
model.generate_qqplots()

# Tail estimation of synthetic data
model.synth_tailest()

###############
# Log results #
###############
wandb.log({"test_loss": test_loss,
            "area_light": area_light,
            "area_heavy": area_heavy,
            "tvar_dif_light": tvar_light,
            "tvar_dif_heavy": tvar_heavy
           })


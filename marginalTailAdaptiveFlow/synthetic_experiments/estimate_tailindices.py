import torch
import argparse

import os.path
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.flows import MAF

if torch.cuda.is_available():
    torch.device("cuda")
    device = "cuda"
else:
    torch.device("cpu")
    device = "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--marginals", type=str, default="mTAF")
parser.add_argument("--df", type=int, default=2)
parser.add_argument("--num_heavy", type=int, default=8)
parser.add_argument("--model_nr", type=int, default=0)
parser.add_argument("--dim", type=int, default=16)
args = parser.parse_args()



model = MAF(args.marginals)
# load data and estimate tails just once
model.load_data(args.dim, num_heavy=args.num_heavy, df=args.df, estimate_tails=True)
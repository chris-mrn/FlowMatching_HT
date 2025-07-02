#!/bin/bash
dev=1
D=8
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 4 --df 3 --dim $D
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 1 --df 3 --dim $D

CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 4 --df 2 --dim $D
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 1 --df 2 --dim $D

D=50
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 5 --df 2 --dim $D
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 5 --df 3 --dim $D
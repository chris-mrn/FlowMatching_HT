#!/bin/bash
num_blocks=2
num_hidden=100
num_layers=5
dev=1
for i in {1..25}
do
  CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals mTAF --num_layers $num_layers
  CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals gTAF --num_layers $num_layers
  CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals TAF --num_layers $num_layers
  CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals vanilla --num_layers $num_layers
done

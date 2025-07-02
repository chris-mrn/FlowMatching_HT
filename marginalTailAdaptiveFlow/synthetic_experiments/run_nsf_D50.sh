#!/bin/bash
num_layers=5
num_heavy=10
tail_bound=2
dev=2
wandb_comment="synthetic_experiments"

# estimate tail indices:
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 10 --df 2 --dim 50
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals mTAF --num_heavy 10 --df 3 --dim 50

for seed in {1..3} 
do
  for i in {1..5}
  do
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals "mTAF(fix)" --flow nsf --model_nr $i --num_heavy $num_heavy --df 2 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals "mTAF(fix)" --flow nsf --model_nr $i --num_heavy $num_heavy --df 3 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment

    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals vanilla --flow nsf --model_nr $i --num_heavy $num_heavy --df 2 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals vanilla --flow nsf --model_nr $i --num_heavy $num_heavy --df 3 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment

    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals TAF --flow nsf --model_nr $i --num_heavy $num_heavy --df 2 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals TAF --flow nsf --model_nr $i --num_heavy $num_heavy --df 3 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment

    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals gTAF --flow nsf --model_nr $i --num_heavy $num_heavy --df 2 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim 50 --marginals gTAF --flow nsf --model_nr $i --num_heavy $num_heavy --df 3 --num_layers $num_layers --tail_bound $tail_bound --seed $seed --num_blocks 2 --wandb_comment $wandb_comment
  done
done

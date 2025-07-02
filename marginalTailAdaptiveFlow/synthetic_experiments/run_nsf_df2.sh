#!/bin/bash
df=2
D=8
num_layers=5
tail_bound=2
dev=1
wandb_comment="synthetic_experiments"

# estimate tail indices
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals "mTAF(fix)" --num_heavy 4 --df $df --dim $D
CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals "mTAF(fix)" --num_heavy 1 --df $df --dim $D

for seed in {1..3}
do
  num_heavy=4
  for i in {1..25} 
  do
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals gTAF --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals vanilla --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals TAF --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals "mTAF(fix)" --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
  done

  num_heavy=1
  for i in {1..25}
  do
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals gTAF --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals vanilla --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals TAF --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --dim $D --marginals "mTAF(fix)" --flow nsf --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --tail_bound $tail_bound --seed $seed --wandb_comment $wandb_comment
  done
done

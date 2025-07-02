#!/bin/bash
df=3
D=8
dev=3 # cuda device
num_layers=5
res_blocks=True
wandb_comment="synthetic_experiments"

# estimate tail indices (already executed in nsf experiments)
#CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals "mTAF(fix)" --num_heavy 4 --df $df --dim $D
#CUDA_VISIBLE_DEVICES=$dev python3 estimate_tailindices.py --marginals "mTAF(fix)"--num_heavy 1 --df $df --dim $D

for seed in {1..3}
do
  num_heavy=4
  for i in {1..25}
  do
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals mTAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals gTAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals vanilla --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals TAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
	done

	num_heavy=1
  for i in {1..25}
  do
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals mTAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals gTAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals vanilla --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
    CUDA_VISIBLE_DEVICES=$dev python3 main.py --marginals TAF --dim $D --model_nr $i --num_heavy $num_heavy --df $df --num_layers $num_layers --res_blocks $res_blocks --seed $seed
	done
done
#!/bin/bash
for seed in {1..3}
do
  python3 copula_baseline.py --num_heavy 10 --dim 50 --df 2 --seed $seed
  python3 copula_baseline.py --num_heavy 10 --dim 50 --df 3 --seed $seed
done

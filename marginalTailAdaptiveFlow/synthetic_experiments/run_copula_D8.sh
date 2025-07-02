#!/bin/bash
df=2
D=8
for seed in {1..1}
do
  python3 copula_baseline.py --num_heavy 4 --dim $D --df $df --seed $seed
  python3 copula_baseline.py --num_heavy 1 --dim $D --df $df --seed $seed
done

df=3
for seed in {1..3}
do
  python3 copula_baseline.py --num_heavy 1 --dim $D --df $df --seed $seed
  python3 copula_baseline.py --num_heavy 4 --dim $D --df $df --seed $seed
done

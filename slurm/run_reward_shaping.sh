#!/bin/bash

mkdir -p ../logs

seeds=(1 12)

for seed in "${seeds[@]}"
do
    echo "Running experiment with seed: $seed"
    nohup python ../her.py --seed $seed --n_episode 300000 --n_bits 35 --device cuda:1 --her_strategy final > ../logs/output_35_seed_$seed.log 2>&1
    echo "Experiment with seed: $seed completed."
done

for seed in "${seeds[@]}"
do
    echo "Running experiment with seed: $seed"
    nohup python ../her.py --seed $seed --n_episode 300000 --n_bits 35 --device cuda:1 --her_strategy episode > ../logs/output_35_seed_$seed.log 2>&1
    echo "Experiment with seed: $seed completed."
done

echo "All tasks have been completed."

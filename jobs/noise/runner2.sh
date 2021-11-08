#!/bin/bash
#SBATCH --job-name=Noise
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

tree_type=$1
method=$2
strategy=$3
noise_frac=$4
seed=$5

. jobs/config.sh

dataset=${datasets[${SLURM_ARRAY_TASK_ID}]}

python3 scripts/experiments/multi_test/noise.py \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --strategy $strategy \
  --noise_frac $noise_frac \
  --seed $seed \
  --n_repeat 1 \

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load miniconda
conda activate jbrophy-20210713

dataset=$2
tree_type=$3
method=$4
inf_obj=$5
trunc_frac=$6
update_set=$7
local_op=$8
global_op=$9

python3 scripts/experiments/roar.py \
  --skip $skip \
  --dataset $dataset \
  --tree_type $tree_type \
  --method $method \
  --inf_obj $inf_obj \
  --trunc_frac $trunc_frac \
  --update_set $update_set \
  --local_op $local_op \
  --global_op $global_op \

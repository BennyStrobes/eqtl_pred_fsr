#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-0:20                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=2G                        # Memory total in MiB (for all cores)


fam_file=${1}
n_folds=${2}
seed=${3}
fold_file=${4}

source ~/.bash_profile
conda activate plink_env

python generate_cross_validation_folds.py --fam_file ${fam_file} --n_folds ${n_folds} --seed ${seed} --fold_file ${fold_file}

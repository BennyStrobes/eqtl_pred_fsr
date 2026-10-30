#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-10:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores)


gibbs_input_summary_file=${1}
fold_file=${2}
held_out_fold=${3}
gibbs_results_stem=${4}
output_file=${5}

source ~/.bash_profile
conda activate plink_env

date

python evaluate_held_out_expression_prediction.py --gibbs_input_summary_file ${gibbs_input_summary_file} --fold_file ${fold_file} --held_out_fold ${held_out_fold} --gibbs_results_stem ${gibbs_results_stem} --output_file ${output_file}

date

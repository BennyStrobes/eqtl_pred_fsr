#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-4:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores): the sparse effect samples and one chromosome's dosages are held in memory


gibbs_results_stem=${1}
output_stem=${2}
# The inputs of the fit, for the in-window LD scores (README.md, section 5h)
gibbs_input_summary_file=${3}
fold_file=${4}
held_out_fold=${5}

source ~/.bash_profile
conda activate plink_env

date

python check_prior_by_variant_annotation.py --gibbs_results_stem ${gibbs_results_stem} --output_stem ${output_stem} --gibbs_input_summary_file ${gibbs_input_summary_file} --fold_file ${fold_file} --held_out_fold ${held_out_fold}

date

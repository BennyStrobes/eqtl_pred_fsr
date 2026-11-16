#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-1:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=30G                        # Memory total in MiB (for all cores)

# Evaluate one fit of step 2 in an external test set (e.g. the African-ancestry individuals of the tissue) instead of its held-out fold
# (evaluate_held_out_expression_prediction.py with --test_expression_file; README.md, section 7b)

gibbs_input_summary_file=${1}
fold_file=${2}
held_out_fold=${3}  # fold the fit held out (defines its training individuals)
gibbs_results_stem=${4}
test_expression_file=${5}
test_genotype_sample_mapping_file=${6}
plink_stem=${7}
output_file=${8}
# Optional: "split_samples" for y_hat and the expected quantities from separate blocks of retained samples (--split_samples)
split_samples=${9:-}

split_args=""
if [ "${split_samples}" == "split_samples" ]; then
	split_args="--split_samples"
fi

source ~/.bash_profile
conda activate plink_env

date

python evaluate_held_out_expression_prediction.py --gibbs_input_summary_file ${gibbs_input_summary_file} --fold_file ${fold_file} --held_out_fold ${held_out_fold} --gibbs_results_stem ${gibbs_results_stem} --output_file ${output_file} --test_expression_file ${test_expression_file} --test_genotype_sample_mapping_file ${test_genotype_sample_mapping_file} --plink_stem ${plink_stem} ${split_args}

date

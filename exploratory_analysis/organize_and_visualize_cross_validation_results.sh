#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-1:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=10G                        # Memory total in MiB (for all cores)


tissue=${1}
n_folds=${2}
gibbs_results_dir=${3}
held_out_evaluation_dir=${4}
visualization_dir=${5}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${visualization_dir}

evaluation_files=""
gibbs_results_stems=""
folds=""
for fold in $(seq 1 ${n_folds}); do
	evaluation_files=${evaluation_files}" "${held_out_evaluation_dir}${tissue}"_fold_"${fold}"_held_out_evaluation.txt"
	gibbs_results_stems=${gibbs_results_stems}" "${gibbs_results_dir}${tissue}"_fold_"${fold}
	folds=${folds}" "${fold}
done

# Concatenate per-fold evaluations and hyper-parameter traces
python organize_cross_validation_results.py --evaluation_files ${evaluation_files} --gibbs_results_stems ${gibbs_results_stems} --folds ${folds} --output_stem ${visualization_dir}${tissue}"_cv_organized"

# pdf plots
Rscript visualize_cross_validation_results.R ${visualization_dir}${tissue}"_cv_organized" ${visualization_dir} ${tissue}

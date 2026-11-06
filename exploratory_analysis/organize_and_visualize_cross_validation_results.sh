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
# Optional: suffix of the evaluation files (e.g. _split_samples for ${tissue}_fold_<k>_held_out_evaluation_split_samples.txt), appended to the
# tissue in the organized files and the pdf names
evaluation_label=${6:-}

source ~/.bash_profile
conda activate plink_env
mkdir -p ${visualization_dir}

evaluation_files=""
gibbs_results_stems=""
folds=""
# Folds whose fit or evaluation is missing (e.g. a chain that did not finish) are left out with a warning; the plots show the folds present
for fold in $(seq 1 ${n_folds}); do
	evaluation_file=${held_out_evaluation_dir}${tissue}"_fold_"${fold}"_held_out_evaluation"${evaluation_label}".txt"
	gibbs_results_stem=${gibbs_results_dir}${tissue}"_fold_"${fold}
	if [ ! -f ${evaluation_file} ] || [ ! -f ${gibbs_results_stem}"_hyperparameter_trace.txt" ]; then
		echo "warning: fold ${fold} of ${tissue} left out (missing ${evaluation_file} or ${gibbs_results_stem}_hyperparameter_trace.txt)"
		continue
	fi
	evaluation_files=${evaluation_files}" "${evaluation_file}
	gibbs_results_stems=${gibbs_results_stems}" "${gibbs_results_stem}
	folds=${folds}" "${fold}
done
if [ -z "${folds}" ]; then
	echo "no fold of ${tissue} has both a fit and an evaluation"
	exit 1
fi

# Concatenate per-fold evaluations and hyper-parameter traces
python organize_cross_validation_results.py --evaluation_files ${evaluation_files} --gibbs_results_stems ${gibbs_results_stems} --folds ${folds} --output_stem ${visualization_dir}${tissue}${evaluation_label}"_cv_organized"

# pdf plots
Rscript visualize_cross_validation_results.R ${visualization_dir}${tissue}${evaluation_label}"_cv_organized" ${visualization_dir} ${tissue}${evaluation_label}

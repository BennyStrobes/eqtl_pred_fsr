#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-0:30                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=10G                        # Memory total in MiB (for all cores)

# Plots of the African-ancestry evaluation of one model (driver step 6; README.md, section 8, plot 10), after step 5 (and step 3, for the comparison)
#   (a) the step 4 plots of the AFR evaluation (organize_cross_validation_results.py with the one fold, visualize_cross_validation_results.R),
#       named ${tissue}${model_label}_afr${evaluation_label}_*
#   (b) EUR held-out fold against AFR (visualize_afr_vs_eur_prediction.R), named ${tissue}${model_label}_afr_vs_eur${evaluation_label}_*

tissue=${1}  # tissue and model label, e.g. Whole_Blood_grid_gibbs (as step 4)
fold=${2}  # fold held out by the fit that was evaluated in AFR
gibbs_results_dir=${3}
held_out_evaluation_dir=${4}
afr_evaluation_dir=${5}
afr_visualization_dir=${6}
evaluation_label=${7}  # suffix of the evaluation files (e.g. _split_samples)

source ~/.bash_profile
conda activate plink_env
mkdir -p ${afr_visualization_dir}

evaluation_file=${afr_evaluation_dir}${tissue}"_fold_"${fold}"_afr_evaluation"${evaluation_label}".txt"
if [ ! -f ${evaluation_file} ]; then
	echo "missing ${evaluation_file}"
	exit 1
fi

# (a) Step 4 plots of the AFR evaluation. visualize_cross_validation_results.R draws its single-fold figures for fold 3 (its single_fold); the by-fold
# figures have one panel
organized_stem=${afr_visualization_dir}${tissue}"_afr"${evaluation_label}"_cv_organized"
python organize_cross_validation_results.py --evaluation_files ${evaluation_file} --gibbs_results_stems ${gibbs_results_dir}${tissue}"_fold_"${fold} --folds ${fold} --output_stem ${organized_stem}
Rscript visualize_cross_validation_results.R ${organized_stem} ${afr_visualization_dir} ${tissue}"_afr"${evaluation_label}

# (b) EUR held-out against AFR
Rscript visualize_afr_vs_eur_prediction.R ${held_out_evaluation_dir} ${afr_evaluation_dir} ${afr_visualization_dir} ${fold} ${evaluation_label} ${tissue}"_afr_vs_eur"${evaluation_label} ${tissue}

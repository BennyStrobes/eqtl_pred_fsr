#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-1:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=10G                        # Memory total in MiB (for all cores)

# Plots of the latent gene classes of one fit (README.md, section 8, plot 9). Run after organize_and_visualize_cross_validation_results.sh, whose
# per-gene file it joins.
tissue=${1}
n_folds=${2}
gibbs_results_dir=${3}
visualization_dir=${4}
# Optional: suffix of the organized files (e.g. _split_samples), as given to organize_and_visualize_cross_validation_results.sh
evaluation_label=${5:-}

source ~/.bash_profile
conda activate plink_env

gibbs_results_stems=""
folds=""
# Folds whose fit is missing (e.g. a chain that did not finish) are left out with a warning
for fold in $(seq 1 ${n_folds}); do
	gibbs_results_stem=${gibbs_results_dir}${tissue}"_fold_"${fold}
	if [ ! -f ${gibbs_results_stem}"_prior_rows.npz" ]; then
		echo "warning: fold ${fold} of ${tissue} left out (missing ${gibbs_results_stem}_prior_rows.npz)"
		continue
	fi
	gibbs_results_stems=${gibbs_results_stems}" "${gibbs_results_stem}
	folds=${folds}" "${fold}
done
if [ -z "${folds}" ]; then
	echo "no fold of ${tissue} has a gene-class fit"
	exit 1
fi

python organize_gene_classes.py --gibbs_results_stems ${gibbs_results_stems} --folds ${folds} --per_gene_fold_file ${visualization_dir}${tissue}${evaluation_label}"_cv_organized_per_gene_fold.txt" --output_stem ${visualization_dir}${tissue}${evaluation_label}"_gene_classes"

# pdf plots
Rscript visualize_gene_classes.R ${visualization_dir}${tissue}${evaluation_label}"_gene_classes" ${visualization_dir} ${tissue}${evaluation_label}

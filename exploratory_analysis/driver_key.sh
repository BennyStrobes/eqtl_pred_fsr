#!/bin/bash
#SBATCH -c 1                               # Request one core
#SBATCH -t 0-20:00                         # Runtime in D-HH:MM format
#SBATCH -p bch-compute                           # Partition to run in
#SBATCH --mem=20G                        # Memory total in MiB (for all cores)

#####################
# Input data
#####################

# Processed GTEx data (from gdl_uncertainty_paper/gtex_eqtl_expression_processing)
gtex_processing_dir="/lab-share/CHIP-Strober-e2/Public/ben/gdl_uncertainty_paper/gtex_eqtl_expression_processing/"

# Residualized, renormalized expression per tissue (EUR individuals): ${tissue}.v8.residualized_expression_renormalized.bed
gtex_expression_dir=${gtex_processing_dir}"residualized_expression/"

# Chromosome-specific plink genotype (gtex_v9_eqtl_chr${chrom}.{bed,bim,fam}) and per-tissue genotype sample mapping files
gtex_genotype_dir=${gtex_processing_dir}"plink_processed_genotype/"
gtex_plink_stem=${gtex_genotype_dir}"gtex_v9_eqtl_chr"

# Protein-coding genes
gtex_pc_genes_gtf="/lab-share/CHIP-Strober-e2/Public/gene_annotation_files/gencode.v39.gtex.protein_coding.genes.gtf"


#####################
# Output directories
#####################

output_root="/lab-share/CHIP-Strober-e2/Public/ben/eqtl_pred_fsr/"

cv_folds_dir=${output_root}"cv_folds/"

gibbs_input_dir=${output_root}"gibbs_input/"

gibbs_results_dir=${output_root}"gibbs_results/"

held_out_evaluation_dir=${output_root}"held_out_evaluation/"

visualization_dir=${output_root}"visualization/"


#####################
# Settings
#####################
n_folds="4"
fold_seed="1"
cis_window="100000"
# Gibbs sampler: total iterations, burn-in, and thinning of the retained samples ((n_iters - burn_in) / thin retained)
n_iters="10000"
burn_in="5000"
thin="5"
fold_file=${cv_folds_dir}"gtex_individuals_"${n_folds}"_folds_seed_"${fold_seed}".txt"
# Tempered Gibbs alternative (README.md, section 5b): same model and settings, results named ${tissue}_tempered_* so that both fits can be compared
tempered_label="_tempered"
# GWFM-style tempered Gibbs (README.md, section 5c): results named ${tissue}_tempered_gwfm_*
gwfm_label="_tempered_gwfm"
# Fixed-grid mixture prior (README.md, section 5e): a different prior on the same data, results named ${tissue}_grid_* (standard Gibbs) and
# ${tissue}_grid_gwfm_* (GWFM-style tempered Gibbs)
grid_label="_grid"
grid_gwfm_label="_grid_gwfm"


#####################
# 0. Assign every genotyped GTEx individual to a cross-validation fold (once, shared by all tissues)
#####################
if false; then
mkdir -p ${cv_folds_dir}
sh generate_cross_validation_folds.sh ${gtex_plink_stem}"1.fam" ${n_folds} ${fold_seed} ${fold_file}
fi



#####################
# Tissue-specific settings
#####################
tissue="Whole_Blood"
gibbs_input_summary_file=${gibbs_input_dir}${tissue}"_gibbs_input_summary.txt"


#####################
# 1. Prepare per-chromosome genotype and expression input for the tissue (fold-agnostic)
#####################
if false; then
expression_file=${gtex_expression_dir}${tissue}".v8.residualized_expression_renormalized.bed"
genotype_sample_mapping_file=${gtex_genotype_dir}"genotype_sample_mapping_to_"${tissue}"_expression_samples.txt"
sbatch prepare_gtex_tissue_data_for_gibbs.sh ${tissue} ${expression_file} ${gtex_plink_stem} ${genotype_sample_mapping_file} ${gtex_pc_genes_gtf} ${cis_window} ${gibbs_input_dir}
fi


#####################
# 2. Fit the spike and two slab Gibbs sampler, training on all folds but one (one job per held-out fold)
#####################
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin}
done
fi

#####################
# 2b. Same as step 2 with the tempered Gibbs sampler (several times the time per iteration, plus about 10 GB of LD neighbour lists in memory).
# Genes run in parallel, so give it cores, e.g. sbatch -c 8 -t 2-00:00 --mem=60G run_eqtl_mixture_prior_gibbs.sh ... tempered
#####################
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${tempered_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} tempered
done
fi

#####################
# 2c. Same as step 2 with the GWFM-style tempered Gibbs sampler (close to the standard sampler's time and memory)
#####################
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${gwfm_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} tempered_gwfm
done
fi

#####################
# 2d. Fit the fixed-grid mixture prior with standard Gibbs (about 10% more time than step 2)
#####################
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin}
done
fi



#####################
# 2e. Fit the fixed-grid mixture prior with the GWFM-style tempered Gibbs sampler
#####################
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm
done
fi






#####################
# 3. Evaluate predictions in each held-out fold (after the fits of step 2 have finished)
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi


#####################
# 3b. Same as step 3 for the tempered fits of step 2b
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${tempered_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${tempered_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi

#####################
# 3c. Same as step 3 for the GWFM-style fits of step 2c
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${gwfm_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${gwfm_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi

#####################
# 3d. Same as step 3 for the fixed-grid fits of step 2d
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi

#####################
# 3e. Same as step 3 for the fixed-grid GWFM-style fits of step 2e
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi



#####################
# 4. Organize results across folds and make pdf plots (after the evaluations of step 3 have finished)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi


#####################
# 4b. Same as step 4 for the tempered fits (the file prefix ${tissue}_tempered selects them; pdfs are named ${tissue}_tempered_*)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${tempered_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi

#####################
# 4c. Same as step 4 for the GWFM-style fits (pdfs named ${tissue}_tempered_gwfm_*)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${gwfm_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi

#####################
# 4d. Same as step 4 for the fixed-grid fits (pdfs named ${tissue}_grid_*; the hyper-parameter plot shows the collapsed small / large groups,
# the per-component weights are in ${gibbs_results_dir}${tissue}_grid_fold_*_grid_weight_summary.txt)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${grid_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi

#####################
# 4e. Same as step 4 for the fixed-grid GWFM-style fits (pdfs named ${tissue}_grid_gwfm_*)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${grid_gwfm_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi

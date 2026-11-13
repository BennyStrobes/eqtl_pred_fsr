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
n_iters="15000"
burn_in="5000"
thin="5"
fold_file=${cv_folds_dir}"gtex_individuals_"${n_folds}"_folds_seed_"${fold_seed}".txt"
# The six models: three priors (spike and two slab; fixed grid; fixed grid with 5 latent gene classes), each fit with standard Gibbs and with the
# GWFM-style tempered Gibbs sampler (README.md, sections 5c, 5e and 5j). Results are named ${tissue}${model_label}_*
model_labels="_spike_slab_gibbs _spike_slab_gwfm _grid_gibbs _grid_gwfm _grid_classes_gibbs _grid_classes_gwfm"
gene_class_model_labels="_grid_classes_gibbs _grid_classes_gwfm"
# One tissue per line (made in step 0b)
tissue_list_file=${output_root}"gtex_tissue_list.txt"


#####################
# 0. Assign every genotyped GTEx individual to a cross-validation fold (once, shared by all tissues)
#####################
if false; then
mkdir -p ${cv_folds_dir}
sh generate_cross_validation_folds.sh ${gtex_plink_stem}"1.fam" ${n_folds} ${fold_seed} ${fold_file}
fi


#####################
# 0b. Save the list of tissues with residualized expression (once; delete lines from the file to run a subset of tissues)
#####################
if false; then
mkdir -p ${output_root}
for expression_file in ${gtex_expression_dir}*.v8.residualized_expression_renormalized.bed; do
	basename ${expression_file} .v8.residualized_expression_renormalized.bed
done > ${tissue_list_file}
fi





#####################
# 1. Prepare per-chromosome genotype and expression input for every tissue in the list of step 0b (fold-agnostic, one job per tissue)
#####################
if false; then
for gtex_tissue in $(cat ${tissue_list_file}); do
	expression_file=${gtex_expression_dir}${gtex_tissue}".v8.residualized_expression_renormalized.bed"
	genotype_sample_mapping_file=${gtex_genotype_dir}"genotype_sample_mapping_to_"${gtex_tissue}"_expression_samples.txt"
	sbatch prepare_gtex_tissue_data_for_gibbs.sh ${gtex_tissue} ${expression_file} ${gtex_plink_stem} ${genotype_sample_mapping_file} ${gtex_pc_genes_gtf} ${cis_window} ${gibbs_input_dir}
done
fi



#####################
# 2. Fit the six models, training on all folds but one (one job per model and held-out fold): the spike and two slab prior, the fixed-grid prior (one prior
# shared by every gene) and the fixed-grid prior with 5 latent gene classes (with the non-centred class moves), each with standard Gibbs and with GWFM
#####################
tissue="Whole_Blood"
gibbs_input_summary_file=${gibbs_input_dir}${tissue}"_gibbs_input_summary.txt"
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_spike_slab_gibbs_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gibbs
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_spike_slab_gwfm_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_grid_gibbs_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gibbs
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_grid_gwfm_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_grid_classes_gibbs_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gibbs gene_classes_5
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_grid_classes_gwfm_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm gene_classes_5
done
fi


#####################
# 3. Evaluate predictions of every model in each held-out fold (after the fits of step 2 have finished), with y_hat and the expected quantities from
# separate blocks of retained samples (README.md, section 7)
#####################
if false; then
mkdir -p ${held_out_evaluation_dir}
for model_label in ${model_labels}; do
	for held_out_fold in $(seq 1 ${n_folds}); do
		sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${model_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${model_label}"_fold_"${held_out_fold}"_held_out_evaluation_split_samples.txt" split_samples
	done
done
fi


#####################
# 4. For every model, organize results across folds and make pdf plots, named ${tissue}${model_label}_split_samples_* (after the evaluations of step 3 have finished)
#####################
if false; then
for model_label in ${model_labels}; do
	sbatch organize_and_visualize_cross_validation_results.sh ${tissue}${model_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir} _split_samples
done
fi




#####################
# 4b. Plots of the gene classes of the two gene-class models (README.md, section 8, plot 9; they join the organized per-gene file, so they run after step 4)
#####################
if false; then
for model_label in ${gene_class_model_labels}; do
	sbatch visualize_gene_classes.sh ${tissue}${model_label} ${n_folds} ${gibbs_results_dir} ${visualization_dir} _split_samples
done
fi

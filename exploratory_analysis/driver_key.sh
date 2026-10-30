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
fold_file=${cv_folds_dir}"gtex_individuals_"${n_folds}"_folds_seed_"${fold_seed}".txt"


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
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}"_fold_"${held_out_fold}
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
# 4. Organize results across folds and make pdf plots (after the evaluations of step 3 have finished)
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
fi

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


#####################
# 5. Self-consistency check of the shared prior (README.md, section 5f), fold 1 of each finished fit: does one posterior sample per gene, pooled
# over genes, look like draws from the fitted prior? Over-dispersion of the per-gene number of large effects means genes are more heterogeneous
# than a shared prior allows. Results in ${prior_check_dir}${tissue}${label}_fold_1_prior_check_*
#####################
if false; then
prior_check_dir=${output_root}"prior_check/"
mkdir -p ${prior_check_dir}
for label in "" ${gwfm_label} ${grid_label}; do
	sbatch check_prior_self_consistency.sh ${gibbs_results_dir}${tissue}${label}"_fold_1" ${prior_check_dir}${tissue}${label}"_fold_1_prior_check"
done
fi

#####################
# 5b. Same as step 5 with the MHC genes (TSS in chr6:28,510,120-33,480,577, GRCh38) left out of the comparison: are they the source of the
# over-dispersion? Results in ${prior_check_dir}${tissue}${label}_fold_1_prior_check_no_mhc_*
#####################
if false; then
prior_check_dir=${output_root}"prior_check/"
mkdir -p ${prior_check_dir}
for label in "" ${gwfm_label} ${grid_label}; do
	sbatch check_prior_self_consistency.sh ${gibbs_results_dir}${tissue}${label}"_fold_1" ${prior_check_dir}${tissue}${label}"_fold_1_prior_check_no_mhc" 6:28510120-33480577
done
fi


#####################
# 6. Expression FSR check (README.md, section 7): re-evaluate with y_hat and the posterior predictive draws taken from separate blocks of retained
# samples (--split_samples), so the predictive draws do not share the Monte Carlo error of y_hat. Results in
# ${held_out_evaluation_dir}${tissue}${label}_fold_*_held_out_evaluation_split_samples.txt. (Re-running step 3e also adds y_hat_mc_variance to
# the default evaluation, which the by-Monte-Carlo-error plot of step 4 needs.)
#####################
if false; then
for label in ${grid_gwfm_label}; do
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${label}"_fold_"${held_out_fold}"_held_out_evaluation_split_samples.txt" split_samples
done
done
fi


#####################
# 6b. Organize and plot the split-sample evaluations of step 6 (pdfs named ${tissue}${label}_split_samples_*)
#####################
if false; then
for label in ${grid_gwfm_label}; do
	sh organize_and_visualize_cross_validation_results.sh ${tissue}${label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir} _split_samples
done
fi

#####################
# 7. Grid GWFM fit with an inclusion probability per bin of the number of cis variants (README.md, section 5g): the shared prior gives every
# variant the same inclusion probability, so a gene's prior number of eQTLs grows linearly with its number of cis variants, and expression FSR
# was over-confident for genes with many variants (+0.05 above 600 variants). Results named ${tissue}${grid_gwfm_bins_label}_*
#####################
n_variant_bins="20"
grid_gwfm_bins_label="_grid_gwfm_bins"${n_variant_bins}
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_bins_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm ${n_variant_bins}
done
fi

#####################
# 7b. Evaluate the fits of step 7 in each held-out fold
#####################
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_bins_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_gwfm_bins_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi


#####################
# 7c. Organize and plot the evaluations of step 7b, then the expression FSR excess by number of cis variants (compare with ${tissue}${grid_gwfm_label})
#####################
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${grid_gwfm_bins_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${grid_gwfm_bins_label}"_cv_organized_per_gene_fold.txt"
fi


#####################
# 7d. Prior self-consistency check (step 5) of fold 1 of the fit of step 7: the slope of K_g on its expected count should move to about 1
#####################
if false; then
prior_check_dir=${output_root}"prior_check/"
mkdir -p ${prior_check_dir}
sbatch check_prior_self_consistency.sh ${gibbs_results_dir}${tissue}${grid_gwfm_bins_label}"_fold_1" ${prior_check_dir}${tissue}${grid_gwfm_bins_label}"_fold_1_prior_check"
fi


#####################
# 8. Do the two samplers of the grid prior agree gene by gene? (compare_sampler_fits.py; README.md, section 9, step 4''): a ratio to the Monte
# Carlo noise that grows with n_cis_variants would mean poor mixing in genes with many correlated variants. Needs y_hat_mc_variance in at least
# one of the two evaluations (rerun step 3 / 4 of a fit evaluated before it was added)
#####################
if false; then
python compare_sampler_fits.py --per_gene_fold_file_a ${visualization_dir}${tissue}${grid_gwfm_label}"_cv_organized_per_gene_fold.txt" --per_gene_fold_file_b ${visualization_dir}${tissue}${grid_label}"_cv_organized_per_gene_fold.txt"
fi


#####################
# 9. Prior by variant annotation (README.md, section 5h): posterior against prior rates of large effects and beta^2 by MAF, TSS distance and
# in-window LD score, for each fold of the grid GWFM fit; also writes each gene's variant composition
#####################
annotation_check_dir=${output_root}"prior_check/"
if false; then
mkdir -p ${annotation_check_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch check_prior_by_variant_annotation.sh ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold} ${annotation_check_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold}"_annotation_check" ${gibbs_input_summary_file} ${fold_file} ${held_out_fold}
done
fi


#####################
# 9b. The expression FSR excess regression of step 7c with the variant composition of each window (from fold 1 of step 9): does it take over
# the log10 n_cis_variants coefficient?
#####################
if false; then
python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${grid_gwfm_label}"_cv_organized_per_gene_fold.txt" --gene_composition_file ${output_root}"prior_check/"${tissue}${grid_gwfm_label}"_fold_1_annotation_check_gene_composition.txt"
fi
#####################
# 10. Gene-pair null (README.md, section 5i): the held-out prediction of each gene correlated with the held-out expression of every gene on
# another chromosome, for each fold of the grid GWFM fit. (n_test - 1) r^2 averages 1 if the held-out individuals are exchangeable
#####################
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_gene_pair_null.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold}"_gene_pair_null.txt"
done
fi


#####################
# 10b. Summary of step 10, joined with the per-gene evaluation: the null dispersion by n_cis_variants and training LRT p-value, and the
# expression FSR excess it predicts against the observed excess
#####################
if false; then
python check_gene_pair_null.py --per_gene_fold_file ${visualization_dir}${tissue}${grid_gwfm_label}"_cv_organized_per_gene_fold.txt" --gene_pair_null_files $(for held_out_fold in $(seq 1 ${n_folds}); do echo ${held_out_evaluation_dir}${tissue}${grid_gwfm_label}"_fold_"${held_out_fold}"_gene_pair_null.txt"; done)
fi

#####################
# 11. Grid GWFM fits with structured priors (README.md, section 5j): grid weights per bin of the variant's MAF (_grid_gwfm_maf), per bin of its
# distance to the TSS (_grid_gwfm_tss_distance), or per latent gene class (10 classes, _grid_gwfm_gene_classes_10)
#####################
structured_priors="maf tss_distance gene_classes_10"
if false; then
mkdir -p ${gibbs_results_dir}
for prior in ${structured_priors}; do
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_"${prior}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm ${prior}
done
done
fi

#####################
# 11b. Evaluate the fits of step 11 in each held-out fold
#####################
if false; then
for prior in ${structured_priors}; do
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_gwfm_label}"_"${prior}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_gwfm_label}"_"${prior}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
done
fi

#####################
# 11c. Organize and plot the evaluations of step 11b, then the expression FSR excess of each (compare with ${tissue}${grid_gwfm_label}: mid-range
# excess +0.022, log10 n_cis_variants coefficient 0.10)
#####################
if false; then
for prior in ${structured_priors}; do
	sh organize_and_visualize_cross_validation_results.sh ${tissue}${grid_gwfm_label}"_"${prior} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
	echo "== ${prior}"
	python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${grid_gwfm_label}"_"${prior}"_cv_organized_per_gene_fold.txt"
done
fi

#####################
# 12. Standard sampler (step 2's model and speed) with the hyper-parameters started where the tempered fit of step 2b ended (fold 1's last
# iteration, for every fold): the tempered chain was still drifting toward a more polygenic prior (1 - pi0 from 0.01 to 0.03) after 10,000
# iterations and is the best calibrated fit. If this chain stays near the tempered end point (and is as well calibrated), the standard
# samplers were stuck in a too-sparse prior; if it drifts back, the tempered sampler's windowed hyper-parameter update is what moved it.
# Results named ${tissue}_warm_*
#####################
warm_label="_warm"
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_mixture_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${warm_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} standard ${gibbs_results_dir}${tissue}${tempered_label}"_fold_1_hyperparameter_trace.txt"
done
fi

#####################
# 12b. Evaluate the fits of step 12, then organize, plot and run the expression FSR excess check
#####################
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${warm_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${warm_label}"_fold_"${held_out_fold}"_held_out_evaluation.txt"
done
fi
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${warm_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir}
python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${warm_label}"_cv_organized_per_gene_fold.txt"
fi

#####################
# 13. The setup that was calibrated in simulations (README.md, section 5k): 10 latent gene classes (with the non-centred class moves), the
# learned residual-variance prior, the GWFM sampler, and the split-sample evaluation (y_hat and the expected quantities from separate halves of
# the 1,000 retained samples). Results named ${tissue}_grid_gwfm_classes_hier_*. Compare with ${tissue}${grid_gwfm_label} (mid-range expression
# FSR excess +0.022, log10 n_cis_variants coefficient 0.10)
#####################
classes_hier_label="_grid_gwfm_classes_hier"
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${classes_hier_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} gwfm gene_classes_10 0 hierarchical
done
fi

#####################
# 13b. Evaluate the fits of step 13 with split samples, then organize, plot and run the expression FSR excess check
#####################
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${classes_hier_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${classes_hier_label}"_fold_"${held_out_fold}"_held_out_evaluation_split_samples.txt" split_samples
done
fi
if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${classes_hier_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir} _split_samples
python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${classes_hier_label}"_split_samples_cv_organized_per_gene_fold.txt"
fi

#####################
# 14. As step 13, but with the standard Gibbs sampler and the fixed residual-variance prior: 10 latent gene classes (with the non-centred class
# moves), the grid prior. In simulations (README.md, section 5k) this cell took the expression FSR excess from +0.03 to +0.00 to +0.01, the same
# as the GWFM / hierarchical variants; the sampler changes PIPs, not predictions. Results named ${tissue}_grid_classes_*. Compare with
# ${tissue}${grid_label} (the same sampler without classes) and ${tissue}${grid_gwfm_label} (mid-range excess +0.022, log10 n_cis_variants
# coefficient 0.10). The class move costs about one sweep per iteration, so 4 cores (genes run in parallel)
#####################
grid_classes_label="_grid_classes"
if false; then
mkdir -p ${gibbs_results_dir}
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch -c 4 run_eqtl_grid_prior_gibbs.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_classes_label}"_fold_"${held_out_fold} ${n_iters} ${burn_in} ${thin} standard gene_classes_10 0 fixed
done
fi

#####################
# 14b. Evaluate the fits of step 14 with split samples, then organize, plot and run the expression FSR excess check
#####################
if false; then
for held_out_fold in $(seq 1 ${n_folds}); do
	sbatch evaluate_held_out_expression_prediction.sh ${gibbs_input_summary_file} ${fold_file} ${held_out_fold} ${gibbs_results_dir}${tissue}${grid_classes_label}"_fold_"${held_out_fold} ${held_out_evaluation_dir}${tissue}${grid_classes_label}"_fold_"${held_out_fold}"_held_out_evaluation_split_samples.txt" split_samples
done
fi

if false; then
sh organize_and_visualize_cross_validation_results.sh ${tissue}${grid_classes_label} ${n_folds} ${gibbs_results_dir} ${held_out_evaluation_dir} ${visualization_dir} _split_samples

python check_expression_fsr_excess.py --per_gene_fold_file ${visualization_dir}${tissue}${grid_classes_label}"_split_samples_cv_organized_per_gene_fold.txt"
fi

#####################
# 14c. What the gene classes of step 14 are, and how calibration varies with them (README.md, section 8, plot 9). Needs the organized per-gene file
# of step 14b. pdfs named ${tissue}${grid_classes_label}_split_samples_gene_classes_*, and the excess by class, number of cis variants and training
# evidence in ${tissue}${grid_classes_label}_split_samples_gene_classes_expression_fsr_excess.txt
#####################
if false; then
sbatch visualize_gene_classes.sh ${tissue}${grid_classes_label} ${n_folds} ${gibbs_results_dir} ${visualization_dir} _split_samples
fi

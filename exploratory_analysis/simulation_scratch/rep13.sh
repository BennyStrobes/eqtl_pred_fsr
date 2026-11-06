#!/bin/bash
# Factorial: prior structure (shared / 10 classes, plain Gibbs / 10 classes + class move) x residual-variance prior (fixed / hierarchical),
# on two independent 40%-null-gene datasets, plus the class-move cells on the correct-model dataset; then PIP and cis-h2 checks of every fit
S=/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis
PY="../venv/bin/python"
cd "$(dirname $0)"
if [ ! -f gene_classes_s13/truth.npz ]; then
	mkdir -p gene_classes_s13
	$PY -W ignore $S/simulate_gibbs_input.py --output_dir gene_classes_s13/ --scenario gene_classes --n_chromosomes 6 --seed 13 > gene_classes_s13/simulate.log
fi
run() { # dataset, name, options
	d=$1; v=$2; shift 2
	stem=$d/long_$v
	if [ ! -f ${stem}_eval.txt ]; then
		$PY -W ignore $S/run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --output_stem $stem --n_iters 6000 --burn_in 1000 --thin 5 "$@" > $stem.log 2>&1
		$PY -W ignore $S/evaluate_held_out_expression_prediction.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --gibbs_results_stem $stem --output_file ${stem}_eval.txt --block_size 50 --split_samples > ${stem}_eval.log 2>&1
	fi
	$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $stem --label "$d $v" >> factorial.txt
	$PY -W ignore pip_calibration.py --sim_dir $d/ --stem $stem --label "$d $v" >> factorial_pip.txt
	$PY -W ignore h2_check.py --sim_dir $d/ --stem $stem --label "$d $v" >> factorial_h2.txt
}
run gene_classes_s13 base
run gene_classes_s13 classes_plain --n_gene_classes 10 --no_gene_class_moves
run gene_classes_s13 classes_move --n_gene_classes 10
run gene_classes_s13 classes_plain_hier --n_gene_classes 10 --no_gene_class_moves --hierarchical_sigma2_e
run gene_classes_s13 classes_hier --n_gene_classes 10 --hierarchical_sigma2_e
echo finished >> factorial.txt

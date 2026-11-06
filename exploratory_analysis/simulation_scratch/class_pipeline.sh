#!/bin/bash
S=/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis
PY="../venv/bin/python"
cd "$(dirname $0)"
while ! grep -q finished_hier ncp_pipeline.txt 2>/dev/null; do sleep 30; done
for d in gene_classes shared; do
	stem=$d/long_classes_hier
	$PY -W ignore $S/run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --output_stem $stem --n_iters 6000 --burn_in 1000 --thin 5 --n_gene_classes 10 --hierarchical_sigma2_e > $stem.log 2>&1
	$PY -W ignore $S/evaluate_held_out_expression_prediction.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --gibbs_results_stem $stem --output_file ${stem}_eval.txt --block_size 50 --split_samples > ${stem}_eval.log 2>&1
	$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $stem --label "truth=$d classes10+hier (6000 it, split)" >> ncp_pipeline.txt
done
echo finished_classes >> ncp_pipeline.txt

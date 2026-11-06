#!/bin/bash
# Shared truth, shared prior, no LD (independent variants): single-site Gibbs mixes essentially perfectly
S=/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis
PY="../venv/bin/python"
cd "$(dirname $0)"
while ! grep -q finished diagnose.txt 2>/dev/null; do sleep 20; done
d=shared_ld0; mkdir -p $d
$PY -W ignore $S/simulate_gibbs_input.py --output_dir $d/ --scenario shared --n_chromosomes 6 --seed 11 --ld_rho 0.0 > $d/simulate.log
$PY -W ignore $S/run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --output_stem $d/fit_shared --n_iters 3000 --burn_in 1500 --thin 5 > $d/fit_shared.log 2>&1
$PY -W ignore $S/evaluate_held_out_expression_prediction.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --gibbs_results_stem $d/fit_shared --output_file $d/fit_shared_eval.txt --block_size 25 > $d/fit_shared_eval.log 2>&1
$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $d/fit_shared --label "E ld_rho 0" >> diagnose.txt
echo finished2 >> diagnose.txt

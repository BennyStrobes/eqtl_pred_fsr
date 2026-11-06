#!/bin/bash
# Shared truth, shared prior: where does the over-confidence under the correct model come from?
S=/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis
PY="../venv/bin/python"
cd "$(dirname $0)"
d=shared
fit() { # stem, extra args
	$PY -W ignore $S/run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --output_stem $1 "${@:2}" > $1.log 2>&1
}
evaluate() { # stem, block size, extra
	$PY -W ignore $S/evaluate_held_out_expression_prediction.py --gibbs_input_summary_file $d/gibbs_input_summary.txt --fold_file $d/folds.txt --held_out_fold 1 --gibbs_results_stem $1 --output_file $1_eval.txt --block_size $2 "${@:3}" > $1_eval.log 2>&1
}
# D. split-sample evaluation of the existing fit (Monte Carlo coupling)
mkdir -p $d/split; cp $d/fit_shared_posterior_beta_samples.npz $d/split/fit_posterior_beta_samples.npz
evaluate $d/split/fit 25 --split_samples
$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $d/split/fit --label "D split samples" >> diagnose.txt
# B. GWFM sampler
fit $d/fit_gwfm --n_iters 3000 --burn_in 1500 --thin 5 --gwfm; evaluate $d/fit_gwfm 25
$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $d/fit_gwfm --label "B gwfm" >> diagnose.txt
# A. 4x longer chain
fit $d/fit_long --n_iters 12000 --burn_in 3000 --thin 30; evaluate $d/fit_long 25
$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $d/fit_long --label "A 12000 iterations" >> diagnose.txt
# C. weaker LD (new data, same scenario)
d=shared_ld08; mkdir -p $d
$PY -W ignore $S/simulate_gibbs_input.py --output_dir $d/ --scenario shared --n_chromosomes 6 --seed 11 --ld_rho 0.8 > $d/simulate.log
fit $d/fit_shared --n_iters 3000 --burn_in 1500 --thin 5; evaluate $d/fit_shared 25
$PY -W ignore sim_metrics.py --sim_dir $d/ --stem $d/fit_shared --label "C ld_rho 0.8" >> diagnose.txt
echo finished >> diagnose.txt

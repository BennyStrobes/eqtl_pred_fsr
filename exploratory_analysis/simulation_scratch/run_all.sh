#!/bin/bash
# 4 simulated scenarios x 4 priors: fit (held-out fold 1), evaluate, metrics
S=/Users/benstrober/Library/CloudStorage/Dropbox/Research/code/eqtl_pred_fsr/exploratory_analysis
PY="$(dirname $0)/../venv/bin/python -W ignore"
cd "$(dirname $0)"
for scenario in shared maf tss_distance gene_classes; do
	mkdir -p ${scenario}
	if [ ! -f ${scenario}/truth.npz ]; then
		$PY $S/simulate_gibbs_input.py --output_dir ${scenario}/ --scenario ${scenario} --n_chromosomes 6 --seed 11 > ${scenario}/simulate.log
	fi
	for prior in shared maf tss_distance gene_classes; do
		case ${prior} in
			shared) opt="" ;;
			maf) opt="--variant_categories maf" ;;
			tss_distance) opt="--variant_categories tss_distance" ;;
			gene_classes) opt="--n_gene_classes 10" ;;
		esac
		stem=${scenario}/fit_${prior}
		if [ ! -f ${stem}_eval.txt ]; then
			$PY $S/run_eqtl_grid_prior_gibbs.py --gibbs_input_summary_file ${scenario}/gibbs_input_summary.txt --fold_file ${scenario}/folds.txt --held_out_fold 1 --output_stem ${stem} --n_iters 3000 --burn_in 1500 --thin 5 ${opt} > ${stem}.log 2>&1
			$PY $S/evaluate_held_out_expression_prediction.py --gibbs_input_summary_file ${scenario}/gibbs_input_summary.txt --fold_file ${scenario}/folds.txt --held_out_fold 1 --gibbs_results_stem ${stem} --output_file ${stem}_eval.txt --block_size 25 > ${stem}_eval.log 2>&1
		fi
		$PY sim_metrics.py --sim_dir ${scenario}/ --stem ${stem} --label "truth=${scenario} prior=${prior}" >> metrics.txt
	done
done
echo finished >> metrics.txt
